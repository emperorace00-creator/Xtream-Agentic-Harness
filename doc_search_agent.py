from utils import robust_rmtree, _embed, _embed_batched, _cosine_similarity, save_json_atomic, console
from renderer import _build_master_enhanced, _copy_and_enhance, _slice_master_into_tiles
# doc_search_agent.py - Semantic document search via chunk embeddings
"""
Two responsibilities:

1. INGEST TIME — chunk_and_embed(txt_path):
   Called by PDFIngestAgent after saving the OCR'd .txt to scratch.
   Splits the text into overlapping ~400-token chunks, embeds each chunk
   via nvidia/llama-nemotron-embed-1b-v2, and saves a <name>.chunks.json alongside
   the .txt in scratch.

2. QUERY TIME — search(query, top_k):
   Called by _handle_doc_search in tool_handlers.py.
   Embeds the query (input_type="query"), loads all .chunks.json files
   from scratch, computes cosine similarity, returns the top-K passages
   with source file and line numbers.

Design decisions:
  - Same config.NVIDIA_BASE_URL + config.NVIDIA_API_KEY_FILE as everything else — zero
    new credentials or infrastructure.
  - nvidia/llama-nemotron-embed-1b-v2 uses separate input_type for query vs passage
    (bi-encoder), which is why it outperforms generic embedders for retrieval.
  - Chunks stored as plain JSON — no vector DB dependency.
  - Chunking is paragraph-first, then character-limit fallback (~1600 chars
    ≈ 400 tokens at 4 chars/token), with ~200-char overlap so context is
    never lost at chunk boundaries.
  - Embeddings are batched (up to 32 per API call) to stay under rate limits.
  - numpy used only for cosine similarity — it's already a transitive dep.
"""

import os
import json
import time
import shutil
import tempfile
import hashlib
import numpy as np
from pathlib import Path
import config

# ── Chunk file suffix ─────────────────────────────────────────────────────────
CHUNKS_SUFFIX   = ".chunks.json"


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str) -> list:
    """
    Split text into overlapping chunks for embedding.

    Strategy:
      1. Walk through all paragraphs (double-newline split), tracking exact
         char offsets as we go so start_char/end_char are always correct.
      2. Accumulate paragraphs into a window until CHUNK_SIZE is reached.
      3. Emit the window, then seed the next window with the last CHUNK_OVERLAP
         chars of paras so context is never lost at boundaries.

    Returns:
        List of {"text": str, "start_char": int, "end_char": int}
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Build (stripped_para, char_offset) pairs in ONE pass - offsets always
    # match the actual text, no strip/filter desync possible.
    # Adjust `pos` by leading whitespace stripped by `.strip()` so char offsets
    # point at the actual first character.
    raw_paras  = text.split("\n\n")
    para_items = []
    pos = 0
    for raw in raw_paras:
        stripped = raw.strip()
        if stripped:
            leading_ws = len(raw) - len(raw.lstrip())
            para_items.append((stripped, pos + leading_ws))
        pos += len(raw) + 2   # +2 for the \n\n consumed by split

    chunks     = []
    window     = []      # list of (stripped_para_text, start_offset_in_full_text)
    window_len = 0

    for para, offset in para_items:
        if window_len + len(para) + 2 > config.EMBED_CHUNK_SIZE and window:
            # ── Emit current window ───────────────────────────────────────────
            chunk_text = "\n\n".join(p for p, _ in window)
            start_char = window[0][1]
            last_para_text, last_para_offset = window[-1]
            end_char = last_para_offset + len(last_para_text)

            chunks.append({
                "text":       chunk_text,
                "start_char": start_char,
                "end_char":   end_char,
            })

            # ── Seed next window with overlap ─────────────────────────────────
            # Walk backwards through window paras until we have >= CHUNK_OVERLAP
            # chars - use whole paragraphs so overlap text is always clean prose.
            overlap_len       = 0
            overlap_start_idx = len(window)
            for i in range(len(window) - 1, -1, -1):
                if len(window[i][0]) >= config.EMBED_CHUNK_SIZE:
                    overlap_start_idx = i + 1   # exclude the oversized paragraph from carry-forward
                    break
                overlap_len += len(window[i][0]) + 2
                overlap_start_idx = i
                if overlap_len >= config.EMBED_CHUNK_OVERLAP:
                    break

            window     = window[overlap_start_idx:]
            window_len = sum(len(p) + 2 for p, _ in window)

        window.append((para, offset))
        window_len += len(para) + 2

    # ── Flush remaining window ────────────────────────────────────────────────
    if window:
        chunk_text = "\n\n".join(p for p, _ in window)
        chunks.append({
            "text":       chunk_text,
            "start_char": window[0][1],
            "end_char":   len(text),
        })

    # Filter trivially short chunks (page headers, OCR artefacts)
    chunks = [c for c in chunks if len(c["text"]) >= config.EMBED_MIN_CHUNK_CHARS]
    return chunks
def _char_to_line(text: str, char_offset: int) -> int:
    """Convert a character offset to a 1-based line number."""
    return text[:char_offset].count("\n") + 1


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING API

# ══════════════════════════════════════════════════════════════════════════════
# DOC SEARCH AGENT
# ══════════════════════════════════════════════════════════════════════════════

class DocSearchAgent:
    """
    Manages chunk embeddings for all ingested documents and provides
    semantic search over them.

    Lifecycle:
      - chunk_and_embed() is called once per PDF at ingest time.
      - search() is called at query time from _handle_doc_search().
    """

    def __init__(self, scratch_dir: str = config.SCRATCH_DIR):
        self.scratch_dir = scratch_dir
        # ── Chunk cache: path → (mtime, list[chunk]) ─────────────────────────
        # Avoids re-reading and re-parsing the entire .chunks.json on every
        # doc_search call. Entry is invalidated only when the file's mtime
        # changes (i.e. after a new ingest_pdf run).
        self._chunks_cache: dict = {}

    # ── Ingest-time ───────────────────────────────────────────────────────────

    def chunk_and_embed(self, txt_path: str, source_hash: str = None) -> dict:
        """
        Chunk a .txt file and embed each chunk. Saves results as
        <stem>.chunks.json in the same directory as the .txt.

        Called by PDFIngestAgent.ingest() after OCR completes.

        Args:
            txt_path:    Absolute path to the OCR'd .txt file in scratch.
            source_hash: MD5 of the source PDF's bytes (if known). Stored as a
                         top-level field in .chunks.json so a later ingest_pdf
                         call can detect unchanged content and skip re-OCR'ing.
                         Never read by search() - purely a re-ingest guard.

        Returns:
            {"success": bool, "chunks": int, "chunks_path": str} or {"success": False, "error": str}
        """
        txt_path = os.path.abspath(txt_path)
        stem     = Path(txt_path).stem
        out_path = os.path.join(os.path.dirname(txt_path), f"{stem}{CHUNKS_SUFFIX}")

        console.print(f"\n🔢 [cyan]DocSearchAgent: chunking + embedding '{Path(txt_path).name}'[/cyan]")

        # Read the OCR'd text
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return {"success": False, "error": f"Could not read {txt_path}: {e}"}

        if not text.strip():
            return {"success": False, "error": "File is empty — nothing to embed."}

        # Bug #21 fix: if the 'text' is actually an OCR error string rather
        # than real document content, don't embed it - the error would become
        # a searchable chunk that matches future doc_search queries falsely.
        if text.strip().startswith("[OCR ERROR"):
            return {"success": False, "error": f"OCR failed for this document — not embedded: {text[:120]}"}

        # Bug #21 fix: if the 'text' is actually an OCR error string rather
        # than real document content, don't embed it - the error would become
        # a searchable chunk that matches future doc_search queries falsely.
        if text.strip().startswith("[OCR ERROR"):
            return {"success": False, "error": f"OCR failed for this document — not embedded: {text[:120]}"}

        # Chunk
        raw_chunks = _chunk_text(text)
        if not raw_chunks:
            return {"success": False, "error": "No chunks produced — file may be too short."}

        console.print(f"   {len(raw_chunks)} chunk(s) produced")

        # Embed all chunks (passage mode)
        chunk_texts  = [c["text"] for c in raw_chunks]
        embeddings   = _embed_batched(chunk_texts, input_type="passage")

        # Attach line numbers and embeddings
        rel_source = os.path.relpath(txt_path, self.scratch_dir).replace("\\", "/")
        chunks_out = []
        for i, (chunk, emb) in enumerate(zip(raw_chunks, embeddings)):
            if emb is None:
                continue
            start_line = _char_to_line(text, chunk["start_char"])
            end_line   = _char_to_line(text, min(chunk["end_char"], len(text) - 1))
            chunks_out.append({
                "id":         i,
                "source":     rel_source,          # relative path for portability
                "start_line": start_line,
                "end_line":   end_line,
                "text":       chunk["text"],
                "embedding":  emb,
            })

        # Bug #20 fix: if any chunk failed to embed, abort - don't write
        # source_hash to the sidecar. A partial index would block future re-ingest
        # attempts (the hash-match skip guard says "already indexed") while silently
        # returning fewer results than the document actually has.
        failed_count = sum(1 for e in embeddings if e is None)
        if failed_count:
            return {
                "success": False,
                "error":   (
                    f"{failed_count}/{len(embeddings)} chunk(s) failed to embed. "
                    "Not saving partial index — retry ingest to try again."
                ),
            }

        # Save atomically: write to .tmp then replace, so a crash never leaves
        # a corrupt .chunks.json that causes doc_search to silently lose the document.
        try:
            payload = {"source": rel_source, "chunks": chunks_out}
            if source_hash:
                payload["source_hash"] = source_hash
            save_json_atomic(out_path, payload)
            console.print(
                f"   ✅ [green]Saved {len(chunks_out)} chunks → "
                f"'{Path(out_path).name}'[/green]"
            )
        except Exception as e:
            return {"success": False, "error": f"Could not save chunks file: {e}"}

        return {
            "success":     True,
            "chunks":      len(chunks_out),
            "chunks_path": out_path,
        }

    # ── Query-time ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Semantic search across all .chunks.json files in scratch.

        Embeds the query (input_type="query"), scores all chunks via
        cosine similarity, returns the top-K passages as a formatted
        string ready to return as a tool result to the main model.

        Args:
            query:  The search query (constructed by the main model).
            top_k:  Number of passages to return.

        Returns:
            Formatted string with passages, sources, and line numbers.
        """
        console.print(f"\n📚 [cyan]DocSearch: '{query[:80]}'[/cyan]")

        # Discover all .chunks.json files in scratch
        chunks_files = self._find_chunks_files()
        if not chunks_files:
            return (
                "[SYSTEM: DOC SEARCH] No embedded documents found. "
                "Ingest a PDF first: <ingest_pdf>your_file.pdf</ingest_pdf>"
            )

        # Load all chunks - using mtime-keyed cache to avoid full JSON parse
        # on every query. Re-reads only chunks files that changed since last load.
        all_chunks = []
        for cf in chunks_files:
            try:
                mtime = os.path.getmtime(cf)
                cached = self._chunks_cache.get(cf)
                if cached and cached[0] == mtime:
                    # Cache hit - no disk I/O needed
                    all_chunks.extend(cached[1])
                else:
                    # Cache miss - read from disk and cache
                    with open(cf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chunks = data.get("chunks", [])
                    self._chunks_cache[cf] = (mtime, chunks)
                    all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"   [yellow]⚠️  Could not load {cf}: {e}[/yellow]")

        if not all_chunks:
            return "[SYSTEM: DOC SEARCH] Chunks files exist but contain no data."

        # Embed query (query mode - different projection from passage)
        try:
            query_emb = _embed([query], input_type="query")[0]
        except Exception as e:
            return f"[SYSTEM: DOC SEARCH ERROR] Failed to embed query: {e}"

        # Score all chunks
        # Guard against partial writes or schema mismatches -
        # skip any chunk that is missing its embedding vector.
        valid_chunks = [c for c in all_chunks if c.get("embedding")]
        if not valid_chunks:
            return "[SYSTEM: DOC SEARCH] Chunks found but none have embeddings. Re-ingest the PDF."
        passage_vecs = [c["embedding"] for c in valid_chunks]
        all_chunks   = valid_chunks   # keep in sync for result lookup
        scores       = _cosine_similarity(query_emb, passage_vecs)

        # Rank
        ranked_indices = np.argsort(scores)[::-1]

        # Bug 33: apply minimum similarity threshold to avoid flooding the
        # model's context with irrelevant passages on low-relevance queries.
        _MIN_SCORE = getattr(config, "DOC_SEARCH_MIN_SCORE", 0.15)
        ranked_indices = [
            idx for idx in ranked_indices
            if float(scores[idx]) >= _MIN_SCORE
        ][:top_k]

        if not ranked_indices:
            return (
                f"[SYSTEM: DOC SEARCH: '{query[:60]}'] — "
                "No sufficiently relevant passages found (all scores below threshold)."
            )

        # Format output
        lines = [
            f"[SYSTEM: DOC SEARCH: '{query[:60]}'] — top {min(top_k, len(ranked_indices))} passage(s)\n"
        ]
        for rank, idx in enumerate(ranked_indices, 1):
            chunk  = all_chunks[idx]
            score  = float(scores[idx])
            source = chunk.get("source", "unknown")
            s_line = chunk.get("start_line", "?")
            e_line = chunk.get("end_line", "?")
            text   = chunk.get("text", "").strip()

            # Show full chunk text - at 3200 chars (~800 tokens) each chunk is
            # self-contained and the model needs the full content to answer correctly.
            # 5 results * ~800 tokens = ~4000 tokens, well within context budget.

            lines.append(
                f"{'─' * 50}\n"
                f"📄 [{rank}] {source}  lines {s_line}–{e_line}  "
                f"(similarity: {score:.3f})\n\n"
                f"{text}\n"
            )

        lines.append(
            "[SYSTEM: Tip — use view_lines(file, start, end) to read more context around any passage.]"
        )
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_chunks_files(self) -> list:
        """Recursively find all .chunks.json files in scratch."""
        found = []
        if not os.path.isdir(self.scratch_dir):
            return found
        for root, _, files in os.walk(self.scratch_dir):
            for fn in files:
                if fn.endswith(CHUNKS_SUFFIX):
                    chunks_path = os.path.join(root, fn)

                    # Bug 1: don't guess the source extension as ".txt" - that
                    # breaks for .md files ingested via ingest_text (they keep
                    # their original extension, e.g. "notes.md"), causing this
                    # cleanup to delete valid, still-referenced chunks files.
                    # Instead trust the "source" field chunk_and_embed() already
                    # writes into the sidecar, which records the real relative
                    # path of the ingested file.
                    source_exists = None  # None = couldn't determine, don't delete
                    try:
                        with open(chunks_path, "r", encoding="utf-8") as _f:
                            _data = json.load(_f)
                        source_rel = _data.get("source")
                        if source_rel:
                            source_path = os.path.join(self.scratch_dir, source_rel)
                            source_exists = os.path.exists(source_path)
                        else:
                            # Older chunks file predating the "source" field -
                            # fall back to the legacy .txt guess, but only as
                            # a last resort, and never delete just because the
                            # field happens to be missing.
                            legacy_txt = chunks_path[:-len(CHUNKS_SUFFIX)] + ".txt"
                            source_exists = os.path.exists(legacy_txt) or None
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Could not verify source for {fn}: {e} — keeping it.[/yellow]")
                        source_exists = None

                    if source_exists is False:
                        try:
                            os.remove(chunks_path)
                            console.print(f"[dim]🗑️  Deleted orphaned chunks: {fn}[/dim]")
                        except Exception as e:
                            console.print(f"[yellow]⚠️  Failed to delete orphaned chunks {chunks_path}: {e}[/yellow]")
                    else:
                        # source_exists is True, or None (unknown/legacy) - keep it.
                        found.append(chunks_path)
        return found


# ══════════════════════════════════════════════════════════════════════════════
# PDF INGEST AGENT  (was pdf_ingest.py)
# ══════════════════════════════════════════════════════════════════════════════



class PDFIngestAgent:
    """
    Ingests a PDF file into the scratch workspace as a plain text file.

    Pipeline:
      1. pymupdf renders each page to a PNG image in a temp folder
      2. Each page image is enhanced and optionally tiled (mirrors the upload pipeline):
           - IMAGE_TILING=True  → _build_master_enhanced + _slice_master_into_tiles
             OCR runs on all tiles (overview + N×M tile grid) per page.
           - IMAGE_TILING=False → _copy_and_enhance (basic PIL enhance)
             OCR runs on the single enhanced image per page.
      3. Results are concatenated in page order and saved as <name>.txt in scratch

    Reuses the existing ImageOCRAgent - no new API calls, no new models.
    All NIM tier fallback logic is inherited.
    """

    def __init__(self, ocr_agent, scratch_dir: str):
        self.ocr        = ocr_agent
        self.scratch    = scratch_dir
        self.doc_search = DocSearchAgent(scratch_dir=scratch_dir)

    # ── Public entry point ───────────────────────────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str) -> dict:
        """
        Convert a PDF to a .txt file in scratch and register with workspace tracker.

        Args:
            pdf_path: Absolute path to the PDF in the uploads folder.

        Returns:
            dict with success, txt_path, pages, output_file, or error.
        """
        pdf_path = os.path.abspath(pdf_path)
        stem     = Path(pdf_path).stem          # e.g. "chapter3"
        out_name = f"{stem}.txt"
        out_path = os.path.join(self.scratch, out_name)

        console.print(f"\n📄 [cyan]PDF ingest: {Path(pdf_path).name}[/cyan]")

        # ── Step 0: MD5 skip guard ───────────────────────────────────────────────
        # The model explicitly controls when ingest_pdf is called - this does NOT
        # auto re-ingest on content change. It only saves a wasted 2+ minute OCR
        # run when the model calls ingest_pdf again on a file whose bytes are
        # identical to what's already ingested. If content differs (same
        # filename, new upload), ingest proceeds normally below.
        try:
            with open(pdf_path, "rb") as _f:
                pdf_hash = hashlib.md5(_f.read()).hexdigest()
        except Exception as e:
            return {"success": False, "error": f"Could not read PDF: {e}"}

        chunks_path = os.path.join(self.scratch, f"{stem}{CHUNKS_SUFFIX}")
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as _f:
                    existing = json.load(_f)
                # Bug 6: previously this only checked chunks_path and returned
                # skipped=True unconditionally, even if out_path (the .txt) had
                # been deleted separately. That left the model calling
                # view_lines on a nonexistent file, permanently stuck, since
                # skipped=True kept being returned. Now also verify the .txt
                # is actually present before taking the skip shortcut.
                if existing.get("source_hash") == pdf_hash and os.path.exists(out_path):
                    console.print(
                        f"   ⚡ [green]Already ingested, content unchanged — skipping OCR[/green]"
                    )
                    return {
                        "success": True,
                        "skipped": True,
                        "txt_path": out_path,
                        "output_file": out_name,
                        "message": "Already ingested (content unchanged). Use doc_search to query it.",
                    }
                elif existing.get("source_hash") == pdf_hash:
                    console.print(
                        f"   ⚠️  [yellow]Chunks exist but '{out_name}' is missing — re-ingesting[/yellow]"
                    )
                    # fall through to a fresh ingest below
            except Exception:
                pass  # Corrupt/partial chunks file - fall through to a fresh ingest

        # ── Step 1: open PDF ─────────────────────────────────────────────────────
        try:
            import fitz  # pymupdf
        except ImportError:
            return {"success": False, "error": "pymupdf not installed. Run: pip install pymupdf"}

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return {"success": False, "error": f"Could not open PDF: {e}"}

        total_pages = len(doc)
        console.print(
            f"   {total_pages} page(s) — rendering at 200 DPI + OCR (pipelined)..."
        )

        tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")

        # ── Steps 2 & 3 (pipelined): render pages while OCR'ing earlier ones ───
        #
        # Producer thread: renders pages at 200 DPI and puts (page_idx, path)
        # into a bounded queue as each page completes.
        #
        # Consumer (main thread): reads from the queue as pages arrive and submits
        # each one to the OCR thread pool immediately - OCR of page N overlaps
        # with rendering of pages N+1, N+2, ...
        #
        # The AIMD gate inside ImageOCRAgent controls actual API concurrency;
        # no fixed inter-batch sleep is needed.
        # ────────────────────────────────────────────────────────────────────────
        import queue as _q
        import concurrent.futures as _cf

        RENDER_LOOKAHEAD = 8          # max pages rendered ahead of OCR in-flight
        render_q    = _q.Queue(maxsize=RENDER_LOOKAHEAD)
        render_errs = []

        import threading as _threading
        cancel_event = _threading.Event()  # Bug 21: used to signal producer to stop

        def _render_pages():
            try:
                for i, page in enumerate(doc):
                    if cancel_event.is_set():  # Bug 21: bail early if consumer failed
                        break
                    # 200 DPI: ~78% more pixels than 150 DPI - better for dense
                    # math, small subscripts, and tables.
                    mat  = fitz.Matrix(200 / 72, 200 / 72)
                    pix  = page.get_pixmap(matrix=mat)
                    path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
                    pix.save(path)
                    # Bug 21: use timeout put so the producer never blocks
                    # forever if the consumer has already exited.
                    while True:
                        try:
                            render_q.put((i, path), timeout=1.0)
                            break
                        except _q.Full:
                            if cancel_event.is_set():
                                return
            except Exception as e:
                render_errs.append(e)
            finally:
                doc.close()
                # Bug 7 fix: thread deadlock on sentinel put
                while True:
                    try:
                        render_q.put(None, timeout=1.0)
                        break
                    except _q.Full:
                        if cancel_event.is_set():
                            break

        render_thread = _threading.Thread(target=_render_pages, daemon=True)
        render_thread.start()

        ordered_results = [None] * total_pages
        ocr_futures     = {}
        # Partial 1 fix: accumulate exact master/overview/tile paths written
        # during page prep, so _cleanup() can delete them directly instead of
        # glob-guessing by PDF stem (which never matches - see _cleanup docstring).
        written_image_files = []

        try:
            # Cap thread pool: the AIMD gate limits actual API concurrency to
            # MAX_CONCURRENCY. Spawning one thread per page wastes OS resources
            # on large PDFs without any throughput benefit.
            _max_workers = min(total_pages or 1, config.PDF_OCR_MAX_WORKERS)
            # Bug 10 fix: bounding inflight tasks to prevent unbounded queue from 
            # rendering all pages instantly, defeating RENDER_LOOKAHEAD limits.
            inflight_sem = _threading.Semaphore(_max_workers + 2)
            with _cf.ThreadPoolExecutor(max_workers=_max_workers) as ocr_pool:
                while True:
                    item = render_q.get()
                    if item is None:
                        break
                    
                    inflight_sem.acquire()
                    page_idx, img_path = item
                    # ── Apply IMAGE_TILING / IMAGE_ENHANCE (mirrors upload pipeline) ──
                    # All enhancement is done in the consumer thread (here) so the
                    # AIMD gate inside ImageOCRAgent still controls API concurrency.
                    ocr_input = self._prepare_page_images(img_path, page_idx, extra_files=written_image_files)
                    if getattr(config, "IMAGE_TILING", False):
                        # Tiling: all tiles (overview + crops) go to the model in
                        # ONE API call so it sees full context + zoomed detail at once
                        # and returns ONE unified transcription - mirrors vision path.
                        fut = ocr_pool.submit(self.ocr.process_image_group, ocr_input)
                    else:
                        fut = ocr_pool.submit(self.ocr.process_images, ocr_input)
                    
                    fut.add_done_callback(lambda _: inflight_sem.release())
                    ocr_futures[page_idx] = fut

                for page_idx in sorted(ocr_futures):
                    try:
                        page_result = ocr_futures[page_idx].result()
                        if page_result:
                            # process_image_group → single dict
                            # process_images      → list; take first element
                            if isinstance(page_result, list):
                                if page_result:
                                    ordered_results[page_idx] = page_result[0]
                            else:
                                ordered_results[page_idx] = page_result
                    except Exception as e:
                        console.print(
                            f"   [yellow]⚠️  Page {page_idx + 1} OCR failed: {e}[/yellow]"
                        )


        except Exception as e:
            # Bug 21: signal producer to stop and drain queue so it unblocks
            cancel_event.set()
            render_thread.join(timeout=5)
            self._cleanup(tmp_dir, stem=stem, extra_files=written_image_files)
            return {"success": False, "error": f"OCR pipeline failed: {e}"}

        render_thread.join()

        if render_errs:
            self._cleanup(tmp_dir, extra_files=written_image_files)
            return {"success": False, "error": f"Page rendering failed: {render_errs[0]}"}

        # ── Step 4: assemble and save ────────────────────────────────────────────
        written_files = []  # Bug 32: track files written so we can clean up partials on failure
        try:
            parts = []
            for page_num, r in enumerate(ordered_results, 1):
                if r is None:
                    continue
                markdown = r.get("markdown", "").strip()
                if markdown:
                    parts.append(f"--- Page {page_num} ---\n{markdown}")
            full_text = "\n\n".join(parts)

            # Bug 4: on total OCR failure (every page returned None or blank),
            # full_text is "". Previously this still wrote an empty .txt,
            # let chunk_and_embed fail silently (warning only), and returned
            # success=True regardless - reporting the PDF as ingested when
            # nothing was actually captured. Fail explicitly instead, before
            # writing anything to disk or attempting to embed.
            if not full_text.strip():
                self._cleanup(tmp_dir, stem=stem, extra_files=written_image_files)
                return {
                    "success": False,
                    "error": (
                        f"OCR produced no text from any of the {total_pages} page(s) — "
                        f"the PDF may be corrupt, password-protected, or all-image "
                        f"pages the OCR chain could not read."
                    ),
                }

            os.makedirs(self.scratch, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            written_files.append(out_path)  # Bug 32: track

            console.print(f"   ✅ [green]Saved: {out_name} ({len(full_text):,} chars)[/green]")

        except Exception as e:
            self._cleanup(tmp_dir, stem=stem, extra_files=written_image_files)
            # Bug 32: remove any partially-written output files
            for _wf in written_files:
                try:
                    if os.path.exists(_wf):
                        os.remove(_wf)
                except Exception:
                    pass
            return {"success": False, "error": f"Failed to write output: {e}"}

        self._cleanup(tmp_dir, stem=stem, extra_files=written_image_files)  # Bug 2 + Partial 1: also removes tile/overview/master images

        # ── Step 5: chunk + embed for semantic doc search ────────────────────────
        # Bug 32: also track chunks file for cleanup on failure
        embed_result = {"success": False, "chunks": 0}
        try:
            embed_result = self.doc_search.chunk_and_embed(out_path, source_hash=pdf_hash)
            if embed_result.get("success"):
                _cpath = os.path.join(self.scratch, f"{stem}{CHUNKS_SUFFIX}")
                written_files.append(_cpath)
            else:
                console.print(
                    f"   [yellow]⚠️  Embedding skipped: {embed_result.get('error')}[/yellow]"
                )
        except Exception as e:
            console.print(f"   [yellow]⚠️  Embedding step failed: {e}[/yellow]")

        return {
            "success":     True,
            "txt_path":    out_path,
            "output_file": out_name,
            "pages":       total_pages,
            "chars":       len(full_text),
            "chunks":      embed_result.get("chunks", 0),
            # Bug 4 (second layer): text extraction succeeding is not the same
            # as the file being searchable via doc_search - surface embedding
            # status separately instead of implying doc_search always works
            # whenever "success" is True.
            "embedded":    embed_result.get("success", False),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _prepare_page_images(self, img_path: str, page_idx: int, extra_files: list = None) -> str:
        """
        Apply IMAGE_TILING / IMAGE_ENHANCE to a rendered PDF page PNG and
        return a shlex-safe string of file paths ready for ocr_agent.process_images().

        Mirrors the logic in renderer.load_images_for_vision():
          - IMAGE_TILING=True  → master-enhance → tile → return all tile paths
          - IMAGE_TILING=False → copy-and-enhance → return single enhanced path

        Falls back silently to the raw rendered PNG on any error so the OCR
        pipeline never drops a page due to an enhancement failure.

        Partial 1 fix: if `extra_files` (a list) is passed, every
        master/overview/tile file actually written to SCRATCH_DIR is appended
        to it, so the caller can delete exactly those paths at cleanup time
        instead of glob-guessing by PDF stem (which can never match, since
        these filenames are built from the *page* stem, e.g. "page_0001",
        not the PDF stem, e.g. "textbook").
        """
        import shlex as _shlex
        filename = Path(img_path).name
        stem     = Path(img_path).stem

        if getattr(config, "IMAGE_TILING", False):
            try:
                master_path = _build_master_enhanced(img_path, filename)
                if extra_files is not None:
                    extra_files.append(master_path)
                tile_dicts  = _slice_master_into_tiles(master_path, stem)
                if tile_dicts:
                    # Tiles are already written to config.SCRATCH_DIR by
                    # _slice_master_into_tiles - reconstruct their file paths.
                    tile_paths = [
                        os.path.join(config.SCRATCH_DIR, d["filename"])
                        for d in tile_dicts
                        if os.path.isfile(os.path.join(config.SCRATCH_DIR, d["filename"]))
                    ]
                    if extra_files is not None:
                        extra_files.extend(tile_paths)
                    if tile_paths:
                        console.print(
                            f"   [dim cyan]PDF page {page_idx + 1}: tiled into "
                            f"{len(tile_paths)} image(s)[/dim cyan]"
                        )
                        return " ".join(f'"{p}"' for p in tile_paths)
                console.print(
                    f"   [dim yellow]PDF page {page_idx + 1}: tiling produced no tiles — using enhanced single image[/dim yellow]"
                )
                # Fall through to single enhanced image
                return f'"{master_path}"'
            except Exception as tile_err:
                console.print(
                    f"   [dim yellow]PDF page {page_idx + 1}: tiling failed ({tile_err}) — using original[/dim yellow]"
                )
                return f'"{img_path}"'
        else:
            # IMAGE_TILING=False: basic PIL enhancement (auto_enhance) only
            try:
                enhanced_path = _copy_and_enhance(img_path, filename)
                if extra_files is not None:
                    extra_files.append(enhanced_path)
                    # _copy_and_enhance() also writes a raw, un-enhanced copy
                    # to SCRATCH_DIR/<filename> before enhancing it - that
                    # raw copy would otherwise leak the same way the
                    # enhanced copy did. Track it too.
                    raw_scratch_copy = os.path.join(config.SCRATCH_DIR, filename)
                    extra_files.append(raw_scratch_copy)
                console.print(
                    f"   [dim]PDF page {page_idx + 1}: enhanced (PIL-only)[/dim]"
                )
                return f'"{enhanced_path}"'
            except Exception as enh_err:
                console.print(
                    f"   [dim yellow]PDF page {page_idx + 1}: enhancement failed ({enh_err}) — using original[/dim yellow]"
                )
                return f'"{img_path}"'

    def _cleanup(self, tmp_dir: str, stem: str = None, extra_files: list = None):
        """Remove temp image folder after OCR.
        If stem is provided, also glob-sweeps for tile/overview/master images
        (legacy/secondary mechanism - see Partial 1 note below).
        If extra_files is provided (list of exact paths written during this
        ingest, from _prepare_page_images), those are removed directly - this
        is the primary, reliable cleanup mechanism (Partial 1 fix)."""
        try:
            robust_rmtree(tmp_dir)
        except Exception:
            pass

        # Partial 1 fix: delete exactly the files we know we wrote. Tile/
        # master/overview filenames are built from the *page* stem
        # ("page_0001"), never the PDF stem ("textbook"), so the glob sweep
        # below can never match them - exact-path tracking is what actually
        # prevents the leak.
        if extra_files:
            for _f in extra_files:
                try:
                    if _f and os.path.exists(_f):
                        os.remove(_f)
                except Exception:
                    pass

        # Bug 2: secondary glob sweep - harmless, catches any stragglers left
        # over from ingests that ran before the extra_files tracking existed.
        # Kept as a best-effort safety net, not the primary mechanism.
        if stem:
            import glob as _glob
            for _pattern in [f"master_{stem}*", f"overview_{stem}*", f"tile_*_{stem}*",
                             f"master_*{stem}*", f"overview_*{stem}*", f"tile_*{stem}*"]:
                for _f in _glob.glob(os.path.join(config.SCRATCH_DIR, _pattern)):
                    try:
                        os.remove(_f)
                    except Exception:
                        pass
