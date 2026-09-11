# tool_handlers.py
# ToolHandlersMixin — all _handle_* methods, the dispatch table, and workspace
# scan helpers extracted from emperor_agent.py.
#
# EmperorAgent inherits this mixin so its external interface is unchanged.
# To add a new tool:
#   1. Add its definition to core_tool_definitions.py (or the appropriate *_tools.py file)
#   2. Add one entry to _build_dispatch() below
#   3. Write one _handle_* method here

import hashlib
import json
import os
import re
import shutil
import subprocess
import shlex
import threading
import requests
import config
from renderer import show_image_in_terminal
from utils import rerank_passages, compute_view_window, CODE_EXTENSIONS, BASH_BLOCKLIST, console, read_api_key_cached
# Shared path normalizer — used by view_lines and search_in_file
_norm = lambda p: os.path.normcase(os.path.normpath(p))


# Extensions show_image will attempt to preview via chafa. Kept separate from
# UPLOAD_EXTENSIONS below (that list is for reading uploaded files as text/
# code — these are for rendering as a raster image).
IMAGE_PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

# File extensions recognised as "uploadable" / "project" code files.
# Built on top of utils.CODE_EXTENSIONS (the shared code/text extension base
# also used by workspace_tracker's reconciler) plus the extra non-code types
# uploads can contain, so the core list only has to be maintained in one place.
UPLOAD_EXTENSIONS = CODE_EXTENSIONS | {
    ".env",
    ".pdf",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif", ".tiff", ".tif",
    ".csv", ".xml", ".xlsx", ".xls", ".parquet",
    ".zip", ".tar", ".gz",
}



_ALLOWED_READ_PREFIXES = None  # lazy-init to avoid import-order issues

def _is_allowed_read_path(filepath: str) -> bool:
    '''Check if an absolute path is within any allowed read directory.'''
    global _ALLOWED_READ_PREFIXES
    if _ALLOWED_READ_PREFIXES is None:
        _ALLOWED_READ_PREFIXES = [
            _norm(config.SCRATCH_DIR),
            _norm(config.UPLOADS_FOLDER),
            _norm(config.OUTPUTS_DIR),
            _norm(config.DATABASE_DIR),
            _norm(config.CHAT_HISTORIES_DIR),
        ]
    normed = _norm(filepath)
    return any(normed == prefix or normed.startswith(prefix + os.sep) for prefix in _ALLOWED_READ_PREFIXES)


def _smart_truncate(text: str, max_chars: int = 8000, head: int = 3000) -> str:
    """Keep the first `head` chars and the last `(max_chars - head)` chars of text.

    The tail is where errors and summaries live (pytest failures, compile errors,
    script exit summaries). A plain head-only cut would silently discard them.
    A bridge line shows how many characters were omitted.
    """
    if len(text) <= max_chars:
        return text
    tail = max_chars - head
    omitted = len(text) - max_chars
    bridge = f"\n[SYSTEM: {omitted:,} chars omitted — use view_lines for the full content] ...\n"
    return text[:head] + bridge + text[-tail:]


class ToolHandlersMixin:
    """
    Mixin that provides all tool-handler methods for EmperorAgent.

    Assumes the host class (EmperorAgent) has the following attributes:
        self.file_ops, self.workspace_tracker, self.doc_search_agent,
        self.search_history_agent, self.ocr_agent, self.summarizer
    """

    # ══════════════════════════════════════════════════════════════════════════
    # DISPATCH TABLE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_dispatch(self) -> dict:
        """
        Build the tool-name → handler-method dispatch table.
        Adding a new tool = one entry here + one private _handle_* method. No if/elif needed.
        """
        d = {
            # ── Core ──────────────────────────────────────────────────────────
            "quick_search":             self._handle_quick_search,
            "url_search":               self._handle_url_search,

            "search_history":           self._handle_search_history,
            "ingest_chat":              self._handle_ingest_chat,
            "workspace_search":         self._handle_workspace_search,
            "doc_search":               self._handle_doc_search,
            "ingest_pdf":               self._handle_ingest_pdf,
            "ingest_text":              self._handle_ingest_text,
            # ── Research ──────────────────────────────────────────────────────
            "search_semantic_scholar":  self._handle_search_semantic_scholar,
            # ── File ops ──────────────────────────────────────────────────────
            "str_replace":              self._handle_str_replace,
            "view_lines":               self._handle_view_lines,
            "search_in_file":           self._handle_search_in_file,
            # ── Bash ──────────────────────────────────────────────────────────
            "bash":                     self._handle_bash_exec,
            "show_image":               self._handle_show_image,
        }
        return d

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL DISPATCHER
    # ══════════════════════════════════════════════════════════════════════════

    def _call_tool(self, fn: str, args: dict, web_agent) -> str:
        """
        Dispatch a tool call to the appropriate private handler method.
        Note: `web_agent` is passed to *every* tool handler even though 90% of tools 
        don't use it. This is a deliberate design choice because `url_search` and 
        `quick_search` require the web agent instance, and keeping a uniform method 
        signature simplifies the dispatcher.

        Also triggers the lazy pre-turn scratch backup (item 4 / SP-1) on the
        first call — see _start_scratch_backup() in emperor_agent.py.
        """
        if self._active_groups:
            self._start_scratch_backup()
        handler = self._dispatch.get(fn)
        if handler:
            return handler(args, web_agent)
        return f"[SYSTEM: ERROR] Unknown tool: '{fn}'"

    # ══════════════════════════════════════════════════════════════════════════
    # CORE TOOL HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_search_history(self, args: dict, web_agent) -> str:
        """
        Search past chat sessions for turns relevant to a query.
        Parsing, incremental embedding, scoring, reranking, and formatting
        all live in SearchHistoryAgent (search_history_agent.py) — this
        handler just validates args and delegates.
        """
        query = args.get("query", "").strip()
        if not query:
            return "[SYSTEM: ERROR] search_history requires a 'query' argument."

        try:
            top_k = min(int(args.get("top_k", 5)), 10)
        except (TypeError, ValueError):
            top_k = 5

        return self.search_history_agent.search(query, top_k=top_k)

    def _handle_ingest_chat(self, args: dict, web_agent) -> str:
        """
        Import a raw chat transcript (pasted from Gemini/ChatGPT web — unstructured
        prose, no "You:"/"AI:" labels) into GLOBAL_HISTORIES_DIR as a synthetic
        two-line .jsonl, so search_history can find it via semantic similarity.

        The synthetic .jsonl mirrors exactly what _parse_turns() in
        _handle_search_history already understands: a "user" line (a stub
        carrying the filename for display) followed by an "assistant" line
        (the full pasted text, which is what actually gets embedded). This
        produces one turn-pair — no changes needed to the search_history
        pipeline itself.
        """
        # Accept both <ingest_chat>name.txt</ingest_chat> (falls through to the
        # generic {"query": ...} parser) and a future <filename> child tag.
        filename = (args.get('filename') or args.get('query') or '').strip()
        if not filename:
            return "[SYSTEM: ERROR] ingest_chat requires a filename, e.g. <ingest_chat>gemini_chat.txt</ingest_chat>"

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return f"[SYSTEM: ERROR] '{filename}' is a PDF — use ingest_pdf for PDFs, not ingest_chat."
        if ext not in ('.txt', '.md'):
            return f"[SYSTEM: ERROR] '{filename}' must be a plaintext file (.txt or .md)."

        # Resolve path: /uploads first, then scratch/
        src_path = os.path.join(config.UPLOADS_FOLDER, filename)
        if not os.path.isfile(src_path):
            src_path = os.path.join(config.SCRATCH_DIR, filename)
        if not os.path.isfile(src_path):
            return f"[SYSTEM: ERROR] '{filename}' not found in /uploads or scratch."

        try:
            with open(src_path, 'rb') as f:
                raw_bytes = f.read()
        except Exception as e:
            return f"[SYSTEM: ERROR] Could not read '{filename}': {e}"

        if not raw_bytes.strip():
            return f"[SYSTEM: ERROR] '{filename}' is empty — nothing to import."

        file_hash = hashlib.md5(raw_bytes).hexdigest()

        stem     = os.path.splitext(os.path.basename(filename))[0]
        out_name = f"{stem}_imported.jsonl"
        out_path = os.path.join(config.GLOBAL_HISTORIES_DIR, out_name)
        sidecar_path = out_path + ".import.json"

        # ── MD5 skip guard ──────────────────────────────────────────────────
        if os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path, 'r', encoding='utf-8') as f:
                    sidecar = json.load(f)
                if sidecar.get('source_hash') == file_hash:
                    return (
                        f"⚡ [ingest_chat] '{filename}' already imported, unchanged "
                        f"(content hash matches) → '{out_name}'. Already searchable via search_history."
                    )
            except Exception:
                pass  # corrupt/partial sidecar — fall through and re-import

        text = raw_bytes.decode('utf-8', errors='ignore')

        # ── Write the synthetic two-line .jsonl ─────────────────────────────
        try:
            os.makedirs(config.GLOBAL_HISTORIES_DIR, exist_ok=True)
            user_line      = json.dumps({"role": "user", "content": f"[SYSTEM: imported chat — {filename}]"})
            assistant_line = json.dumps({"role": "assistant", "content": text})
            tmp_path = out_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(user_line + "\n")
                f.write(assistant_line + "\n")
            os.replace(tmp_path, out_path)
        except Exception as e:
            return f"[ingest_chat] Failed to write '{out_name}': {e}"

        from utils import save_json_atomic
        save_json_atomic(sidecar_path, {"source_hash": file_hash, "source_file": filename})

        # Invalidate any existing turn index so search_history re-embeds fresh content
        idx_path = out_path + ".idx.json"
        if os.path.isfile(idx_path):
            try:
                os.remove(idx_path)
            except Exception:
                pass

        return (
            f"✅ Imported '{filename}' → '{out_name}' ({len(text):,} chars). "
            f"Now searchable via search_history."
        )

    def _handle_quick_search(self, args: dict, web_agent) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "[SYSTEM: ERROR] quick_search requires a 'query' argument."
        from_date = str(args.get("from_date", "")).strip() or None
        return web_agent.search(query, from_date=from_date)

    def _handle_url_search(self, args: dict, web_agent) -> str:
        """
        Read and smart-extract content from a specific URL.
        Uses DocumentationExtractor for technical docs, ContentExtractor otherwise.
        Returns the most relevant portion of the page content.
        """
        url     = args.get("url", "").strip()
        context = args.get("context", "").strip() or None

        if not url:
            return "[SYSTEM: ERROR] url_search requires a 'url' argument."

        if not (url.startswith("http://") or url.startswith("https://")):
            return f"[SYSTEM: ERROR] Invalid URL: '{url}'. Must start with http:// or https://"

        console.print(f"[dim]url_search: reading '{url[:70]}'...[/dim]")
        return web_agent.read_url(url, query_context=context)



    def _handle_workspace_search(self, args: dict, web_agent) -> str:
        query = args.get('query', '')
        file_filter = args.get('file_filter')
        _sem_raw = args.get('semantic', False)
        semantic = str(_sem_raw).lower() == 'true' if isinstance(_sem_raw, str) else bool(_sem_raw)
        return self.file_ops.search_workspace(query, file_filter, semantic=semantic)

    def _handle_doc_search(self, args: dict, web_agent) -> str:
        """
        Semantic passage search across all embedded documents in scratch.
        Uses chunk embeddings (nvidia/llama-nemotron-embed-1b-v2) for conceptual matching —
        works even when the user's words don't appear verbatim in the document.
        Searches .chunks.json files produced by ingest_pdf or ingest_text — not raw code files.
        """
        query = args.get("query", "").strip()
        # int() cast — LLMs sometimes pass numbers as strings.
        # Cap at 10 so the model can't accidentally request top_k=100
        # and flood its own context with passages.
        try:
            top_k = min(int(args.get("top_k", 5)), 10)
        except (TypeError, ValueError):
            top_k = 5

        if not query:
            return "[SYSTEM: ERROR] doc_search requires a 'query' argument."

        return self.doc_search_agent.search(query, top_k=top_k)

    def _handle_ingest_pdf(self, args: dict, web_agent) -> str:
        """
        Convert a PDF from /uploads into searchable text.
        Runs OCR on every page → saves .txt to /workspace/scratch →
        builds semantic chunk index for doc_search.
        Non-PDF files don't need this; read them directly from /uploads.
        """
        # Accept both <ingest_pdf>name.pdf</ingest_pdf> (filename key) and
        # <ingest_pdf><filename>name.pdf</filename></ingest_pdf>
        filename = (args.get('filename') or args.get('query') or '').strip()
        if not filename:
            return "[SYSTEM: ERROR] ingest_pdf requires a filename, e.g. <ingest_pdf>paper.pdf</ingest_pdf>"

        if not filename.lower().endswith('.pdf'):
            return (
                f"[ingest_pdf] '{filename}' is not a PDF. "
                "Non-PDF files (.py, .txt, .md, .csv, images) can be read directly "
                "from /uploads with view_lines or bash — no ingestion needed."
            )

        # Resolve path inside /uploads
        pdf_path = os.path.join(config.UPLOADS_FOLDER, filename)
        if not os.path.exists(pdf_path):
            # Fuzzy search — match by basename anywhere under /uploads
            for root, _, files in os.walk(config.UPLOADS_FOLDER):
                if filename in files:
                    pdf_path = os.path.join(root, filename)
                    break
            else:
                return f"[SYSTEM: ERROR] '{filename}' not found in /uploads."

        from doc_search_agent import PDFIngestAgent
        pdf_agent = PDFIngestAgent(
            ocr_agent=self.ocr_agent,
            scratch_dir=config.SCRATCH_DIR,
        )
        res = pdf_agent.ingest(pdf_path)

        if not res["success"]:
            return f"[ingest_pdf] Failed: {res['error']}"

        if res.get("skipped"):
            return (
                f"⚡ [ingest_pdf] '{filename}' is already ingested and unchanged "
                f"(content hash matches). No re-processing needed. "
                f"Use doc_search to query it, or workspace_search for exact keyword grep."
            )

        # Register output .txt with workspace tracker
        txt_path = res["txt_path"]
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content_txt = f.read()
            line_count = content_txt.count("\n") + 1
            self.workspace_tracker.track_file_write(txt_path, content_txt)
        except Exception as e:
            line_count = 0
            console.print(f"[yellow]⚠️  ingest_pdf tracker sync failed: {e}[/yellow]")

        # Invalidate the uploads-scan cache (item 11): the "ALREADY INGESTED"
        # note depends on files just written to SCRATCH_DIR, which doesn't
        # touch UPLOADS_FOLDER's mtime, so the cache wouldn't otherwise notice.
        self._uploads_cache = None

        # Bug 4 (second layer): don't imply doc_search will work just because
        # text extraction succeeded — surface it plainly when embedding failed
        # for a reason other than empty text (e.g. embedding API was down).
        if not res.get("embedded", True):
            return (
                f"⚠️ PDF text extracted: '{filename}' → '{res['output_file']}' "
                f"({res['pages']} pages, {line_count:,} lines, {res['chars']:,} chars), "
                f"but embedding failed — it is NOT yet searchable via doc_search. "
                f"Use workspace_search or view_lines on '{res['output_file']}' directly, "
                f"or try ingest_pdf again to retry embedding."
            )

        return (
            f"✅ PDF ingested: '{filename}' → '{res['output_file']}' "
            f"({res['pages']} pages, {line_count:,} lines, {res['chars']:,} chars, "
            f"{res.get('chunks', 0)} embedded chunks). "
            f"Use doc_search to query it semantically, or workspace_search for exact keyword grep."
        )

    def _handle_ingest_text(self, args: dict, web_agent) -> str:
        """
        Chunk + embed a plaintext file so doc_search can query it semantically.
        Mirrors the ingest_pdf pipeline (MD5 skip guard, copy-into-scratch,
        workspace-tracker registration) but skips OCR entirely — the file is
        already plain text.
        """
        filename = (args.get('filename') or args.get('query') or '').strip()
        if not filename:
            return "[SYSTEM: ERROR] ingest_text requires a filename, e.g. <ingest_text>notes.txt</ingest_text>"

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return f"[SYSTEM: ERROR] '{filename}' is a PDF — use ingest_pdf for PDFs, not ingest_text."
        if ext not in ('.txt', '.md'):
            return f"[SYSTEM: ERROR] '{filename}' must be a plaintext file (.txt or .md)."

        # Resolve path: /uploads first, then scratch/
        src_path   = os.path.join(config.UPLOADS_FOLDER, filename)
        in_uploads = True
        if not os.path.isfile(src_path):
            src_path   = os.path.join(config.SCRATCH_DIR, filename)
            in_uploads = False
        if not os.path.isfile(src_path):
            return f"[SYSTEM: ERROR] '{filename}' not found in /uploads or scratch."

        try:
            with open(src_path, 'rb') as f:
                raw_bytes = f.read()
        except Exception as e:
            return f"[SYSTEM: ERROR] Could not read '{filename}': {e}"

        file_hash   = hashlib.md5(raw_bytes).hexdigest()
        stem        = os.path.splitext(os.path.basename(filename))[0]
        chunks_path = os.path.join(config.SCRATCH_DIR, f"{stem}.chunks.json")

        # ── MD5 skip guard ──────────────────────────────────────────────────
        if os.path.isfile(chunks_path):
            try:
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get('source_hash') == file_hash:
                    return (
                        f"⚡ [ingest_text] '{filename}' already ingested and unchanged "
                        f"(content hash matches). Use doc_search to query it."
                    )
            except Exception:
                pass  # corrupt/partial chunks file — fall through to a fresh ingest

        # Copy into scratch/ if it came from uploads (mirrors the PDF pipeline,
        # where OCR output always lands in scratch alongside its .chunks.json)
        txt_path = os.path.join(config.SCRATCH_DIR, os.path.basename(filename))
        if in_uploads:
            try:
                shutil.copy2(src_path, txt_path)
            except Exception as e:
                return f"[ingest_text] Failed to copy '{filename}' into scratch: {e}"
        else:
            txt_path = src_path

        result = self.doc_search_agent.chunk_and_embed(txt_path, source_hash=file_hash)
        if not result.get('success'):
            return f"[ingest_text] Failed: {result.get('error', 'unknown error')}"

        # Register the .txt with the workspace tracker (same as ingest_pdf does)
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_txt = f.read()
            self.workspace_tracker.track_file_write(txt_path, content_txt)
        except Exception as e:
            console.print(f"[yellow]⚠️  ingest_text tracker sync failed: {e}[/yellow]")

        # Invalidate the uploads-scan cache, mirroring ingest_pdf's cache invalidation
        self._uploads_cache = None

        return (
            f"✅ Text ingested: '{filename}' → {result['chunks']} embedded chunks. "
            f"Use doc_search to query it."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FILE OPS HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_str_replace(self, args: dict, web_agent) -> str:
        try:
            # Parse count: "all" → 0 (replace everything), integer → first N, default 1
            count_raw = args.get("count", "1")
            if str(count_raw).strip().lower() == "all":
                count = 0
            else:
                try:
                    count = max(1, int(count_raw))
                except (ValueError, TypeError):
                    count = 1

            filepath = args.get('file')
            old_str  = args.get('old_str')
            new_str  = args.get('new_str', '')

            missing = [k for k, v in [('file', filepath), ('old_str', old_str)] if not v]
            if missing:
                return json.dumps({
                    "success": False,
                    "error": f"Missing required argument(s): {', '.join(f'<{m}>' for m in missing)}. "
                             f"Ensure your <str_replace> block includes all required child tags."
                })

            result = self.file_ops.str_replace(
                filepath=filepath,
                old_str=old_str,
                new_str=new_str,
                verify=args.get('verify', True),
                count=count,
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _handle_show_image(self, args: dict, web_agent) -> str:
        """
        Preview an image the model already created (via bash — dot, matplotlib,
        etc.) inline via chafa, and copy it to /outputs so the user can open it.

        Deliberately takes an explicit filename rather than auto-detecting the
        "latest" file in scratch: a bash call can produce more than one file
        (e.g. a debug CSV alongside the actual plot), and an explicit path lets
        a wrong filename come back as a clean, retryable error — same pattern
        as str_replace's old_str-not-found — instead of silently showing the
        wrong image.
        """
        filepath = args.get("file", "").strip()
        if not filepath:
            return json.dumps({
                "success": False,
                "error": "show_image requires a filepath, e.g. <show_image>chart.png</show_image>"
            })

        try:
            full_path = self.file_ops._resolve_path(filepath)
        except PermissionError as e:
            return json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            return json.dumps({"success": False, "error": f"Could not resolve path: {e}"})

        if not os.path.isfile(full_path):
            return json.dumps({
                "success": False,
                "error": f"File not found: {filepath}",
                "suggestion": "Generate the image with bash first, then call show_image on the exact filename it saved."
            })

        ext = os.path.splitext(full_path)[1].lower()
        if ext not in IMAGE_PREVIEW_EXTENSIONS:
            return json.dumps({
                "success": False,
                "error": f"'{ext or '(no extension)'}' isn't a previewable image type. "
                         f"Supported: {', '.join(sorted(IMAGE_PREVIEW_EXTENSIONS))}"
            })

        try:
            console.print(f"🖼️  [dim]show_image: {os.path.basename(full_path)}[/dim]")
            show_image_in_terminal(full_path)

            os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
            out_path = os.path.join(config.OUTPUTS_DIR, os.path.basename(full_path))
            shutil.copy2(full_path, out_path)

            # Track for the per-turn SYSTEM block written to chat_history
            _out_name = f"/outputs/{os.path.basename(full_path)}"
            if not hasattr(self, "_images_shown_this_turn"):
                self._images_shown_this_turn = []
            self._images_shown_this_turn.append(_out_name)

            return json.dumps({
                "success": True,
                "file": filepath,
                "shared_at": f"/outputs/{os.path.basename(full_path)}",
                "message": "Image previewed inline and copied to /outputs for the user."
            })
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _resolve_external_read(self, filepath: str):
        """
        Translate a container path to its host equivalent and, if it points
        outside the scratch sandbox (e.g. an absolute chat-history file path),
        validate and read it directly — file_ops itself is scoped to scratch.

        Shared by _handle_view_lines and _handle_search_in_file, which
        previously each reimplemented this translate+guard+open+read block
        independently; only the per-tool logic after this point (line-range
        slicing vs. pattern matching) differs.

        Returns (host_path, lines, error_json):
          - Not external (relative or inside scratch): (host_path, None, None)
            -- caller should proceed via self.file_ops.
          - External and readable: (host_path, list_of_lines, None)
          - External and blocked/missing/unreadable: (host_path, None, error_json_str)
        """
        host_path = config.container_to_host_path(filepath)

        # Use normcase+normpath for reliable comparison on Windows (mixed slashes).
        is_external = os.path.isabs(host_path) and not (
            _norm(host_path) == _norm(config.SCRATCH_DIR)
            or _norm(host_path).startswith(_norm(config.SCRATCH_DIR) + os.sep)
        )
        if not is_external:
            return host_path, None, None

        # Sandbox check: only allow reads from known safe directories
        if not _is_allowed_read_path(host_path):
            return host_path, None, json.dumps({
                "success": False,
                "error": f"Access denied: path outside allowed directories: {host_path}"
            })
        try:
            if not os.path.isfile(host_path):
                return host_path, None, json.dumps({"success": False, "error": f"File not found: {host_path}"})
            with open(host_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return host_path, lines, None
        except Exception as ex:
            return host_path, None, json.dumps({"success": False, "error": str(ex)})

    def _handle_view_lines(self, args: dict, web_agent) -> str:
        filepath = args.get('file', '')
        try:
            start = int(args.get('start', 1))
        except (ValueError, TypeError):
            start = 1
        try:
            end = int(args['end']) if args.get('end') else None
        except (ValueError, TypeError):
            end = None
        try:
            context = int(args.get('context', 5))
        except (ValueError, TypeError):
            context = 5

        filepath, ext_lines, err = self._resolve_external_read(filepath)
        if err:
            return err

        if ext_lines is not None:
            total = len(ext_lines)
            s, e = compute_view_window(total, start, end, context)
            selected = ext_lines[s:e]
            content  = "".join(selected)
            return json.dumps({
                "success":     True,
                "content":     content,
                "start_line":  s + 1,
                "end_line":    s + len(selected),
                "total_lines": total,
            }, indent=2)

        try:
            result = self.file_ops.view_lines(
                filepath=filepath,
                start=start,
                end=end,
                context=context
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _handle_search_in_file(self, args: dict, web_agent) -> str:
        filepath   = args.get('file', '')
        pattern    = args.get('pattern', '')
        _regex_raw = args.get('regex', False)
        use_regex  = str(_regex_raw).lower() == 'true' if isinstance(_regex_raw, str) else bool(_regex_raw)
        
        try:
            ctx_lines = int(args.get('context_lines', 3))
        except (ValueError, TypeError):
            ctx_lines = 3
        try:
            max_res = int(args.get('max_results', 10))
        except (ValueError, TypeError):
            max_res = 10

        filepath, ext_lines, err = self._resolve_external_read(filepath)
        if err:
            return err

        if ext_lines is not None:
            lines = ext_lines
            matches = []
            for i, line in enumerate(lines):
                hit = (re.search(pattern, line) if use_regex
                       else pattern.lower() in line.lower())
                if hit:
                    s = max(0, i - ctx_lines)
                    e = min(len(lines), i + ctx_lines + 1)
                    matches.append({
                        "line_number": i + 1,
                        "line":        line.rstrip(),
                        "context":     "".join(lines[s:e]),
                    })
                    if len(matches) >= max_res:
                        break
            return json.dumps({
                "success":       True,
                "total_matches": len(matches),
                "matches":       matches,
            }, indent=2)

        try:
            result = self.file_ops.search_in_file(
                filepath=filepath,
                pattern=pattern,
                regex=use_regex,
                context_lines=ctx_lines,
                max_results=max_res,
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    # ══════════════════════════════════════════════════════════════════════════
    # BASH HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    def _handle_bash_exec(self, args: dict, web_agent) -> str:
        """
        Run a shell command inside the Docker sandbox container via `docker exec`.

        start.py runs natively on Windows so readline/arrow-keys work perfectly.
        Only the bash tool proxies through Docker for sandbox isolation.

        The container has three mounts that match the host:
          /workspace/scratch  <- host config.SCRATCH_DIR  (read/write)
          /uploads            <- host config.UPLOADS_FOLDER (read-only)
          /outputs            <- host config.OUTPUTS_DIR   (read/write)

        So files the agent writes to config.SCRATCH_DIR are immediately visible
        inside the container at /workspace/scratch — no syncing needed.
        """
        command = args.get("command", "").strip()
        if not command:
            return "[SYSTEM: ERROR] bash: 'command' is required."

        # Hard blocklist — only truly catastrophic patterns via Regex
        # (BASH_BLOCKLIST is defined in utils.py, imported at module top)
        for pattern in BASH_BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE):
                return f"[SYSTEM: BLOCKED] Dangerous bash pattern detected: {pattern}"

        # ── mv pre-capture for tracker post-hook ─────────────────────────────
        # FIX #9: use shlex.split on the full command instead of naive
        # command.split("|") so quoted pipes (e.g. echo "a|b" | mv src dst)
        # don't split into wrong segments and corrupt token detection.
        mv_src = mv_dst = None
        try:
            def _to_host(p):
                translated = config.container_to_host_path(p)
                if translated == p and not os.path.isabs(p):
                    return os.path.join(config.SCRATCH_DIR, p)
                return translated

            all_tokens = shlex.split(command)
            WRAPPERS = {'sudo', 'env', 'command'}
            for i, tok in enumerate(all_tokens):
                if os.path.basename(tok) == 'mv':
                    paths = [t for t in all_tokens[i+1:] if not t.startswith("-")]
                    if len(paths) >= 2:
                        mv_src = _to_host(paths[0])
                        mv_dst = _to_host(paths[-1])
                    break
                if os.path.basename(tok) not in WRAPPERS:
                    break
        except Exception:
            pass

        # ── Execute inside container (STREAMING) ─────────────────────────────
        cmd_preview = command[:100] + ("..." if len(command) > 100 else "")
        console.print(f"🖥️  [dim]bash: {cmd_preview}[/dim]")

        docker_cmd = [
            "docker", "exec",
            "--workdir", config.CONTAINER_SCRATCH,
            config.CONTAINER_NAME,
            "bash", "-c", command,
        ]



        stdout_lines = []
        stderr_lines = []

        def read_stream(stream, lines_list, style):
            for line in stream:
                line = line.rstrip('\r\n')
                lines_list.append(line)
                console.print(line, style=style, highlight=False, markup=False)

        try:
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            
            console.print(f"[dim]┌─ sandbox [running] ───────────────────────────────────────────[/dim]")
            
            t_out = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "dim white"))
            t_err = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "yellow"))
            
            t_out.start()
            t_err.start()
            
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                # Terminate the host-side `docker exec` wrapper
                process.kill()
                
                # Actively kill the runaway process by restarting the container
                try:
                    console.print("[yellow]⚠️ Command timed out. Restarting sandbox to clean up...[/yellow]")
                    subprocess.run(
                        ["docker", "restart", "-t", "1", config.CONTAINER_NAME],
                        capture_output=True, timeout=10
                    )
                except Exception as e:
                    console.print(f"[red]⚠️ Failed to clean up container process: {e}[/red]")
                    
                t_out.join()
                t_err.join()
                
                try:
                    self.workspace_tracker.reconcile_workspace()
                except Exception:
                    pass
                return "[SYSTEM: BASH ERROR] Command timed out after 120 seconds."
            except KeyboardInterrupt:
                process.kill()
                try:
                    console.print("[yellow]⚠️ Interrupted. Restarting sandbox to clean up background processes...[/yellow]")
                    subprocess.run(
                        ["docker", "restart", "-t", "1", config.CONTAINER_NAME],
                        capture_output=True, timeout=10
                    )
                except Exception:
                    pass
                t_out.join()
                t_err.join()
                raise
                
            t_out.join()
            t_err.join()
            
        except FileNotFoundError:
            return "[SYSTEM: BASH ERROR] Docker not found. Is Docker Desktop running?"
        except Exception as e:
            return f"[SYSTEM: BASH ERROR] Execution failed: {e}"

        # FIX #3: always reconcile workspace before any early return so that
        # files partially created before a Docker-level error are still indexed.
        try:
            self.workspace_tracker.reconcile_workspace()
        except Exception as _re:
            console.print(f"[dim yellow]post-bash reconcile failed: {_re}[/dim yellow]")

        # Container not running / doesn't exist
        if process.returncode == 125:
            return (
                f"[SYSTEM: BASH ERROR] Container '{config.CONTAINER_NAME}' is not running.\n"
                f"Start it by running start.py first."
            )

        # Command found inside container but not executable (permissions issue)
        if process.returncode == 126:
            return (
                f"[SYSTEM: BASH ERROR] Permission denied — command is not executable inside the container.\n"
                f"Check file permissions (chmod +x) or use an interpreter explicitly (e.g. python script.py)."
            )

        # ── mv post-hook ──────────────────────────────────────────────────────
        if mv_src and mv_dst and process.returncode == 0:
            try:
                # Bug 17: if the source is a directory or a wildcard pattern, a
                # single remove_file() leaves all sub-files as phantom registry
                # entries.  reconcile_workspace() does a full diff and handles
                # all cases correctly.
                if os.path.isdir(mv_src) or '*' in mv_src or '?' in mv_src:
                    self.workspace_tracker.reconcile_workspace()
                else:
                    self.workspace_tracker.remove_file(mv_src)
                    # When destination is a directory, the file lands at dst/basename(src)
                    final_dst = (
                        os.path.join(mv_dst, os.path.basename(mv_src))
                        if os.path.isdir(mv_dst) else mv_dst
                    )
                    if os.path.isfile(final_dst):
                        with open(final_dst, "r", encoding="utf-8", errors="ignore") as f:
                            mv_content = f.read()
                        self.workspace_tracker.track_file_write(final_dst, mv_content)
            except Exception as e:
                console.print(f"[yellow]⚠️  mv post-hook failed: {e}[/yellow]")

        # ── Combine outputs for the model ─────────────────────────────────────
        stdout_full = "\n".join(stdout_lines)
        stderr_full = "\n".join(stderr_lines)

        if len(stdout_full) > 8000:
            stdout_full = _smart_truncate(stdout_full, max_chars=8000, head=3000)
        if len(stderr_full) > 2000:
            stderr_full = _smart_truncate(stderr_full, max_chars=2000, head=800)

        exit_color = "green" if process.returncode == 0 else "red"
        console.print(f"[dim]└─ sandbox [{exit_color}]exit {process.returncode}[/{exit_color}] ───────────────────────────────────────────[/dim]")

        parts = [f"[exit {process.returncode}]"]
        if stdout_full:
            parts.append(stdout_full)
        if stderr_full:
            parts.append(f"[stderr]\n{stderr_full}")

        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════════
    # WORKSPACE SCAN HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _scan_uploads_folder(self, pdf_active: bool = True) -> str:
        """
        Recursively scan /uploads and return a metadata listing.
        Runs every turn so the model sees new files immediately.
        All files are directly readable — only PDFs need ingest_pdf first.
        pdf_active: when False, suppress the ingest_pdf hint for PDF files
        (the model has no tool to act on it if the PDF group is disabled).

        Cached with a directory-mtime guard (item 11 / SP-6): the previous
        version did a full os.walk() + per-file os.path.getsize()/isfile()
        on every single turn, even though /uploads rarely changes mid-session.
        The cache key also includes pdf_active (the rendered note differs)
        and is explicitly invalidated by _handle_ingest_pdf() on a successful
        ingest, since the "ALREADY INGESTED" note depends on files written to
        SCRATCH_DIR — not UPLOADS_FOLDER — so an ingest wouldn't otherwise
        bump the uploads folder's mtime.
        """
        if not os.path.isdir(config.UPLOADS_FOLDER):
            self._uploads_cache = None
            return "[UPLOADS FOLDER EMPTY]"

        try:
            uploads_mtime = os.path.getmtime(config.UPLOADS_FOLDER)
        except OSError:
            uploads_mtime = None

        cache = getattr(self, "_uploads_cache", None)
        if (
            uploads_mtime is not None
            and cache is not None
            and cache[0] == uploads_mtime
            and cache[1] == pdf_active
        ):
            return cache[2]

        file_list = []
        for root, dirs, files in os.walk(config.UPLOADS_FOLDER):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext in UPLOAD_EXTENSIONS:
                    filepath = os.path.join(root, fn)
                    rel_path = os.path.relpath(filepath, config.UPLOADS_FOLDER).replace('\\', '/')
                    try:
                        size = os.path.getsize(filepath)
                        size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}KB"
                        if ext == ".pdf" and pdf_active:
                            stem       = os.path.splitext(fn)[0]
                            txt_path   = os.path.join(config.SCRATCH_DIR, stem + ".txt")
                            chunk_path = os.path.join(config.SCRATCH_DIR, stem + ".chunks.json")
                            if os.path.isfile(txt_path) and os.path.isfile(chunk_path):
                                note = f"  \u2190 ALREADY INGESTED \u2192 /workspace/scratch/{stem}.txt (available via doc_search or view_lines when needed \u2014 do NOT ingest again)"
                            else:
                                note = "  \u2190 ingest_pdf required before reading"
                        else:
                            note = ""
                        file_list.append(f"- /uploads/{rel_path} ({size_str}){note}")
                    except Exception:
                        pass

        if not file_list:
            result = "[UPLOADS FOLDER EMPTY]"
            if uploads_mtime is not None:
                self._uploads_cache = (uploads_mtime, pdf_active, result)
            return result

        result = (
            f"Found {len(file_list)} file(s) in /uploads (read directly with view_lines or bash):\n"
            + "\n".join(file_list)
        )
        if uploads_mtime is not None:
            self._uploads_cache = (uploads_mtime, pdf_active, result)
        return result

    def _handle_search_semantic_scholar(self, args: dict, web_agent) -> str:
        """
        Search the Semantic Scholar academic database.
        Returns title, year, TL;DR/abstract, and Open Access PDF link per paper.
        Results are reranked by relevance (NVIDIA cross-encoder) and truncated
        at 21,000 chars — same pattern as quick_search.
        """
        query = (args.get("query") or args.get("_raw") or "").strip()
        if not query:
            return "[SYSTEM: ERROR] search_semantic_scholar requires a 'query' argument."

        try:
            limit = min(int(args.get("limit", 10)), 20)
        except (ValueError, TypeError):
            limit = 10

        year_filter = (args.get("year") or "").strip() or None

        api_key = ""
        if getattr(config, "SEMANTIC_SCHOLAR_API_KEY_FILE", ""):
            api_key = read_api_key_cached(config.SEMANTIC_SCHOLAR_API_KEY_FILE)
        if not api_key:
            return (
                "[SYSTEM: ERROR] Semantic Scholar API key not configured. "
                "Set SEMANTIC_SCHOLAR_API_KEY_FILE in .env and add your key. "
                "Free key at semanticscholar.org/product/api."
            )

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,year,tldr,abstract,openAccessPdf,externalIds",
        }
        if year_filter:
            params["year"] = year_filter

        year_note = f" [{year_filter}]" if year_filter else ""
        console.print(f"📚 [dim]search_semantic_scholar: '{query[:60]}'{year_note} (limit={limit})...[/dim]")

        try:
            resp = requests.get(
                url,
                params=params,
                headers={"x-api-key": api_key},
                timeout=15,
            )
            if resp.status_code == 429:
                return (
                    "[SYSTEM: ERROR] Semantic Scholar API rate-limited (429 Too Many Requests). "
                    "Free tier is capped at 1 request/second — wait a moment and retry."
                )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"[SYSTEM: ERROR] Semantic Scholar request failed: {e}"

        papers = data.get("data", [])
        if not papers:
            return f"[search_semantic_scholar: '{query}'] No papers found."

        snippets = []
        for p in papers:
            title    = p.get("title") or "Untitled"
            year     = p.get("year") or "N/A"
            tldr_obj = p.get("tldr")
            tldr     = tldr_obj.get("text") if isinstance(tldr_obj, dict) else None
            abstract = (p.get("abstract") or "").strip()
            summary  = tldr if tldr else (abstract[:300] + "..." if len(abstract) > 300 else abstract)
            if not summary:
                summary = "No abstract or TL;DR available."

            ext_ids  = p.get("externalIds") or {}
            doi      = ext_ids.get("DOI")
            arxiv    = ext_ids.get("ArXiv")
            ref_str  = f" | DOI: {doi}" if doi else (f" | arXiv: {arxiv}" if arxiv else "")

            pdf_obj  = p.get("openAccessPdf")
            pdf_url  = pdf_obj.get("url") if isinstance(pdf_obj, dict) else None
            pdf_str  = f"\n   PDF: {pdf_url}" if pdf_url else ""

            snippets.append(f"{title} ({year}){ref_str}\n   {summary}{pdf_str}")

        # Rerank by relevance — same NVIDIA cross-encoder used by quick_search.
        # Falls back to original order if NVIDIA key absent or reranker fails.
        nvidia_key = getattr(web_agent, "_nvidia_key", None) if web_agent else None
        order = rerank_passages(query, snippets, api_key=nvidia_key, label="papers")
        snippets = [snippets[i] for i in order]

        header = f"[search_semantic_scholar: '{query}' — {len(snippets)} paper(s)]"
        body   = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(snippets))
        final  = f"{header}\n\n{body}"

        if len(final) > 21000:
            final = final[:21000] + "\n\n[SYSTEM: TRUNCATED — result exceeded 21,000 chars]"

        console.print(f"[dim]{final}[/dim]")
        return final