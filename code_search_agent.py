# code_search_agent.py - Semantic code search via tree-sitter chunking + Codestral Embed
"""
Powers workspace_search(semantic=true). Two responsibilities:

1. CHUNKING — chunk_code_file(filepath, rel_path):
   Parses a source file into "interesting" units (functions, classes, methods)
   using tree-sitter when a grammar is available for the language, with a
   regex-based (blank-line + fixed-window) fallback otherwise. Every chunk is
   given an enriched `embed_text` ("# File: X | function: Y\n# <docstring>\n\n<code>")
   so retrieval keys on file/symbol context, not just raw code tokens.

2. INDEX + SEARCH — CodeSearchAgent:
   Maintains database/code_index/code_index.json: a per-file MD5 hash map plus
   a flat list of embedded chunks. Rebuilds are incremental — only files whose
   hash has changed are re-chunked and re-embedded — and lazy: mark_stale()
   (called after every bash/str_replace) just flips a boolean; the actual
   re-embed only happens the next time search() is called with semantic=true.

Design decisions (see implementation_plan.md for the full writeup):
  - Codestral Embed is a *separate* provider/model from the NVIDIA Nemotron
    embeddings used by doc_search_agent.py / search_history — hence its own
    _embed_code()/_embed_code_batched() in utils.py rather than reusing _embed().
  - Chunk/file paths are stored in *container* form (/workspace/scratch/...),
    matching what the model already uses in other tool calls, via
    config.host_to_container_path().
  - tree-sitter is optional: if tree-sitter-languages isn't importable (e.g.
    no wheel for the running Python version), or parsing a given file raises
    for any reason, that file silently falls back to the regex chunker. No
    hard crash either way.
"""

import os
import json
import fnmatch
import hashlib
from dataclasses import dataclass
from typing import Optional

import config
from utils import (
    console,
    detect_language,
    read_api_key_cached,
    save_json_atomic,
    _embed_code,
    _embed_code_batched,
    _cosine_similarity,
)

# ── Graceful tree-sitter import ────────────────────────────────────────────
# If tree-sitter-language-pack is not installed, AST-aware chunking will 
# gracefully fall back to the regex chunker for all files.
try:
    from tree_sitter_language_pack import get_parser
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE / NODE-TYPE TABLES
# ══════════════════════════════════════════════════════════════════════════════

# AST node types that tree-sitter's grammar for each language uses to mark
# "interesting" chunks (functions, classes, methods, top-level types).
# Languages not listed here always use the regex fallback, no tree-sitter
# attempt is made — this includes markdown/txt/yaml/json/css/html/sql/bash,
# which don't have a meaningful "function/class" structure worth chunking on.
CHUNK_NODE_TYPES = {
    "python":     {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "arrow_function",
                   "method_definition", "export_statement"},
    "typescript": {"function_declaration", "class_declaration", "arrow_function",
                   "method_definition", "export_statement"},
    "go":         {"function_declaration", "method_declaration", "type_declaration"},
    "c":          {"function_definition"},
    "cpp":        {"function_definition", "class_specifier"},
    "java":       {"method_declaration", "class_declaration", "interface_declaration"},
    "rust":       {"function_item", "impl_item", "struct_item"},
    "ruby":       {"method", "class", "module"},
    "php":        {"function_definition", "method_declaration", "class_declaration"},
}

# Buckets used to classify a matched node into a chunk "kind". A node type may
# appear in exactly one of these — used across every language's node set above.
_FUNCTION_TYPES = {
    "function_definition", "function_declaration", "function_item",
    "arrow_function", "method_definition", "method_declaration", "method",
}
_CLASS_TYPES = {
    "class_definition", "class_declaration", "class_specifier", "class", "impl_item",
}
_TYPE_TYPES = {
    "type_declaration", "struct_item", "interface_declaration", "module",
}
# Wrapper nodes that don't themselves carry a name/kind — the real definition
# is one of their children. We keep the WRAPPER's line span (so decorators /
# `export` keywords are included in the chunk) but classify + name from the
# inner node.
_WRAPPER_TYPES = {"decorated_definition", "export_statement"}

# Regex-fallback chunking window, in lines. Used for any language without an
# entry in CHUNK_NODE_TYPES, when tree-sitter isn't installed, or when
# tree-sitter parsing of a specific file raises for any reason.
_REGEX_WINDOW  = 60
_REGEX_OVERLAP = 15


# ══════════════════════════════════════════════════════════════════════════════
# CODE CHUNK
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeChunk:
    text: str                    # raw code, as it appears in the file
    embed_text: str              # enriched: "# File: X | kind: name\n# <doc>\n\n<code>"
    file: str                    # container-style path, e.g. /workspace/scratch/foo.py
    name: str                    # function/class/method name (or a "L<start>-<end>" label)
    kind: str                    # "function" | "class" | "method" | "type" | "block"
    start_line: int              # 1-indexed, inclusive
    end_line: int                # 1-indexed, inclusive
    language: str
    parent: Optional[str] = None  # enclosing class name, for kind == "method"


def _build_embed_text(rel_path: str, kind: str, name: str, doc_first_line: str,
                      code_text: str, parent: Optional[str] = None) -> str:
    # For methods, qualify with parent class: "Foo.bar" so queries like
    # "Foo rate limiter" or "Foo.bar" surface this chunk ahead of unrelated
    # functions also named "bar".
    display_name = f"{parent}.{name}" if parent else name
    header = f"# File: {rel_path} | {kind}: {display_name}"
    if doc_first_line:
        header += f"\n# {doc_first_line}"
    return f"{header}\n\n{code_text}"


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING — tree-sitter path
# ══════════════════════════════════════════════════════════════════════════════

def _classify(node_type: str) -> str:
    if node_type in _CLASS_TYPES:
        return "class"
    if node_type in _TYPE_TYPES:
        return "type"
    if node_type in _FUNCTION_TYPES:
        return "function"
    return "block"


def _chunk_via_treesitter(text: str, language: str, rel_path: str) -> list:
    """
    AST-parse `text` with tree-sitter and extract one CodeChunk per function,
    class, and method. Raises on any unexpected error — callers must catch
    and fall back to _chunk_via_regex().
    """
    parser    = get_parser(language)
    src_bytes = text.encode("utf-8", errors="ignore")
    tree      = parser.parse(src_bytes)
    root      = tree.root_node
    node_types = CHUNK_NODE_TYPES[language]
    chunks: list = []

    def node_text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def extract_name(node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return node_text(name_node)
        # JS arrow functions assigned to a variable carry their name on the
        # *parent* (variable_declarator), not on the arrow_function node itself.
        try:
            parent = node.parent
        except AttributeError:
            parent = None
        if parent is not None and parent.type == "variable_declarator":
            pname = parent.child_by_field_name("name")
            if pname is not None:
                return node_text(pname)
        # Wrapper nodes (decorated_definition/export_statement): look inside.
        for child in node.children:
            if child.type in _FUNCTION_TYPES or child.type in _CLASS_TYPES or child.type in _TYPE_TYPES:
                inner_name = extract_name(child)
                if inner_name:
                    return inner_name
        return "<anonymous>"

    def docstring_first_line(node) -> str:
        """Best-effort: python/JS-style — first statement of the body if it's
        a bare string expression (python docstring) or a leading line comment
        immediately preceding the definition."""
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                if child.type == "expression_statement":
                    for gc in child.children:
                        if gc.type in ("string", "string_literal"):
                            raw = node_text(gc).strip("\"'").strip()
                            if raw:
                                return raw.splitlines()[0].strip()[:120]
                    break
                if child.type not in ("comment",):
                    break
        return ""

    def unwrap(node):
        """For decorated_definition/export_statement, find the inner def/class
        used for naming + kind classification. Returns (inner_node, kind)."""
        if node.type not in _WRAPPER_TYPES:
            return node, _classify(node.type)
        for child in node.children:
            if child.type in _FUNCTION_TYPES or child.type in _CLASS_TYPES or child.type in _TYPE_TYPES:
                return child, _classify(child.type)
        return node, "block"

    def walk(node, parent_class: Optional[str] = None):
        for child in node.children:
            if child.type not in node_types:
                walk(child, parent_class)
                continue

            inner, kind = unwrap(child)
            if kind == "block":
                # Wrapper node with nothing recognisable inside — recurse past it.
                walk(child, parent_class)
                continue

            name        = extract_name(inner)
            start_line  = child.start_point[0] + 1   # keep wrapper's span (decorators/export included)
            end_line    = child.end_point[0] + 1
            code_text   = node_text(child)
            doc         = docstring_first_line(inner)
            kind_final  = "method" if (kind == "function" and parent_class) else kind

            chunks.append(CodeChunk(
                text=code_text,
                embed_text=_build_embed_text(rel_path, kind_final, name, doc, code_text,
                                             parent=parent_class if kind_final == "method" else None),
                file=rel_path,
                name=name,
                kind=kind_final,
                start_line=start_line,
                end_line=end_line,
                language=language,
                parent=parent_class if kind_final == "method" else None,
            ))

            if kind == "class":
                walk(inner, parent_class=name)
            # Do not recurse into function/method bodies — nested closures stay
            # embedded in their parent's chunk text rather than becoming noisy
            # duplicate mini-chunks.

    walk(root)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING — regex fallback path
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_via_regex(text: str, rel_path: str, language: str) -> list:
    """
    Used when tree-sitter is unavailable, the language has no grammar entry,
    or AST parsing failed. Splits on blank lines into logical blocks, then
    merges small consecutive blocks up to _REGEX_WINDOW lines, or sub-windows
    oversized blocks with _REGEX_OVERLAP lines of overlap so context is never
    lost at a chunk boundary.
    """
    lines = text.split("\n")
    if not lines:
        return []

    blocks: list = []       # list of list[(line_no, text)]
    current: list = []
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append((i, line))
    if current:
        blocks.append(current)
    if not blocks:
        return []

    chunks: list = []

    def flush(buf):
        if not buf:
            return
        start_line = buf[0][0]
        end_line   = buf[-1][0]
        chunk_text = "\n".join(l for _, l in buf)
        name       = f"{os.path.basename(rel_path)}:L{start_line}-{end_line}"
        chunks.append(CodeChunk(
            text=chunk_text,
            embed_text=_build_embed_text(rel_path, "block", name, "", chunk_text),
            file=rel_path, name=name, kind="block",
            start_line=start_line, end_line=end_line,
            language=language, parent=None,
        ))

    buf: list = []
    step = _REGEX_WINDOW - _REGEX_OVERLAP
    for block in blocks:
        if len(block) > _REGEX_WINDOW:
            flush(buf)
            buf = []
            j, blen = 0, len(block)
            while j < blen:
                flush(block[j:j + _REGEX_WINDOW])
                if j + _REGEX_WINDOW >= blen:
                    break
                j += step
        else:
            if buf and (len(buf) + len(block) > _REGEX_WINDOW):
                flush(buf)
                buf = buf[-_REGEX_OVERLAP:] if len(buf) > _REGEX_OVERLAP else buf
            buf.extend(block)
    flush(buf)
    return chunks


def _split_oversized_chunk(chunk: CodeChunk) -> list:
    """A single AST node (e.g. a huge class) exceeded CODE_CHUNK_MAX_LINES —
    split it into fixed windows so no one chunk dominates the embedding
    budget or the search results."""
    lines     = chunk.text.split("\n")
    max_lines = config.CODE_CHUNK_MAX_LINES
    parts: list = []
    i, part_num = 0, 1
    while i < len(lines):
        sub_lines  = lines[i:i + max_lines]
        start_line = chunk.start_line + i
        end_line   = start_line + len(sub_lines) - 1
        sub_text   = "\n".join(sub_lines)
        name       = f"{chunk.name} (part {part_num})"
        parts.append(CodeChunk(
            text=sub_text,
            embed_text=_build_embed_text(chunk.file, chunk.kind, name, "", sub_text,
                                         parent=chunk.parent),
            file=chunk.file, name=name, kind=chunk.kind,
            start_line=start_line, end_line=end_line,
            language=chunk.language, parent=chunk.parent,
        ))
        i += max_lines
        part_num += 1
    return parts


def chunk_code_file(filepath: str, rel_path: str) -> list:
    """
    Parse a file on disk into CodeChunks.

    1. Detect language from extension (reuses utils.detect_language — the
       same map view_lines/search_in_file already use for syntax highlighting).
    2. If tree-sitter is available and the language has a node-type table,
       try an AST parse; any exception falls back to regex chunking for this
       file only (never a hard crash).
    3. Enforce CODE_CHUNK_MIN_LINES / CODE_CHUNK_MAX_LINES.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    language = detect_language(filepath)

    raw_chunks: list = []
    if TREESITTER_AVAILABLE and language in CHUNK_NODE_TYPES:
        try:
            raw_chunks = _chunk_via_treesitter(text, language, rel_path)
        except Exception as e:
            console.print(
                f"[dim yellow]tree-sitter chunking failed for {rel_path} ({e}); "
                f"falling back to regex[/dim yellow]"
            )
            raw_chunks = []

    # Capture before the regex fallback below overwrites raw_chunks.
    used_treesitter = bool(raw_chunks)

    if not raw_chunks:
        raw_chunks = _chunk_via_regex(text, rel_path, language)

    final_chunks: list = []
    for c in raw_chunks:
        n_lines = c.end_line - c.start_line + 1
        if n_lines < config.CODE_CHUNK_MIN_LINES:
            continue
        if n_lines <= config.CODE_CHUNK_MAX_LINES:
            final_chunks.append(c)
        else:
            final_chunks.extend(_split_oversized_chunk(c))

    # Bug 4: tree-sitter yielded nodes, but ALL were under CODE_CHUNK_MIN_LINES
    # (e.g. a file of 2-line helpers: def a(): return 1). The regex chunker
    # merges consecutive small blocks into searchable windows \u2014 use it instead
    # of leaving the file completely invisible to semantic search.
    if not final_chunks and used_treesitter:
        for c in _chunk_via_regex(text, rel_path, language):
            n_lines = c.end_line - c.start_line + 1
            if n_lines < config.CODE_CHUNK_MIN_LINES:
                continue
            if n_lines <= config.CODE_CHUNK_MAX_LINES:
                final_chunks.append(c)
            else:
                final_chunks.extend(_split_oversized_chunk(c))

    return final_chunks


# ══════════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY / EXCLUSION
# ══════════════════════════════════════════════════════════════════════════════

def _should_index(filepath: str) -> bool:
    """Files that must NEVER be indexed — secrets, key files, VCS/dep dirs,
    and anything outside the extension allowlist (binaries, joblib/pkl, etc.)."""
    basename = os.path.basename(filepath)
    if basename in config.CODE_INDEX_EXCLUDE:
        return False
    if basename.endswith(".key"):
        return False
    # Bug 3: doc_search artifacts (.chunks.json, .embed.json) are JSON files
    # that contain large embedding arrays — they'd get indexed as "code",
    # polluting search results with doc_search internals and wasting token budget.
    if basename.endswith(".chunks.json") or basename.endswith(".embed.json"):
        return False
    parts = filepath.replace("\\", "/").split("/")
    if any(d in config.CODE_INDEX_EXCLUDE_DIRS for d in parts):
        return False
    ext = os.path.splitext(basename)[1].lower()
    if ext not in config.CODE_INDEX_EXTENSIONS:
        return False
    return True


def _iter_source_files(search_dirs: list) -> list:
    """Walk every dir in search_dirs, returning [(abs_path, container_path), ...]
    for indexable files. De-duplicates by container path so the same logical
    file is never indexed twice even if search_dirs overlap."""
    seen: set = set()
    results: list = []
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in config.CODE_INDEX_EXCLUDE_DIRS]
            for fn in files:
                abs_path = os.path.join(root, fn)
                if not _should_index(abs_path):
                    continue
                container_path = config.host_to_container_path(abs_path)
                if container_path in seen:
                    continue
                seen.add(container_path)
                results.append((abs_path, container_path))
    return results


def _hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# CODE SEARCH AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CodeSearchAgent:
    """
    Semantic code search over scratch/ + uploads/ using Mistral Codestral Embed.

    Lifecycle:
      - Constructed once in EmperorAgent.__init__ with
        search_dirs=[config.SCRATCH_DIR, config.UPLOADS_FOLDER].
      - mark_stale() is called after every bash/str_replace — a cheap flag
        flip, no embed calls.
      - The actual (incremental) re-embed happens lazily, inside search(),
        the first time the model asks for semantic=true after something changed.
      - invalidate_index() is called by /restore — the workspace changed via a
        zip-unpack, so on-disk file hashes can't be trusted incrementally;
        wipe and rebuild fully on the next query.
    """

    def __init__(self, search_dirs: list, index_dir: str):
        self.search_dirs = search_dirs
        self.index_dir   = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path  = os.path.join(self.index_dir, "code_index.json")
        self._index        = None   # lazy-loaded dict: version/embed_model/file_hashes/chunks
        self._index_loaded = False
        self._stale         = True  # starts stale — first semantic query triggers a build

    # ── Public API ──────────────────────────────────────────────────────

    def mark_stale(self):
        """Called after bash/str_replace — just flips a flag, no embed calls."""
        self._stale = True

    def invalidate_index(self):
        """Called by /restore — wipes the index, forces a full rebuild on next query."""
        self._index        = None
        self._index_loaded = False
        self._stale         = True
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        except OSError as e:
            console.print(f"[yellow]⚠️ Could not remove stale code index file: {e}[/yellow]")

    def build_index(self, force: bool = False):
        """
        Incremental index build:
          1. Load existing index from disk (if any), or start from a skeleton.
          2. Walk search_dirs, compute an MD5 hash per file.
          3. Compare against stored file_hashes — unchanged files are skipped
             entirely (their existing chunks + embeddings are kept as-is);
             new/changed files are re-chunked and their chunks re-embedded;
             deleted files have their chunks dropped.
          4. Save the updated index atomically.

        Editing one file only re-embeds that one file's chunks, not the whole
        codebase — this is what keeps semantic search cheap enough to run on
        every query instead of only at startup.

        Commit is per-file, not per-loop: a file's new hash is only written
        once every one of its chunks has either a real embedding, or was
        permanently too large to ever embed (see _embed_code_batched's
        oversized_indices) — or the file produced zero chunks in the first
        place (nothing to embed, nothing that can fail). A file with a
        genuinely failed chunk (rate limit, exhausted retries, etc.) has its
        hash left unwritten — its stale chunks were already dropped, so it's
        simply absent from the index until the next build retries it, rather
        than silently serving a partially-embedded or pre-edit version. An
        oversized chunk is different: retrying changes nothing (the same
        estimate recurs every build), so that one chunk is excluded
        permanently while the rest of the file still commits and stays
        searchable — otherwise a single huge chunk (e.g. one minified line
        in an indexed .json file) would blank out the whole file forever
        and burn an API call re-embedding its other chunks on every query.
        Files that embedded cleanly are still saved immediately, so they
        aren't re-embedded next time.
        """
        if force:
            self._index        = None
            self._index_loaded = False

        if self._index is None:
            loaded = self._load_index()
            # _load_index() returns None when stored model/dimension differs from
            # config (Bug 2) — treat that the same as force=True: start fresh so
            # we never mix old vectors of the wrong shape with new ones.
            self._index = loaded if loaded is not None else {
                "version":         1,
                "embed_model":     config.CODE_EMBED_MODEL,
                "embed_dimension": config.CODE_EMBED_DIMENSION,
                "file_hashes":     {},
                "chunks":          [],
            }
            self._index_loaded = True

        current_files = _iter_source_files(self.search_dirs)
        current_map   = {cpath: apath for apath, cpath in current_files}

        old_hashes = self._index["file_hashes"]
        chunks_by_file: dict = {}
        for c in self._index["chunks"]:
            chunks_by_file.setdefault(c["file"], []).append(c)

        # ── Deleted files ────────────────────────────────────────────────
        for f in [f for f in old_hashes if f not in current_map]:
            del old_hashes[f]
            chunks_by_file.pop(f, None)

        any_failure = False

        # ── New / changed files: chunk first, stage the hash, don't commit
        #    it until the embed step below confirms every chunk succeeded ──
        pending_hashes: dict = {}   # cpath -> new_hash, staged
        pending_chunks: dict = {}   # cpath -> list[CodeChunk], staged
        to_embed_texts: list = []
        to_embed_owner: list = []   # parallel: (container_path, CodeChunk)

        for cpath, apath in current_map.items():
            try:
                new_hash = _hash_file(apath)
            except OSError:
                continue
            if old_hashes.get(cpath) == new_hash:
                continue   # unchanged — keep existing chunks + embeddings

            try:
                new_chunks = chunk_code_file(apath, cpath)
            except Exception as e:
                console.print(f"[yellow]⚠️ Skipping {cpath} — chunking failed: {e}[/yellow]")
                any_failure = True
                continue

            pending_hashes[cpath] = new_hash
            pending_chunks[cpath] = new_chunks
            for ch in new_chunks:
                to_embed_texts.append(ch.embed_text)
                to_embed_owner.append((cpath, ch))

        # Drop stale chunks now, only for files we're actually about to
        # replace (chunking already succeeded for these) — old copies of
        # files that failed to chunk are left untouched above.
        for cpath in pending_hashes:
            chunks_by_file.pop(cpath, None)

        if to_embed_texts:
            # The only place that spends Mistral API calls/tokens — only
            # chunks belonging to genuinely new/changed files.
            embeddings, oversized_idx = _embed_code_batched(to_embed_texts)
            embedded_by_file: dict = {}
            for i, ((cpath, ch), emb) in enumerate(zip(to_embed_owner, embeddings)):
                embedded_by_file.setdefault(cpath, []).append((ch, emb, i in oversized_idx))

            for cpath, triples in embedded_by_file.items():
                # A file's hash is only withheld (forcing a retry on the next
                # build) for a *genuine* embed failure — a chunk whose emb is
                # None but wasn't in oversized_idx (rate limit, exhausted
                # bisect-and-retry, etc.). A chunk that's permanently too big
                # for the model (oversized_idx) is NOT such a failure: its
                # size estimate will be exactly the same on every future
                # build, so retrying only repeats the same skip forever while
                # needlessly re-embedding the file's other, perfectly good
                # chunks on every single semantic query. Treat it instead as
                # a deliberate, permanent exclusion of just that one chunk —
                # the rest of the file still gets indexed and its hash still
                # commits, so it doesn't vanish from search entirely.
                genuinely_failed = any(
                    emb is None and not is_oversized for _, emb, is_oversized in triples
                )
                if genuinely_failed:
                    any_failure = True
                    console.print(
                        f"[yellow]⚠️ {cpath}: one or more chunks failed to embed — "
                        f"will retry on the next search.[/yellow]"
                    )
                    continue

                n_oversized = sum(1 for _, _, is_oversized in triples if is_oversized)
                n_indexed   = len(triples) - n_oversized
                if n_oversized:
                    console.print(
                        f"[yellow]⚠️ {cpath}: {n_oversized} oversized chunk(s) permanently "
                        f"excluded from the semantic index ({n_indexed} other chunk(s) "
                        f"still indexed normally).[/yellow]"
                    )
                for ch, emb, is_oversized in triples:
                    if is_oversized:
                        continue  # deliberately excluded — no embedding to store
                    chunks_by_file.setdefault(cpath, []).append({
                        "file": ch.file, "name": ch.name, "kind": ch.kind,
                        "start_line": ch.start_line, "end_line": ch.end_line,
                        "language": ch.language, "parent": ch.parent,
                        "text": ch.text, "embedding": emb,
                    })
                old_hashes[cpath] = pending_hashes[cpath]

        # Files that produced zero chunks (e.g. everything under
        # CODE_CHUNK_MIN_LINES) never entered to_embed_texts — nothing to
        # embed, nothing that can fail, so commit their hash right away.
        for cpath, new_chunks in pending_chunks.items():
            if not new_chunks:
                old_hashes[cpath] = pending_hashes[cpath]

        self._index["chunks"]          = [c for fc in chunks_by_file.values() for c in fc]
        self._index["file_hashes"]     = old_hashes
        self._index["embed_model"]     = config.CODE_EMBED_MODEL
        self._index["embed_dimension"] = config.CODE_EMBED_DIMENSION

        # Save whatever did commit — successful files must not be
        # re-embedded next time even if others in this batch failed.
        save_json_atomic(self.index_path, self._index)

        # Only clear staleness if nothing failed; otherwise the next
        # search() retries immediately instead of waiting for the next
        # bash/str_replace to call mark_stale() again.
        self._stale = any_failure

    def search(self, query: str, top_k: int = 8, file_filter: str = None) -> str:
        """
        1. If stale (or never loaded) → rebuild index (incremental, fast if
           few files changed).
        2. Embed the query via _embed_code([query]).
        3. If file_filter given, pre-filter chunks by basename glob match.
        4. Cosine similarity against all chunk embeddings.
        5. Return the top-K formatted with file paths, line numbers, and code.

        Raises RuntimeError if MISTRAL_API_KEY_FILE isn't configured or an
        embed call fails outright — callers (file_ops_agent.search_workspace)
        catch this and fall back to BM25.
        """
        api_key = read_api_key_cached(config.MISTRAL_API_KEY_FILE)
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY_FILE is not configured — semantic code search unavailable."
            )

        if self._stale or self._index is None:
            self.build_index()

        chunks = self._index.get("chunks", [])
        if file_filter:
            chunks = [c for c in chunks if fnmatch.fnmatch(os.path.basename(c["file"]), file_filter)]
        chunks = [c for c in chunks if c.get("embedding")]

        if not chunks:
            if self._stale:
                # build_index() just ran and still left us stale — some
                # file(s) failed to embed (rate limit, token overflow, etc.)
                # and nothing else is indexed. Raise so the caller
                # (file_ops_agent.search_workspace) falls back to BM25 for
                # *this* query; the next search() call retries the failed
                # files automatically, no separate trigger needed.
                raise RuntimeError(
                    "Semantic code index has no embedded chunks yet — the last "
                    "embed attempt failed for one or more files. Retrying on "
                    "the next semantic search."
                )
            # Not stale and still empty — a genuinely empty workspace (or a
            # filter that matched nothing), not a failure. Don't fall back.
            suffix = f" matching filter '{file_filter}'" if file_filter else ""
            return (
                f"[SYSTEM: CODE SEARCH: '{query}'] — no indexed code chunks available{suffix}. "
                f"scratch/uploads may be empty, or contain only excluded/binary files."
            )

        query_vec    = _embed_code([query])[0]
        passage_vecs = [c["embedding"] for c in chunks]
        scores       = _cosine_similarity(query_vec, passage_vecs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

        lines = [f"[SYSTEM: CODE SEARCH: '{query}'] — {len(ranked)} result(s)\n"]
        for i, (c, score) in enumerate(ranked, 1):
            lines.append("─" * 50)
            symbol = f"{c['parent']}.{c['name']}" if c.get("parent") else c["name"]
            lines.append(
                f"📄 [{i}] {c['file']} → {symbol} "
                f"({c['kind']}, lines {c['start_line']}-{c['end_line']})"
            )
            lines.append(f"    similarity: {score:.3f}\n")
            lines.append(c["text"])
            lines.append("")
        return "\n".join(lines)

    # ── Introspection helper (also used by ad-hoc verification scripts) ──

    def _chunk_file(self, filepath: str) -> list:
        """Resolve `filepath` against self.search_dirs and return its
        CodeChunks without touching the index."""
        for base in self.search_dirs:
            candidate = filepath if os.path.isabs(filepath) else os.path.join(base, filepath)
            if os.path.isfile(candidate):
                container_path = config.host_to_container_path(candidate)
                return chunk_code_file(candidate, container_path)
        raise FileNotFoundError(f"{filepath!r} not found under any of {self.search_dirs}")

    def _load_index(self) -> dict:
        """Load the on-disk index, or return a fresh empty skeleton.

        Returns None if the on-disk index exists but was built with a different
        embed_model or embed_dimension than the current config — signals the
        caller (build_index) to do a full rebuild rather than mixing old vectors
        of the wrong shape with new ones (which would cause cosine_similarity to
        raise on inhomogeneous arrays, or silently return wrong scores).
        """
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Bug 2: detect model/dimension mismatch before loading old vectors.
                stored_model = data.get("embed_model", "")
                stored_dim   = data.get("embed_dimension", -1)
                if stored_model != config.CODE_EMBED_MODEL or stored_dim != config.CODE_EMBED_DIMENSION:
                    console.print(
                        f"[yellow]⚠️ Code index was built with {stored_model!r} dim={stored_dim}; "
                        f"config now wants {config.CODE_EMBED_MODEL!r} dim={config.CODE_EMBED_DIMENSION}. "
                        f"Rebuilding from scratch.[/yellow]"
                    )
                    return None  # caller interprets None as: force full rebuild
                data.setdefault("file_hashes", {})
                data.setdefault("chunks", [])
                return data
            except Exception as e:
                console.print(
                    f"[yellow]⚠️ Code index on disk was unreadable ({e}); rebuilding from scratch.[/yellow]"
                )
        return {
            "version":         1,
            "embed_model":     config.CODE_EMBED_MODEL,
            "embed_dimension": config.CODE_EMBED_DIMENSION,
            "file_hashes":     {},
            "chunks":          [],
        }
