# workspace_tracker.py - Track files and folders with structure visualization
import os
import re
import glob
import fnmatch
import hashlib
import json
import time
from utils import detect_language, CODE_EXTENSIONS, load_json, save_json_atomic, EXCL_DIRS, console
import config

# BM25 semantic search via rank-bm25 (pure Python, no model download, no token limits)
try:
    from rank_bm25 import BM25Okapi
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False
    console.print("⚠️  [yellow]rank-bm25 not found — semantic search disabled. Run: pip install rank-bm25[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
# CODE-AWARE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

def _code_tokenizer(text: str) -> list:
    """
    Tokenizer designed for source code.

    Handles:
    - snake_case  → ['snake', 'case']
    - camelCase   → ['camel', 'case']
    - PascalCase  → ['pascal', 'case']
    - UPPER_CONST → ['upper', 'const']
    - dotted.names → ['dotted', 'names']

    This dramatically improves recall — a query for "database connection"
    will match a file containing `DatabaseConnection` or `db_connect`.
    """
    # Step 1: split camelCase and PascalCase  (insertBefore → insert Before)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)

    # Step 2: replace underscores, dots, hyphens, slashes with spaces
    text = re.sub(r'[_.\-/]', ' ', text)

    # Step 3: extract alphabetic tokens of length >= 2, lowercase
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{1,}\b', text)
    return [t.lower() for t in tokens]


class WorkspaceTracker:
    """
    Track files and folders in workspace with tree visualization.

    Features:
    - File tracking with deduplication (hash-based)
    - Folder structure tracking
    - BM25 semantic search (no model download, no token limits, code-aware)
    """

    def __init__(self, workspace_path: str = None):
        import config as _cfg
        self.workspace_path  = workspace_path if workspace_path is not None else _cfg.SCRATCH_DIR
        self.file_registry   = {}
        self.folder_registry = set()
        self.registry_file   = config.WORKSPACE_REGISTRY_FILE
        self._load_registry()

        # BM25 index
        self._bm25_index  = {}     # rel_path → raw file content
        self._bm25_model  = None   # fitted BM25Okapi instance
        self._bm25_paths  = []     # ordered list matching BM25 corpus rows
        self._bm25_dirty  = False  # True when index needs rebuild

        # Registry dirty flag — set True whenever file_registry / folder_registry
        # changes. Flushed to disk in one atomic write by flush_registry().
        # This batches all writes from a reconcile cycle into a single I/O op.
        self._registry_dirty = False

        # Path to the BM25 corpus pickle cache (tokenised text, not the model).
        self._bm25_cache_path = os.path.join(
            os.path.dirname(config.WORKSPACE_REGISTRY_FILE), "bm25_corpus.pkl"
        )


    # ══════════════════════════════════════════════════════════════════════════
    # FILE & FOLDER TRACKING
    # ══════════════════════════════════════════════════════════════════════════

    def _is_project_file(self, filepath: str) -> bool:
        """Return False for system-generated temp files (task_XXXXXXXX.py/.sh)."""
        filename = os.path.basename(filepath)
        return not bool(re.match(r'^task_[a-f0-9]{8}\.(py|sh)$', filename))

    def _normalize_rel_path(self, filepath: str) -> str:
        """
        Always return a forward-slash relative path from workspace root.
        Uses os.path.relpath so it works correctly on Windows regardless
        of drive-letter casing — no fragile startswith() check needed.
        """
        rel = os.path.relpath(filepath, self.workspace_path).replace('\\', '/')
        # On Windows, os.path.relpath returns an absolute path (e.g. "D:/...") when
        # filepath is on a different drive than workspace_path — isabs() catches this.
        if os.path.isabs(rel):
            raise ValueError(f"Path traversal detected: {filepath} is on a different drive from the workspace.")
        if rel.startswith('../') or rel == '..':
            raise ValueError(f"Path traversal detected: {filepath} escapes workspace root.")
        return rel

    def track_folder(self, folder_path: str, _internal: bool = False) -> dict:
        """Track a folder creation.

        Args:
            _internal: When True (called from track_file_write), suppress the
                       immediate _save_registry() call. The caller is responsible
                       for flushing via flush_registry() or _save_registry().
        """
        try:
            rel_path = self._normalize_rel_path(folder_path)
        except ValueError:
            # Path is outside the scratch workspace (e.g. /outputs/) — silently skip.
            return {"success": True, "path": folder_path, "new": False}

        try:
            if rel_path not in self.folder_registry:
                self.folder_registry.add(rel_path)
                self._registry_dirty = True
                if not _internal:
                    self._save_registry()
                    self._registry_dirty = False
                console.print(f"[dim]Tracked folder: {rel_path}[/dim]")
                return {"success": True, "path": rel_path, "new": True}

            return {"success": True, "path": rel_path, "new": False}

        except Exception as e:
            console.print(f"⚠️  [yellow]workspace_tracker.track_folder failed for '{folder_path}': {e}[/yellow]")
            return {"success": False, "error": str(e)}

    def track_file_write(self, filepath: str, content: str) -> None:
        """Track a file write — updates registry and BM25 index."""
        try:
            rel_path = self._normalize_rel_path(filepath)
        except ValueError:
            # File is outside the scratch workspace (e.g. /outputs/) — silently skip.
            return

        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        if not self._is_project_file(rel_path):
            return

        # Track parent folder (suppress inner save — we save once below)
        parent_folder = os.path.dirname(rel_path)
        if parent_folder and parent_folder != '.':
            self.track_folder(os.path.join(self.workspace_path, parent_folder), _internal=True)

        meta = {
            "path":          rel_path,
            "hash":          content_hash,
            "last_modified": os.path.getmtime(filepath) if os.path.exists(filepath) else time.time(),
            "size":          len(content.encode("utf-8")),  # byte length matches os.path.getsize()
            "line_count":    content.count('\n') + 1,
            "language":      detect_language(rel_path)
        }

        is_duplicate = (
            rel_path in self.file_registry
            and self.file_registry[rel_path]["hash"] == content_hash
        )

        # Update registry and mark dirty — flush below (and again at end of reconcile)
        self.file_registry[rel_path] = meta
        self._registry_dirty = True

        # Update BM25 index (store full content — no token limit)
        # Bug #33 fix: only skip .txt files that have a paired .chunks.json sidecar
        # (i.e. they're genuinely ingested PDFs covered by doc_search). A plain
        # notes.txt or script.txt without a sidecar deserves BM25 coverage too.
        _is_ingested_txt = (
            rel_path.lower().endswith('.txt')
            and os.path.exists(os.path.join(self.workspace_path, rel_path[:-4] + ".chunks.json"))
        )
        if _SEARCH_AVAILABLE and not is_duplicate and not _is_ingested_txt:
            self._bm25_index[rel_path] = content
            self._bm25_dirty = True

        # Persist immediately so a crash between tool calls can't lose tracking data.
        # flush_registry() is a no-op if called again inside reconcile_workspace().
        self.flush_registry()



    def remove_file(self, filepath: str):
        """
        Remove a file from the tracker and BM25 index.
        MUST be called whenever a file is deleted from disk.
        Without this, deleted files keep appearing in search results forever.
        """
        try:
            rel_path = self._normalize_rel_path(filepath)
        except ValueError:
            # Path is outside the tracked workspace (e.g. a stale registry
            # entry pointing at a deleted drive) — there is nothing in the
            # registry or BM25 index to remove, so this is a no-op rather
            # than a crash. Mirrors the identical guard in track_file_write.
            console.print(f"[dim]remove_file: '{filepath}' is outside the workspace, skipping.[/dim]")
            return

        # Cleanup orphaned chunks file if it's a txt file
        if rel_path.lower().endswith('.txt'):
            chunks_path = os.path.join(self.workspace_path, rel_path[:-4] + ".chunks.json")
            if os.path.exists(chunks_path):
                try:
                    os.remove(chunks_path)
                    console.print(f"[dim]Deleted orphaned chunks: {os.path.basename(chunks_path)}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Failed to delete chunks {chunks_path}: {e}[/yellow]")

        removed = False
        if rel_path in self.file_registry:
            del self.file_registry[rel_path]
            removed = True
        if rel_path in self._bm25_index:
            del self._bm25_index[rel_path]
            self._bm25_dirty = True
            removed = True
        if removed:
            self._save_registry()
            self._registry_dirty = False
            console.print(f"[dim]Untracked: {rel_path}[/dim]")


    # ══════════════════════════════════════════════════════════════════════════
    # WORKSPACE SUMMARY & TREE
    # ══════════════════════════════════════════════════════════════════════════

    def get_workspace_summary(self, max_files: int = 20) -> str:
        """Get workspace summary with folder tree and recent files."""
        if not self.file_registry and not self.folder_registry:
            return "[WORKSPACE] Empty - no files or folders created yet."

        summary   = "[WORKSPACE STRUCTURE]\n\n"
        all_paths = list(self.folder_registry) + list(self.file_registry.keys())
        summary  += self._build_tree(all_paths, file_meta=self.file_registry)

        project_files = {k: v for k, v in self.file_registry.items() if self._is_project_file(k)}

        if project_files:
            recent_files = sorted(project_files.values(), key=lambda x: x['last_modified'], reverse=True)[:5]
            summary += "\n[RECENT FILES]\n"
            for meta in recent_files:
                age_sec = time.time() - meta['last_modified']
                if age_sec < 60:
                    age = "just now"
                elif age_sec < 3600:
                    age = f"{int(age_sec/60)}m ago"
                elif age_sec < 86400:
                    age = f"{int(age_sec/3600)}h ago"
                else:
                    age = f"{int(age_sec/86400)}d ago"
                summary += f"  📄 {meta['path']} ({meta['language']}, {meta['line_count']} lines, {age})\n"

        return summary

    def _build_tree(self, paths: list, file_meta: dict = None) -> str:
        """Build ASCII tree structure from paths, annotating files with line counts."""
        if not paths:
            return "(empty)\n"

        paths           = sorted(set([p.replace('\\', '/') for p in paths]))
        path_parts_list = [p.split('/') for p in paths]
        tree_lines      = []
        file_meta       = file_meta or {}

        def add_tree_node(parts, prefix="", parent=""):
            groups = {}
            for p in parts:
                if p:
                    first = p[0]
                    rest  = p[1:]
                    if first not in groups:
                        groups[first] = []
                    if rest:
                        groups[first].append(rest)

            for i, item in enumerate(sorted(groups.keys())):
                is_last   = (i == len(groups) - 1)
                connector = "└── " if is_last else "├── "
                rel_path  = f"{parent}/{item}".lstrip("/")
                # Bug 31: also check folder_registry so empty tracked directories
                # are rendered with a trailing '/' rather than as plain files.
                is_folder = any(groups[item]) or (rel_path in self.folder_registry)

                if not is_folder and rel_path in file_meta:
                    lines = file_meta[rel_path].get("line_count", 0)
                    tree_lines.append(f"{prefix}{connector}{item}  ({lines:,} lines)")
                else:
                    tree_lines.append(f"{prefix}{connector}{item}{'/' if is_folder else ''}")

                new_prefix = prefix + ("    " if is_last else "│   ")
                if groups[item]:
                    add_tree_node(groups[item], new_prefix, rel_path)

        add_tree_node(path_parts_list)
        return "\n".join(tree_lines) + "\n"

    # ══════════════════════════════════════════════════════════════════════════
    # BM25 SEMANTIC SEARCH
    # ══════════════════════════════════════════════════════════════════════════

    def semantic_search(self, query: str, top_k: int = 5, min_score: float = 0.01,
                         file_filter: str = None) -> list:
        """
        BM25 semantic search across all tracked file contents.

        Why BM25 over TF-IDF for code search:
        - Term saturation: BM25 dampens the effect of a word appearing 50+
          times in a file (TF-IDF over-rewards pure repetition).
        - Document length normalization: long files aren't unfairly boosted
          just because they contain more tokens overall.
        - Same code-aware tokenizer as before — snake_case / camelCase aware.
        - No token limit, no model download, pure Python, ~5ms rebuild.
        - min_score=0.01: BM25 scores are raw (not normalized 0-1), so the
          threshold is low — non-zero score already means at least one query
          token matched. Scores of 1-20+ are typical strong matches.

        Args:
            query:       Natural language or keyword query
            top_k:       Max results to return
            min_score:   Minimum BM25 score threshold (filters zero-match files)
            file_filter: Optional glob filter e.g. '*.py' matched against each
                         candidate's basename. Mirrors CodeSearchAgent.search()'s
                         file_filter so callers see consistent filtering whether
                         they hit the primary semantic index or this BM25 fallback.

        Returns:
            List of dicts: {path, score, preview, language}
        """
        if not _SEARCH_AVAILABLE:
            return []

        # Hydrate index from disk for files loaded from registry at startup
        self._ensure_index_loaded()

        if not self._bm25_index:
            return []

        # Rebuild BM25 model when corpus has changed
        if self._bm25_dirty or self._bm25_model is None:
            self._rebuild_bm25()

        if self._bm25_model is None:
            return []

        try:
            query_tokens = _code_tokenizer(query)
            if not query_tokens:
                return []
            scores = self._bm25_model.get_scores(query_tokens)
        except Exception as e:
            console.print(f"⚠️  [yellow]BM25 search query failed: {e}[/yellow]")
            return []

        # Rank by score descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked:
            if len(results) >= top_k:
                break

            score = float(scores[idx])
            if score < min_score:
                break  # scores are sorted descending — safe early exit

            rel_path = self._bm25_paths[idx]

            # file_filter: skip non-matching candidates but keep scanning —
            # ranked is globally sorted by score, so a filtered-out doc here
            # doesn't mean lower-ranked docs should also be skipped.
            if file_filter and not fnmatch.fnmatch(os.path.basename(rel_path), file_filter):
                continue

            content  = self._bm25_index.get(rel_path, "")

            # Lazily hydrate from disk if content is a cache stub (empty string).
            # This avoids blank previews after a BM25 cache restore.
            if not content:
                _full = os.path.join(self.workspace_path, rel_path)
                try:
                    with open(_full, "r", encoding="utf-8", errors="ignore") as _f:
                        content = _f.read()
                except Exception:
                    content = ""

            # Preview: first 3 meaningful non-comment lines
            preview_lines = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                    preview_lines.append(stripped)
                if len(preview_lines) >= 3:
                    break

            results.append({
                "path":     rel_path,
                "score":    round(score, 4),
                "preview":  " | ".join(preview_lines)[:200],
                "language": detect_language(rel_path),
            })

        return results

    def _ensure_index_loaded(self):
        """
        Load file contents from disk for any tracked files not yet in memory.
        On first call after a process restart, tries to restore the tokenised
        corpus from the pickle cache before falling back to full file reads.
        """
        # Fast path: try pickle cache first (avoids reading every file from disk)
        if not self._bm25_index and os.path.exists(self._bm25_cache_path):
            try:
                import pickle
                registry_mtime  = os.path.getmtime(self.registry_file) if os.path.exists(self.registry_file) else 0
                cache_mtime     = os.path.getmtime(self._bm25_cache_path)
                if cache_mtime >= registry_mtime:
                    with open(self._bm25_cache_path, "rb") as _pf:
                        cached = pickle.load(_pf)
                    paths  = cached.get("paths", [])
                    corpus = cached.get("corpus", [])
                    if paths and corpus and len(paths) == len(corpus):
                        self._bm25_paths = paths
                        # Re-index content as empty strings — enough for cache hit;
                        # actual content loaded lazily if a file changes later.
                        self._bm25_index = {p: "" for p in paths}
                        self._bm25_model = BM25Okapi(corpus)
                        self._bm25_dirty = False
                        console.print(f"[dim]BM25 restored from cache: {len(paths)} files[/dim]")
                        return
            except Exception:
                pass  # fall through to normal load

        for rel_path in self.file_registry:
            if rel_path not in self._bm25_index:
                # Skip .txt files — these are ingested PDF documents, not code.
                # doc_search (chunk embeddings) handles retrieval for those.
                if rel_path.lower().endswith('.txt'):
                    sidecar = os.path.join(self.workspace_path, rel_path[:-4] + ".chunks.json")
                    if os.path.exists(sidecar):
                        continue
                full_path = os.path.join(self.workspace_path, rel_path)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        self._bm25_index[rel_path] = f.read()
                    self._bm25_dirty = True
                except Exception as e:
                    console.print(f"⚠️  [yellow]workspace_tracker: could not read '{rel_path}' for index: {e}[/yellow]")

    def _rebuild_bm25(self):
        """
        Tokenize the full corpus and fit a fresh BM25Okapi instance.
        Persists the tokenised corpus to a pickle cache so the next
        process restart can skip re-reading all files from disk.
        """
        if not self._bm25_index:
            return

        # Bug 24: Hydrate any cache-stub entries (content == "") before tokenising.
        # When the BM25 index is restored from pickle, all content strings are set
        # to "" as an optimisation. If a workspace change then triggers a rebuild,
        # those stubs would produce empty token lists, corrupting the corpus.
        for _rel, _content in list(self._bm25_index.items()):
            if not _content:
                _full_path = os.path.join(self.workspace_path, _rel)
                try:
                    with open(_full_path, "r", encoding="utf-8", errors="ignore") as _f:
                        self._bm25_index[_rel] = _f.read()
                except Exception:
                    pass  # leave as "" — produces empty token list (acceptable)

        try:
            self._bm25_paths     = list(self._bm25_index.keys())
            tokenized_corpus     = [_code_tokenizer(self._bm25_index[p]) for p in self._bm25_paths]
            self._bm25_model     = BM25Okapi(tokenized_corpus)
            self._bm25_dirty     = False

            total_terms = sum(len(doc) for doc in tokenized_corpus)
            console.print(
                f"[dim]BM25 index rebuilt: {len(self._bm25_paths)} files, "
                f"{total_terms} total tokens[/dim]"
            )

            # Persist corpus so next startup can skip re-tokenising
            try:
                import pickle
                with open(self._bm25_cache_path, "wb") as _pf:
                    pickle.dump({"paths": self._bm25_paths, "corpus": tokenized_corpus}, _pf,
                                protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass  # non-fatal — worst case we rebuild on next restart

        except Exception as e:
            console.print(f"⚠️  [yellow]BM25 rebuild failed: {e}[/yellow]")
            self._bm25_model = None

    # ══════════════════════════════════════════════════════════════════════════
    # FILESYSTEM RECONCILER
    # ══════════════════════════════════════════════════════════════════════════

    # Built on top of the shared utils.CODE_EXTENSIONS base (also used by
    # tool_handlers.UPLOAD_EXTENSIONS) plus ".env", which reconcile treats as
    # trackable but which has no associated syntax-highlighting language.
    _RECONCILE_EXTENSIONS = CODE_EXTENSIONS | {".env"}

    def reconcile_workspace(self) -> dict:
        """
        Sync tracker registry against what's actually on disk.

        Called once per turn before workspace summary is built. Detects:
          - Added:    file on disk but unknown to registry  → index it
          - Removed:  file in registry but gone from disk   → untrack it
          - Modified: file in both but mtime newer          → re-index

        Returns:
            dict with counts: {added, removed, modified, unchanged}
        """
        counts = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}

        if not os.path.isdir(self.workspace_path):
            return counts

        # Build ground truth: what's on disk
        on_disk = {}
        on_disk_dirs = set()
        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in EXCL_DIRS]
            
            # Bug 26 fix: track all visible directories to prune deleted ones from folder_registry
            for d in dirs:
                full_d = os.path.join(root, d)
                rel_d = os.path.relpath(full_d, self.workspace_path).replace('\\', '/')
                on_disk_dirs.add(rel_d)
                
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self._RECONCILE_EXTENSIONS:
                    continue
                if re.match(r'^task_[a-f0-9]{8}\.(py|sh)$', fn):
                    continue
                if fn.endswith('.chunks.json') or fn.endswith('.embed.json'):
                    continue
                full = os.path.join(root, fn)
                rel  = os.path.relpath(full, self.workspace_path).replace('\\', '/')
                try:
                    on_disk[rel] = os.path.getmtime(full)
                except OSError:
                    pass

        known = set(self.file_registry.keys())
        disk  = set(on_disk.keys())

        for rel in known - disk:
            self.remove_file(os.path.join(self.workspace_path, rel))
            counts["removed"] += 1

        for rel in disk:
            full  = os.path.join(self.workspace_path, rel)
            mtime = on_disk[rel]

            if rel not in self.file_registry:
                self._reconcile_ingest(rel, full)
                counts["added"] += 1
            else:
                current_record = self.file_registry[rel]
                last_indexed = current_record.get("last_modified", 0)
                
                if mtime > last_indexed + 0.5:
                    self._reconcile_ingest(rel, full)
                    counts["modified"] += 1
                else:
                    size_changed = os.path.getsize(full) != current_record.get("size", -1)
                    if size_changed:
                        self._reconcile_ingest(rel, full)
                        counts["modified"] += 1
                    else:
                        # mtime + size match → treat as unchanged (MD5 check was too expensive per turn).
                        counts["unchanged"] += 1

        # Bug 26 fix: Phantom Deleted Folders Persist
        stale_dirs = self.folder_registry - on_disk_dirs
        if stale_dirs:
            self.folder_registry -= stale_dirs
            self._registry_dirty = True

        if counts["added"] or counts["removed"] or counts["modified"] or stale_dirs:
            console.print(
                f"[dim]Workspace reconciled — "
                f"+{counts['added']} added, "
                f"-{counts['removed']} removed, "
                f"~{counts['modified']} modified[/dim]"
            )

        # Flush all registry changes accumulated during reconcile in one write
        self.flush_registry()

        return counts

    def flush_registry(self):
        """Write registry to disk only if it changed since last save."""
        if self._registry_dirty:
            self._save_registry()
            self._registry_dirty = False

    def _reconcile_ingest(self, rel: str, full: str):
        """Read a file from disk and sync into tracker index."""
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            console.print(f"[dim yellow]reconcile: could not read '{rel}': {e}[/dim yellow]")
            return

        self.track_file_write(full, content)

    # ══════════════════════════════════════════════════════════════════════════
    # RESET & PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def invalidate_bm25_cache(self):
        """
        Bug 11: clear the in-memory BM25 index AND delete the on-disk pickle
        cache. Previously only the in-memory index was cleared on /restore,
        leaving bm25_cache.pkl on disk. Since /restore overwrites the registry
        file with an older snapshot (older mtime) while the pickle keeps its
        newer mtime from more recent turns, _ensure_index_loaded()'s
        `cache_mtime >= registry_mtime` freshness check would treat the
        stale pickle as valid and reload phantom file paths from turns that
        no longer exist post-restore. Explicit deletion here can't be fooled
        by mtime quirks the way a timestamp comparison can.
        """
        self._bm25_index.clear()
        self._bm25_dirty = True
        try:
            if os.path.exists(self._bm25_cache_path):
                os.remove(self._bm25_cache_path)
        except Exception as e:
            console.print(f"[yellow]⚠️  could not remove stale bm25 cache: {e}[/yellow]")

    def reset_workspace(self):
        """Clear all tracked files, folders, BM25 index, and chunk embeddings."""
        self.file_registry.clear()
        self.folder_registry.clear()
        self._bm25_index.clear()
        self._bm25_model  = None
        self._bm25_paths  = []
        self._bm25_dirty  = False
        self._save_registry()

        # Delete all .chunks.json embedding files so doc_search doesn't return
        # passages from previous sessions after /reset.
        pattern = os.path.join(self.workspace_path, "**", "*.chunks.json")
        deleted = 0
        for f in glob.glob(pattern, recursive=True):
            try:
                os.remove(f)
                deleted += 1
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not delete {f}: {e}[/yellow]")
        if deleted:
            console.print(f"[dim]Deleted {deleted} chunk embedding file(s)[/dim]")

        console.print("[dim]Workspace tracker reset[/dim]")

    def _load_registry(self):
        """Load file/folder registry from disk."""
        if not os.path.exists(self.registry_file):
            return
        data = load_json(self.registry_file, default={})
        self.file_registry   = data.get('files', {})
        self.folder_registry = set(data.get('folders', []))
        console.print(f"📄 [dim]Loaded {len(self.file_registry)} files, {len(self.folder_registry)} folders[/dim]")

    def _save_registry(self):
        """Save file/folder registry to disk."""
        save_json_atomic(self.registry_file, {
            'files':   self.file_registry,
            'folders': list(self.folder_registry),
        })