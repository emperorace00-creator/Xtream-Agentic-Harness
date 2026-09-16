# file_ops_agent.py - Surgical File Operations
import os
import re
import fnmatch
from typing import Optional, List
from rich.syntax import Syntax
from rich.panel import Panel
from utils import detect_language, compute_view_window, console
import config


class FileOpsAgent:
    """
    Surgical file operations - str_replace edits instead of full file regeneration.

    Primary tool: str_replace - modify only what's needed.
    """

    def __init__(self, workspace_root: str = None, workspace_tracker=None, code_search_agent=None):
        self.workspace_root = workspace_root if workspace_root is not None else config.SCRATCH_DIR
        self.tracker = workspace_tracker
        self.code_search_agent = code_search_agent

    def search_workspace(self, query: str, file_filter: str = None, semantic: bool = False) -> str:
        """
        Search for a string in all workspace files (Scratch + Context).

        Args:
            query:       Exact string/regex (semantic=False) or concept (semantic=True)
            file_filter: Optional glob filter e.g. '*.py' - used in grep mode and
                         Codestral Embed semantic search
            semantic:    True = Codestral Embed concept search (BM25 fallback),
                         False = grep (default)
        """

        # ── Semantic mode ──────────────────────────────────────────────────────
        if semantic:
            # Primary: Codestral Embed semantic code search
            if self.code_search_agent:
                try:
                    return self.code_search_agent.search(query, top_k=8, file_filter=file_filter)
                except Exception as e:
                    console.print(f"[yellow]⚠️ Semantic code search failed: {e} — falling back to BM25[/yellow]")

            # Fallback: BM25 keyword search (if embed API unreachable)
            # min_score=0.01: BM25 scores are raw (not normalized 0–1), so any
            # non-zero score means at least one query token matched. Strong matches
            # score 1–20+. Setting floor to 0.01 filters only true zero-match files.
            if self.tracker:
                results = self.tracker.semantic_search(query, top_k=8, min_score=0.01,
                                                         file_filter=file_filter)
                if not results:
                    return (
                        f"No semantically related files found for '{query}'.\n"
                        f"Tip: try grep mode (semantic=False) for exact string matches.\n"
                        f"[SYSTEM: Note — used keyword search (BM25) as fallback. Semantic index unavailable.]"
                    )
                lines = [f"[SYSTEM: SEMANTIC SEARCH: '{query}'] — {len(results)} result(s)\n"]
                for r in results:
                    lines.append(
                        f"  📄 {r['path']}  (score: {r['score']}, {r['language']})\n"
                        f"     {r['preview']}"
                    )
                lines.append("[SYSTEM: Note — used keyword search (BM25) as fallback. Semantic index unavailable.]")
                return "\n".join(lines)
            return "Semantic search is disabled (no tracker attached)."

        # ── Grep mode (original behaviour) ────────────────────────────────────
        results = []
        search_dirs = [config.SCRATCH_DIR, config.UPLOADS_FOLDER] 
        
        pattern = file_filter if file_filter else "*"
        
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            
            dir_label = "[SCRATCH]" if s_dir == config.SCRATCH_DIR else "[UPLOADS]"
            
            # os.walk handles nested folders automatically
            # Bug #13 fix: capture dirs and prune heavy/excluded subtrees
            # (node_modules, .git, venv etc.) so grep can't flood context.
            from utils import EXCL_DIRS, EXCL_PREFIXES, EXCL_SUFFIXES
            for root, dirs, files in os.walk(s_dir):
                dirs[:] = [
                    d for d in dirs
                    if d not in EXCL_DIRS
                    and not any(d.startswith(p) for p in EXCL_PREFIXES)
                    and not any(d.endswith(s) for s in EXCL_SUFFIXES)
                ]
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        filepath = os.path.join(root, filename)
                        rel_path = os.path.relpath(filepath, s_dir)
                        # Normalize path separators
                        rel_path = rel_path.replace('\\', '/')
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                for i, line in enumerate(f, 1):
                                    if query.lower() in line.lower():
                                        clean_line = line.strip()[:200]
                                        results.append(f"{dir_label} {rel_path}:{i}: {clean_line}")
                        except Exception:
                            continue
        
        if not results:
            return f"No matches found for '{query}' in Scratch or Uploads."
        
        # Limit results
        if len(results) > 50:
            return "\n".join(results[:50]) + f"\n\n... and {len(results)-50} more matches."
        
        return "\n".join(results)

    def str_replace(self, filepath: str, old_str: str, new_str: str,
                     verify: bool = True, count: int = 1) -> dict:
        """
        Replace occurrences of old_str with new_str in a file.

        Args:
            filepath:  Path to file (absolute or relative to workspace).
            old_str:   Exact string to find.
            new_str:   Replacement string.
            verify:    If True, print a diff summary (edit is always applied).
            count:     How many occurrences to replace.
                         1   - default; fails if old_str is not unique (original behaviour).
                         0   - replace ALL occurrences (pass count="all" from XML).
                         N>1 - replace the first N occurrences.

        Returns:
            dict with success, changes (replaced count), diff, error.
        """
        try:
            full_path = self._resolve_path(filepath)

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_str not in content:
                _preview = old_str[:100] + ("..." if len(old_str) > 100 else "")
                return {
                    "success": False,
                    "error": f"String not found in file: '{_preview}'",
                    "suggestion": "Use search_in_file() to find the correct string first"
                }

            occurrences = content.count(old_str)

            # count=1 keeps the original strict-uniqueness guarantee
            if count == 1 and occurrences > 1:
                return {
                    "success": False,
                    "error": f"String appears {occurrences} times in file. Must be unique for count=1.",
                    "suggestion": "Make old_str more specific, or pass <count>all</count> to replace every occurrence.",
                    "preview": self._show_occurrences(content, old_str)
                }

            # Determine actual replacement count
            if count == 0:                              # "all"
                new_content = content.replace(old_str, new_str)
                replaced = occurrences
            else:                                       # 1 or first-N
                actual = min(count, occurrences)
                new_content = content.replace(old_str, new_str, actual)
                replaced = actual

            diff = self._generate_diff(old_str, new_str, filepath, replaced=replaced)

            if verify:
                console.print(Panel(diff, title="[cyan]Preview Changes[/cyan]", border_style="cyan"))

            tmp_path = full_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            os.replace(tmp_path, full_path)

            if replaced == 1:
                console.print(f"✅ [green]Surgical edit applied to {os.path.basename(filepath)}[/green]")
            else:
                console.print(f"✅ [green]{replaced} replacement(s) applied to {os.path.basename(filepath)}[/green]")

            # Update BM25 semantic search index with new content
            if self.tracker:
                self.tracker.track_file_write(full_path, new_content)

            return {
                "success": True,
                "file": filepath,
                "changes": {
                    "replaced": replaced,
                    "removed": (old_str.count('\n') + 1) * replaced,
                    "added": (new_str.count('\n') + 1) * replaced,
                },
                "diff": diff,
                "message": f"{replaced} occurrence(s) replaced"
            }

        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {filepath}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def view_lines(self, filepath: str, start: int = 1, end: Optional[int] = None, context: int = 5) -> dict:
        """
        View a specific line range from a file.

        Args:
            filepath: Path to file
            start: Starting line (1-indexed)
            end: Ending line (None = to end of file)
            context: Extra lines to show before/after
        """
        try:
            full_path = self._resolve_path(filepath)

            with open(full_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            start_idx, end_idx = compute_view_window(total_lines, start, end, context)
            selected_lines = all_lines[start_idx:end_idx]

            formatted = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted.append(f"{i:4d} | {line.rstrip()}")

            content = "\n".join(formatted)

            syntax = Syntax(
                content,
                detect_language(filepath),
                theme="catppuccin-mocha",
                line_numbers=False,
                word_wrap=False
            )
            console.print(Panel(
                syntax,
                title=f"[cyan]{os.path.basename(filepath)}[/cyan] (lines {start_idx + 1}-{end_idx})",
                border_style="cyan"
            ))

            return {
                "success": True,
                "file": filepath,
                "total_lines": total_lines,
                "view_range": (start_idx + 1, end_idx),
                "content": content,
                "lines": selected_lines,
                "message": f"Viewing {len(selected_lines)} lines (context ±{context})"
            }

        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {filepath}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_in_file(self, filepath: str, pattern: str, regex: bool = False,
                       context_lines: int = 3, max_results: int = 10) -> dict:
        """
        Search for a pattern in a file (like grep).

        Args:
            filepath: Path to file
            pattern: String or regex to search for
            regex: Treat pattern as regex. (If the regex is invalid, the 
                   search safely falls back to a literal string match).
            context_lines: Lines of context around each match
            max_results: Max matches to return
        """
        try:
            full_path = self._resolve_path(filepath)

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            matches = []

            for i, line in enumerate(lines, start=1):
                try:
                    matched = re.search(pattern, line) if regex else (pattern.lower() in line.lower())
                except re.error:
                    # Model passed an invalid regex (e.g. unbalanced parenthesis like
                    # "self.store_version(" - fall back to literal string search so the
                    # tool returns useful results instead of an exception dict.
                    matched = pattern in line
                if matched:
                    matches.append(self._extract_match_context(lines, i, pattern, context_lines))
                if len(matches) >= max_results:
                    break

            if not matches:
                return {
                    "success": True,
                    "file": filepath,
                    "pattern": pattern,
                    "matches": [],
                    "message": "No matches found"
                }

            console.print(f"\n🔍 [cyan]Found {len(matches)} match(es) in {os.path.basename(filepath)}[/cyan]\n")
            for match in matches:
                console.print(f"[yellow]Line {match['line']}:[/yellow]")
                console.print(match['context'])
                console.print()

            return {
                "success": True,
                "file": filepath,
                "pattern": pattern,
                "total_matches": len(matches),
                "matches": matches,
                "message": f"Found {len(matches)} match(es)"
            }

        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {filepath}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Helper methods ────────────────────────────────────────────────────

    def _resolve_path(self, filepath: str) -> str:
        """Resolve file path, translating container paths to host paths first.
        Prevents directory traversal escaping the workspace root.
        Allows paths inside config.OUTPUTS_DIR in addition to SCRATCH_DIR/workspace
        so that str_replace and show_image work on /outputs/ files."""
        filepath = config.container_to_host_path(filepath)
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.workspace_root, filepath)

        resolved      = os.path.realpath(filepath)
        workspace_abs = os.path.realpath(self.workspace_root)
        outputs_abs   = os.path.realpath(config.OUTPUTS_DIR)

        in_workspace = (resolved == workspace_abs or
                        resolved.startswith(workspace_abs + os.sep))
        in_outputs   = (resolved == outputs_abs or
                        resolved.startswith(outputs_abs + os.sep))

        if not (in_workspace or in_outputs):
            raise PermissionError(
                f"Access denied: path traversal outside workspace/outputs is forbidden ({filepath})"
            )
        return resolved

    def _generate_diff(self, old_str: str, new_str: str, filename: str, replaced: int = 1) -> str:
        """Generate a line-count diff summary of the replaced region."""
        removed = (old_str.count('\n') + 1) * replaced
        added   = (new_str.count('\n') + 1) * replaced
        return f"{filename}: -{removed} lines / +{added} lines"

    def _show_occurrences(self, content: str, pattern: str) -> str:
        """Show all occurrences of pattern with line numbers."""
        occurrences = []
        start = 0
        while len(occurrences) < 10:
            idx = content.find(pattern, start)
            if idx == -1:
                break
            line_num = content[:idx].count('\n') + 1
            first_line = pattern.split('\n')[0].strip()[:80]
            occurrences.append(f"Line {line_num}: {first_line}")
            start = idx + len(pattern)
        suffix = f"\n... ({len(occurrences)} total)" if len(occurrences) == 10 else ""
        return '\n'.join(occurrences[:5]) + suffix

    def _extract_match_context(self, lines: List[str], match_line: int, pattern: str, context: int) -> dict:
        """Extract context around a matched line."""
        start = max(0, match_line - 1 - context)
        end = min(len(lines), match_line + context)

        context_lines = []
        for i in range(start, end):
            prefix = ">>>" if i == match_line - 1 else "   "
            context_lines.append(f"{prefix} {i + 1:4d} | {lines[i].rstrip()}")

        return {
            "line": match_line,
            "text": lines[match_line - 1].strip(),
            "context": '\n'.join(context_lines)
        }