# tool_summarizer.py - Deterministic tool call summarizer for compact history storage
"""
Converts raw tool call chains into compact, human-readable audit summaries.
No LLM needed - tool calls are structured data, not natural language.

Example output stored in history:
  [POWERS USED - 3 call(s)]
  1. bash(python start.py) → ✓ exit 0
  2. str_replace(api/start.py) → ✓ (3 lines → 7 lines)
  3. doc_search('photosynthesis') → 4 passage(s) found
"""

import json
import re
from utils import console


class ToolCallSummarizer:
    """
    Deterministically summarize tool call chains for compact history storage.

    Design principle: extract only what the model needs to remember across sessions -
    WHAT was done and WHETHER it succeeded. Not the full content.
    """

    def summarize_turn(self, tool_call_log: list) -> str:
        """
        Convert a logged list of (tool_name, args, result) tuples into a compact summary.

        Args:
            tool_call_log: List of dicts with keys: fn, args, result

        Returns:
            Compact multi-line summary string, or "" if no tools were called.
        """
        if not tool_call_log:
            return ""

        lines = [f"[POWERS USED — {len(tool_call_log)} call(s)]"]

        for i, entry in enumerate(tool_call_log, 1):
            fn = entry.get("fn", "unknown")
            args = entry.get("args", {})
            result = entry.get("result", "")
            summary = self._summarize_one(fn, args, result)
            lines.append(f"  {i}. {summary}")

        return "\n".join(lines)

    def _summarize_one(self, fn: str, args: dict, result: str) -> str:
        """Dispatch to per-tool summarizer."""
        try:
            dispatch = {
                "url_search":             self._url_search,
                "quick_search":           self._quick_search,
                "search_history":         self._search_history,
                "str_replace":            self._str_replace,
                "view_lines":             self._view_lines,
                "search_in_file":         self._search_in_file,
                "workspace_search":       self._workspace_search,
                "doc_search":             self._doc_search,
                "ingest_pdf":             self._ingest_pdf,
                "ingest_chat":            self._ingest_chat,
                "ingest_text":            self._ingest_text,
                "bash":                   self._bash,
                "show_image":             self._show_image,
                "search_semantic_scholar": self._search_semantic_scholar,
            }

            handler = dispatch.get(fn)
            if handler:
                return handler(args, result)

            # Generic fallback
            return self._generic(fn, args, result)

        except Exception as e:
            console.print(f"⚠️  [yellow]tool_summarizer: failed to summarize '{fn}': {e}[/yellow]")
            return f"{fn}(...) → [summary error]"

    # Per-tool handlers

    def _url_search(self, args: dict, result: str) -> str:
        """Check for [SOURCE:] prefix to determine success."""
        url = str(args.get("url", "")).strip()[:70]
        context = str(args.get("context", "")).strip()[:30]
        success = result.lstrip().startswith("[SOURCE:")
        icon = "✓" if success else "✗"
        ctx_str = f", ctx='{context}'" if context else ""
        return f"url_search({url}{ctx_str}) → {icon}"

    def _quick_search(self, args: dict, result: str) -> str:
        query = str(args.get("query", "")).strip()[:70]
        icon = "✓" if "[CONTENT]:" in result else "✗"
        return f"quick_search('{query}') → {icon}"

    def _search_history(self, args: dict, result: str) -> str:
        """Summarise search_history call by counting result lines."""
        query = str(args.get("query", "")).strip()[:60]
        stripped = result.lstrip()
        no_hit = stripped.startswith("[search_history]") or stripped.startswith("[ERROR]")
        if no_hit:
            return f"search_history('{query}') → ✗ ({result.strip()[:80]})"
        n = len(re.findall(r'\n\[\d+\]', result))
        return f"search_history('{query}') → ✓ {n} result(s)"

    def _str_replace(self, args: dict, result: str) -> str:
        filepath = args.get("file", "?")
        success = '"success": true' in result.lower() or '"success":true' in result.lower()
        icon = "✓" if success else "✗"

        # Only attempt the line-count detail on success - a failure dict has
        # no "changes" key, so this used to fall back to "?" for both counts
        # and print "(? lines -> ? lines)" as if a real edit had happened.
        detail = ""
        if success:
            try:
                parsed = json.loads(result)
                changes = parsed.get("changes")
                if changes:
                    removed = changes.get("removed", "?")
                    added = changes.get("added", "?")
                    detail = f" ({removed} lines → {added} lines)"
            except Exception:
                pass

        return f"str_replace({filepath}) → {icon}{detail}"

    def _view_lines(self, args: dict, result: str) -> str:
        """
        Summarize a view_lines tool call, returning the file path, line range,
        and extracting the success/failure status from the JSON result.
        """
        filepath = args.get("file", "?")
        start = args.get("start", "?")
        end = args.get("end", "?")
        try:
            success = bool(json.loads(result).get("success", False))
            icon = "✓" if success else "✗"
        except Exception:
            icon = "?"
        return f"view_lines({filepath}, {start}–{end}) → {icon}"

    def _search_in_file(self, args: dict, result: str) -> str:
        filepath = args.get("file", "?")
        pattern = str(args.get("pattern", ""))[:40]
        try:
            parsed = json.loads(result)
            n = parsed.get("total_matches", "?")
        except Exception:
            n = "?"
        return f"search_in_file({filepath}, '{pattern}') → {n} match(es)"

    def _workspace_search(self, args: dict, result: str) -> str:
        """Summarise workspace_search call by counting result lines."""
        query = str(args.get("query", ""))[:50]
        no_hit = (
            not result
            or result.startswith("No matches")
            or result.startswith("No semantically related")
            or result.startswith("Semantic search is disabled")
            # CodeSearchAgent.search()'s genuine-miss path (empty/filtered-out
            # workspace) shares the same "[SYSTEM: CODE SEARCH: '...']" prefix
            # as a real hit, so it can't be told apart by prefix alone - match
            # on the fixed phrase that only appears in the miss message.
            or "no indexed code chunks available" in result
        )
        if no_hit:
            return f"workspace_search('{query}') → 0 line(s)"
        n_lines = result.count("\n") + 1
        return f"workspace_search('{query}') → {n_lines} line(s)"

    def _doc_search(self, args: dict, result: str) -> str:
        query = str(args.get("query", ""))[:60]
        no_results = "No embedded documents" in result or "No chunks" in result or "ERROR" in result[:30]
        if no_results:
            return f"doc_search('{query}') → no results"
        # Count passage blocks by the 📄 marker emitted by DocSearchAgent.search()
        n_chunks = result.count("📄 [")
        return f"doc_search('{query}') → {n_chunks} passage(s) found"

    def _bash(self, args: dict, result: str) -> str:
        # bash handler returns a plain string: "[exit 0]\nstdout..." - never JSON.
        cmd = str(args.get("command", "")).strip()
        short_cmd = cmd[:60] + ("…" if len(cmd) > 60 else "")
        icon = "✓" if result.startswith("[exit 0]") else "✗"
        # Show the LAST meaningful non-blank output line as a preview.
        # The last line is where pytest/make/compiler put their final summaries;
        # the first line is almost always a progress/startup message.
        lines = result.split("\n")
        non_blank = [l.strip() for l in lines[1:] if l.strip()]
        preview = non_blank[-1] if non_blank else ""
        detail = f" | {preview[:80]}" if preview else ""
        return f"bash({short_cmd}) → {icon}{detail}"

    def _show_image(self, args: dict, result: str) -> str:
        filepath = args.get("file", "?")
        try:
            success = bool(json.loads(result).get("success", False))
        except Exception:
            success = False
        icon = "✓" if success else "✗"
        return f"show_image({filepath}) → {icon}"

    def _ingest_pdf(self, args: dict, result: str) -> str:
        """Summarise ingest_pdf call by extracting pages and chars."""
        filename = str(args.get("_raw", args.get("filename", "?"))).strip()
        stripped = result.strip()
        skipped = stripped.startswith("⚡ [ingest_pdf]") and "already ingested and unchanged" in stripped
        success = stripped.startswith("✅ PDF ingested:") or skipped
        icon = "✓" if success else "✗"
        detail = ""
        if skipped:
            detail = " (unchanged, skipped)"
        elif success:
            pages = re.search(r'(\d+)\s*pages', result)
            chars = re.search(r'([\d,]+)\s*chars', result)
            if pages:
                detail += f" {pages.group(1)} pages"
            if chars:
                detail += f", {chars.group(1)} chars"
        return f"ingest_pdf({filename}) → {icon}{detail}"

    def _ingest_chat(self, args: dict, result: str) -> str:
        """Summarise ingest_chat call - success/skip both count as done."""
        filename = str(args.get("filename") or args.get("query", "?")).strip()
        stripped = result.strip()
        skipped = stripped.startswith("⚡ [ingest_chat]") and "already imported" in stripped
        success = stripped.startswith("✅ Imported") or skipped
        icon = "✓" if success else "✗"
        detail = " (unchanged, skipped)" if skipped else ""
        return f"ingest_chat({filename}) → {icon}{detail}"

    def _ingest_text(self, args: dict, result: str) -> str:
        """Summarise ingest_text call by extracting the embedded chunk count."""
        filename = str(args.get("filename") or args.get("query", "?")).strip()
        stripped = result.strip()
        skipped = stripped.startswith("⚡ [ingest_text]") and "already ingested" in stripped
        success = stripped.startswith("✅ Text ingested:") or skipped
        icon = "✓" if success else "✗"
        detail = ""
        if skipped:
            detail = " (unchanged, skipped)"
        elif success:
            n_chunks = re.search(r'(\d+)\s*embedded chunks', result)
            if n_chunks:
                detail = f" {n_chunks.group(1)} chunks"
        return f"ingest_text({filename}) → {icon}{detail}"

    def _generic(self, fn: str, args: dict, result: str) -> str:
        """Generic fallback checking explicit success/failure markers."""
        key_args = list(args.keys())[:3]
        result_lower = result.lower().strip()
        is_failure = (
            '"success": false' in result_lower
            or '"success":false' in result_lower
            or result_lower.startswith("[error]")
            or result_lower.startswith("error:")
        )
        is_success = (
            '"success": true' in result_lower
            or '"success":true' in result_lower
        )
        icon = "✗" if is_failure else ("✓" if is_success else "?")
        return f"{fn}({', '.join(key_args)}) → {icon}"

    def _search_semantic_scholar(self, args: dict, result: str) -> str:
        query = str(args.get("query", ""))[:50]
        year  = args.get("year", "")
        year_note = f" [{year}]" if year else ""
        if "No papers found" in result or "ERROR" in result[:40]:
            return f"search_semantic_scholar('{query}'{year_note}) → ✗ 0 results"
        n = len(re.findall(r'^\d+\.\s+', result, re.MULTILINE))
        icon = "✓" if n > 0 else "?"
        return f"search_semantic_scholar('{query}'{year_note}) → {icon} {n} paper(s)"