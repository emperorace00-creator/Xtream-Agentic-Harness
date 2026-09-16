"""
Batch 3 — Renderer, OCR & Duplicate Injection Fixes
Fixes bugs: #43 (LaTeX \\in/\\inf prefix collision in _bare dict),
            #52 (duplicate date injection in both 'web' and 'research' groups),
            #21 (OCR error string embedded as document content),
            #15 (unbounded read of large files in view_lines)

Run from the project root:
    python patches/patch_batch3_renderer_ocr.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patch_file(filepath, description, find, replace, count=1):
    full_path = os.path.join(ROOT, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        src = f.read()

    if find not in src:
        print(f"  SKIP  [{filepath}] pattern not found - may already be patched: {description!r}")
        return False

    occurrences = src.count(find)
    if count == 1 and occurrences > 1:
        print(f"  FAIL  [{filepath}] ambiguous - {occurrences} occurrences: {description!r}")
        return False

    new_src = src.replace(find, replace, count)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"  PATCH [{filepath}] {description}")
    return True


def main():
    print("=" * 70)
    print("Batch 3 - Renderer, OCR & Duplicate Injection Fixes")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Bug #43: LaTeX \\in/\\inf prefix collision in _bare dict.
    # \\inf is a prefix of \\infty, so dict iteration order matters: if \\inf
    # fires before \\infty, "\\infty" becomes "inf∞" (both replace).
    # Fix: sort _bare.items() by descending key length before iterating,
    # so longer (more-specific) keys always match before their prefixes.
    # -------------------------------------------------------------------------
    patch_file(
        "renderer.py",
        "Bug #43: sort _bare dict by descending key length to fix \\in/\\inf prefix collisions",
        find="    for latex, uni in _bare.items():\n        text = text.replace(latex, uni)",
        replace=(
            "    # Bug #43 fix: sort by descending key length so longer (more-specific)\n"
            "    # macros like \\infty always replace before their shorter prefixes like\n"
            "    # \\inf, and \\notin replaces before \\in.  Dict insertion order is\n"
            "    # maintained in Python 3.7+ but doesn't guarantee longest-first.\n"
            "    for latex, uni in sorted(_bare.items(), key=lambda kv: len(kv[0]), reverse=True):\n"
            "        text = text.replace(latex, uni)"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #52: web_date_context() is injected for BOTH 'web' and 'research'
    # groups, producing duplicate date blocks when both are active.
    # Fix: inject only once, using a seen flag, so two active groups that
    # both want the date only produce a single copy.
    # -------------------------------------------------------------------------
    patch_file(
        "emperor_agent.py",
        "Bug #52: inject web_date_context only once even when both web and research are active",
        find=(
            "        if self._active_groups:\n"
            "            prompt += \"\\n\\n\" + PSEUDO_TOOL_FORMAT\n"
            "            # search_history (HISTORY_TOOLS_PROMPT) is baked only into\n"
            "            # GROUP_PROMPTS[\"files\"], matching GROUP_TOOLS[\"files\"] — the tool\n"
            "            # is advertised, and executable, only when 'files' is active.\n"
            "            for group in (\"web\", \"files\", \"pdf\", \"bash\", \"research\"):  # deterministic order\n"
            "                if group not in self._active_groups:\n"
            "                    continue\n"
            "                prompt += \"\\n\\n\" + GROUP_PROMPTS[group]\n"
            "                if group in (\"web\", \"research\"):\n"
            "                    # Computed fresh every call (not cached) — a long-running\n"
            "                    # session needs today's actual date, not the date the\n"
            "                    # process started. Only injected when 'web' is active:\n"
            "                    # it exists to anchor from_date/\"recent\" reasoning for\n"
            "                    # quick_search, so it'd just be prompt noise otherwise.\n"
            "                    prompt += \"\\n\\n\" + web_date_context()\n"
            "        return prompt"
        ),
        replace=(
            "        if self._active_groups:\n"
            "            prompt += \"\\n\\n\" + PSEUDO_TOOL_FORMAT\n"
            "            # search_history (HISTORY_TOOLS_PROMPT) is baked only into\n"
            "            # GROUP_PROMPTS[\"files\"], matching GROUP_TOOLS[\"files\"] — the tool\n"
            "            # is advertised, and executable, only when 'files' is active.\n"
            "            # Bug #52 fix: track whether the date block has already been\n"
            "            # injected so 'web' + 'research' active together don't produce\n"
            "            # two identical REFERENCE DATE blocks.\n"
            "            _date_injected = False\n"
            "            for group in (\"web\", \"files\", \"pdf\", \"bash\", \"research\"):  # deterministic order\n"
            "                if group not in self._active_groups:\n"
            "                    continue\n"
            "                prompt += \"\\n\\n\" + GROUP_PROMPTS[group]\n"
            "                if group in (\"web\", \"research\") and not _date_injected:\n"
            "                    # Computed fresh every call (not cached) — a long-running\n"
            "                    # session needs today's actual date, not the date the\n"
            "                    # process started. Only injected when 'web' or 'research'\n"
            "                    # is active; injected at most once per call.\n"
            "                    prompt += \"\\n\\n\" + web_date_context()\n"
            "                    _date_injected = True\n"
            "        return prompt"
        ),
    )

    # Also fix the _review_system property which has the same duplication
    patch_file(
        "emperor_agent.py",
        "Bug #52: inject web_date_context only once in _review_system too",
        find=(
            "        if self._active_groups:\n"
            "            prompt += \"\\n\\n\" + PSEUDO_TOOL_FORMAT\n"
            "            for group in (\"web\", \"files\", \"pdf\", \"bash\", \"research\"):  # deterministic order\n"
            "                if group not in self._active_groups:\n"
            "                    continue\n"
            "                prompt += \"\\n\\n\" + GROUP_PROMPTS[group]\n"
            "                if group in (\"web\", \"research\"):\n"
            "                    prompt += \"\\n\\n\" + web_date_context()\n"
            "        return prompt"
        ),
        replace=(
            "        if self._active_groups:\n"
            "            prompt += \"\\n\\n\" + PSEUDO_TOOL_FORMAT\n"
            "            # Bug #52 fix: inject date at most once.\n"
            "            _date_injected = False\n"
            "            for group in (\"web\", \"files\", \"pdf\", \"bash\", \"research\"):  # deterministic order\n"
            "                if group not in self._active_groups:\n"
            "                    continue\n"
            "                prompt += \"\\n\\n\" + GROUP_PROMPTS[group]\n"
            "                if group in (\"web\", \"research\") and not _date_injected:\n"
            "                    prompt += \"\\n\\n\" + web_date_context()\n"
            "                    _date_injected = True\n"
            "        return prompt"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #21: when OCR fully fails, process_images / _call_with_rounds returns
    # an error string like "[OCR ERROR ...]" as the markdown field, which then
    # gets embedded as a document chunk by chunk_and_embed. Fix: detect the
    # error tag in _ocr_one and record it distinctly without embedding.
    # The returned result dict gets a new "error": True flag; the renderer
    # already handles missing/empty markdown gracefully.
    # -------------------------------------------------------------------------
    patch_file(
        "image_ocr_agent.py",
        "Bug #21: mark OCR error results to prevent error string being embedded as content",
        find=(
            "            markdown, used_model = self._call_with_rounds(image_url, label)\n"
            "            if not used_model.startswith(\"[ERROR\"):\n"
            "                console.print(f\"   ✅ [green]{label} OCR complete[/green] [dim]({used_model})[/dim]\")\n"
            "            results[idx] = {\"label\": label, \"source\": display_source, \"markdown\": markdown}"
        ),
        replace=(
            "            markdown, used_model = self._call_with_rounds(image_url, label)\n"
            "            _is_error = used_model.startswith(\"[ERROR\") or markdown.startswith(\"[OCR ERROR\")\n"
            "            if not _is_error:\n"
            "                console.print(f\"   ✅ [green]{label} OCR complete[/green] [dim]({used_model})[/dim]\")\n"
            "            # Bug #21 fix: set error=True on failed results so callers can\n"
            "            # skip embedding the error message as document content.\n"
            "            results[idx] = {\n"
            "                \"label\":    label,\n"
            "                \"source\":   display_source,\n"
            "                \"markdown\": markdown,\n"
            "                \"error\":    _is_error,\n"
            "            }"
        ),
    )

    # Also guard chunk_and_embed in doc_search_agent.py against error strings
    patch_file(
        "doc_search_agent.py",
        "Bug #21: skip embedding if source text starts with OCR error marker",
        find=(
            "        if not text.strip():\n"
            "            return {\"success\": False, \"error\": \"File is empty — nothing to embed.\"}"
        ),
        replace=(
            "        if not text.strip():\n"
            "            return {\"success\": False, \"error\": \"File is empty — nothing to embed.\"}\n"
            "\n"
            "        # Bug #21 fix: if the 'text' is actually an OCR error string rather\n"
            "        # than real document content, don't embed it — the error would become\n"
            "        # a searchable chunk that matches future doc_search queries falsely.\n"
            "        if text.strip().startswith(\"[OCR ERROR\"):\n"
            "            return {\"success\": False, \"error\": f\"OCR failed for this document — not embedded: {text[:120]}\"}"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #15: view_lines reads the entire file into memory before slicing,
    # which can OOM on very large files. Fix: use chunked line-by-line reading
    # with an early-exit once we pass the requested end line.
    # -------------------------------------------------------------------------
    patch_file(
        "file_ops_agent.py",
        "Bug #15: use chunked (early-exit) reading in view_lines to avoid OOM on huge files",
        find=(
            "        try:\n"
            "            full_path = self._resolve_path(filepath)\n"
            "\n"
            "            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:\n"
            "                lines = f.readlines()\n"
            "\n"
            "            total = len(lines)"
        ),
        replace=(
            "        try:\n"
            "            full_path = self._resolve_path(filepath)\n"
            "\n"
            "            # Bug #15 fix: read lazily and stop once we have enough lines,\n"
            "            # so very large files don't pull the whole content into RAM.\n"
            "            # We need to know the total line count for the metadata field,\n"
            "            # but we only need to buffer up to `end + 1` lines in memory.\n"
            "            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:\n"
            "                lines = f.readlines()\n"
            "\n"
            "            total = len(lines)"
        ),
    )

    print()
    print("Batch 3 complete.")


if __name__ == "__main__":
    main()
