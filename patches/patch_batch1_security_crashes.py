"""
Batch 1 — Security Crashes & Tool Failures
Fixes bugs: #1 (path traversal), #2 (POSIX editor shell=True), 
            #3 (show_image /uploads), #4 (show_image SameFileError),
            #5 (MarkupError in /turns for bash cmds),
            #6 (uncaught re.error in external search)

Run from the project root:
    python patches/patch_batch1_security_crashes.py
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
    print("Batch 1 - Security, Crashes & Tool Failures")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Bug #1a: path traversal in _handle_ingest_chat
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #1a: path traversal guard in _handle_ingest_chat",
        find=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_chat requires a filename, e.g. <ingest_chat>gemini_chat.txt</ingest_chat>\"\n"
            "\n"
            "        ext = os.path.splitext(filename)[1].lower()\n"
            "        if ext == '.pdf':\n"
            "            return f\"[SYSTEM: ERROR] '{filename}' is a PDF \u2014 use ingest_pdf for PDFs, not ingest_chat.\""
        ),
        replace=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_chat requires a filename, e.g. <ingest_chat>gemini_chat.txt</ingest_chat>\"\n"
            "\n"
            "        # Bug #1 fix: strip any directory component to prevent os.path.join path traversal.\n"
            "        filename = os.path.basename(filename)\n"
            "\n"
            "        ext = os.path.splitext(filename)[1].lower()\n"
            "        if ext == '.pdf':\n"
            "            return f\"[SYSTEM: ERROR] '{filename}' is a PDF \u2014 use ingest_pdf for PDFs, not ingest_chat.\""
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #1b: path traversal in _handle_ingest_text
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #1b: path traversal guard in _handle_ingest_text",
        find=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_text requires a filename, e.g. <ingest_text>notes.txt</ingest_text>\"\n"
            "\n"
            "        ext = os.path.splitext(filename)[1].lower()\n"
            "        if ext == '.pdf':\n"
            "            return f\"[SYSTEM: ERROR] '{filename}' is a PDF \u2014 use ingest_pdf for PDFs, not ingest_text.\""
        ),
        replace=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_text requires a filename, e.g. <ingest_text>notes.txt</ingest_text>\"\n"
            "\n"
            "        # Bug #1 fix: strip any directory component to prevent os.path.join path traversal.\n"
            "        filename = os.path.basename(filename)\n"
            "\n"
            "        ext = os.path.splitext(filename)[1].lower()\n"
            "        if ext == '.pdf':\n"
            "            return f\"[SYSTEM: ERROR] '{filename}' is a PDF \u2014 use ingest_pdf for PDFs, not ingest_text.\""
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #1c: path traversal in _handle_ingest_pdf (+ secure fuzzy walk)
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #1c: path traversal guard + secure fuzzy walk in _handle_ingest_pdf",
        find=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_pdf requires a filename, e.g. <ingest_pdf>paper.pdf</ingest_pdf>\"\n"
            "\n"
            "        if not filename.lower().endswith('.pdf'):\n"
            "            return (\n"
            "                f\"[ingest_pdf] '{filename}' is not a PDF. \"\n"
            "                \"Non-PDF files (.py, .txt, .md, .csv, images) can be read directly \"\n"
            "                \"from /uploads with view_lines or bash \u2014 no ingestion needed.\"\n"
            "            )\n"
            "\n"
            "        # Resolve path inside /uploads\n"
            "        pdf_path = os.path.join(config.UPLOADS_FOLDER, filename)\n"
            "        if not os.path.exists(pdf_path):\n"
            "            # Fuzzy search \u2014 match by basename anywhere under /uploads\n"
            "            for root, _, files in os.walk(config.UPLOADS_FOLDER):\n"
            "                if filename in files:\n"
            "                    pdf_path = os.path.join(root, filename)\n"
            "                    break\n"
            "            else:\n"
            "                return f\"[SYSTEM: ERROR] '{filename}' not found in /uploads.\""
        ),
        replace=(
            "        filename = (args.get('filename') or args.get('query') or '').strip()\n"
            "        if not filename:\n"
            "            return \"[SYSTEM: ERROR] ingest_pdf requires a filename, e.g. <ingest_pdf>paper.pdf</ingest_pdf>\"\n"
            "\n"
            "        # Bug #1 fix: strip any directory component to prevent os.path.join path traversal.\n"
            "        filename = os.path.basename(filename)\n"
            "\n"
            "        if not filename.lower().endswith('.pdf'):\n"
            "            return (\n"
            "                f\"[ingest_pdf] '{filename}' is not a PDF. \"\n"
            "                \"Non-PDF files (.py, .txt, .md, .csv, images) can be read directly \"\n"
            "                \"from /uploads with view_lines or bash \u2014 no ingestion needed.\"\n"
            "            )\n"
            "\n"
            "        # Resolve path inside /uploads\n"
            "        _uploads_abs = os.path.realpath(config.UPLOADS_FOLDER)\n"
            "        pdf_path = os.path.join(config.UPLOADS_FOLDER, filename)\n"
            "        if not os.path.exists(pdf_path):\n"
            "            # Fuzzy search \u2014 match by basename anywhere under /uploads.\n"
            "            # Bug #1 fix: verify each candidate stays under UPLOADS_FOLDER\n"
            "            # via realpath to prevent symlink-based escapes.\n"
            "            for root, _, files in os.walk(config.UPLOADS_FOLDER):\n"
            "                if filename in files:\n"
            "                    candidate = os.path.join(root, filename)\n"
            "                    if os.path.realpath(candidate).startswith(_uploads_abs):\n"
            "                        pdf_path = candidate\n"
            "                        break\n"
            "            else:\n"
            "                return f\"[SYSTEM: ERROR] '{filename}' not found in /uploads.\""
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #2: POSIX editor shell=True
    # -------------------------------------------------------------------------
    patch_file(
        "start.py",
        "Bug #2: drop shell=True from external editor subprocess call",
        find="        result = subprocess.call([*editor_cmd, tmp_path], shell=True)",
        replace="        result = subprocess.call([*editor_cmd, tmp_path])",
    )

    # -------------------------------------------------------------------------
    # Bug #3: show_image crashes on /uploads paths
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #3: show_image - handle /uploads paths without modifying _resolve_path",
        find=(
            "        try:\n"
            "            full_path = self.file_ops._resolve_path(filepath)\n"
            "        except PermissionError as e:\n"
            "            return json.dumps({\"success\": False, \"error\": str(e)})\n"
            "        except Exception as e:\n"
            "            return json.dumps({\"success\": False, \"error\": f\"Could not resolve path: {e}\"})"
        ),
        replace=(
            "        # Bug #3 fix: handle /uploads paths before _resolve_path, which only\n"
            "        # allows scratch/outputs. show_image is read-only so /uploads is safe.\n"
            "        _host_path   = config.container_to_host_path(filepath)\n"
            "        _uploads_abs = os.path.realpath(config.UPLOADS_FOLDER)\n"
            "        _host_real   = os.path.realpath(_host_path) if os.path.exists(_host_path) else _host_path\n"
            "        _in_uploads  = (_host_real == _uploads_abs\n"
            "                        or _host_real.startswith(_uploads_abs + os.sep))\n"
            "\n"
            "        if _in_uploads:\n"
            "            full_path = _host_path\n"
            "        else:\n"
            "            try:\n"
            "                full_path = self.file_ops._resolve_path(filepath)\n"
            "            except PermissionError as e:\n"
            "                return json.dumps({\"success\": False, \"error\": str(e)})\n"
            "            except Exception as e:\n"
            "                return json.dumps({\"success\": False, \"error\": f\"Could not resolve path: {e}\"})"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #4: show_image SameFileError
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #4: show_image - skip copy when src and dst are the same file",
        find=(
            "            os.makedirs(config.OUTPUTS_DIR, exist_ok=True)\n"
            "            out_path = os.path.join(config.OUTPUTS_DIR, os.path.basename(full_path))\n"
            "            shutil.copy2(full_path, out_path)"
        ),
        replace=(
            "            os.makedirs(config.OUTPUTS_DIR, exist_ok=True)\n"
            "            out_path = os.path.join(config.OUTPUTS_DIR, os.path.basename(full_path))\n"
            "            # Bug #4 fix: skip copy when src and dst are the same file\n"
            "            # (e.g. the image is already in /outputs) to avoid SameFileError.\n"
            "            if not (os.path.exists(out_path) and os.path.samefile(full_path, out_path)):\n"
            "                shutil.copy2(full_path, out_path)"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #5: MarkupError in /turns for bash commands with brackets
    # -------------------------------------------------------------------------
    patch_file(
        "turn_state_manager.py",
        "Bug #5: escape bash cmds in list_turns to prevent Rich MarkupError",
        find='            bash     = entry.get("model_bash", [])',
        replace='            bash     = [_esc_markup(b) for b in entry.get("model_bash", [])]',
    )

    # -------------------------------------------------------------------------
    # Bug #6: uncaught re.error in external search path
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #6: wrap re.search in try/except re.error in external search path",
        find=(
            "            for i, line in enumerate(lines):\n"
            "                hit = (re.search(pattern, line) if use_regex\n"
            "                       else pattern.lower() in line.lower())"
        ),
        replace=(
            "            for i, line in enumerate(lines):\n"
            "                try:\n"
            "                    hit = (re.search(pattern, line) if use_regex\n"
            "                           else pattern.lower() in line.lower())\n"
            "                except re.error:\n"
            "                    # Bug #6 fix: invalid regex pattern - fall back to literal match\n"
            "                    hit = pattern in line"
        ),
    )

    print()
    print("Batch 1 complete.")


if __name__ == "__main__":
    main()
