"""
Batch 2 — State & Data Integrity Fixes
Fixes bugs: #12 (scratch size ignores EXCL dirs, inflating reported size),
            #13 (grep floods context - EXCL_DIRS prune in file_ops search_workspace),
            #18 (rerun deletes archive before _run_turn succeeds),
            #19 (/edit erases image-attachment marker without re-adding it),
            #22 (ingest_chat collides on same stem for different source files),
            #23 (/reset ghost archive race - clear_all doesn't wait for in-flight zips),
            #25 (/delete corrupts rerun_of pointers after renumbering)

Run from the project root:
    python patches/patch_batch2_state_integrity.py
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
    print("Batch 2 - State & Data Integrity Fixes")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Bug #12: _scratch_size_mb ignores EXCL dirs (node_modules etc.),
    # inflating the reported scratch size and permanently disabling backups.
    # Note: EXCL_DIRS, EXCL_PREFIXES, EXCL_SUFFIXES are already imported at
    # the top of turn_state_manager.py, so no extra import needed here.
    # -------------------------------------------------------------------------
    patch_file(
        "turn_state_manager.py",
        "Bug #12: _scratch_size_mb - prune excluded dirs same as _zip_scratch",
        find=(
            "    def _scratch_size_mb(self) -> float:\r\n"
            "        \"\"\"Return total size of config.SCRATCH_DIR in megabytes.\"\"\"\r\n"
            "        total = 0\r\n"
            "        try:\r\n"
            "            for root, _, files in os.walk(config.SCRATCH_DIR):\r\n"
            "                for fn in files:\r\n"
            "                    try:\r\n"
            "                        total += os.path.getsize(os.path.join(root, fn))\r\n"
            "                    except OSError:\r\n"
            "                        pass\r\n"
            "        except Exception:\r\n"
            "            pass\r\n"
            "        return total / (1024 * 1024)"
        ),
        replace=(
            "    def _scratch_size_mb(self) -> float:\r\n"
            "        \"\"\"Return total size of config.SCRATCH_DIR in megabytes.\r\n"
            "\r\n"
            "        Bug #12 fix: prune the same excluded dirs/prefixes/suffixes as\r\n"
            "        _zip_scratch does, so node_modules / venv / .git don't inflate\r\n"
            "        the reported size and permanently disable backups.\r\n"
            "        \"\"\"\r\n"
            "        total = 0\r\n"
            "        try:\r\n"
            "            for root, dirs, files in os.walk(config.SCRATCH_DIR):\r\n"
            "                # Mirror _zip_scratch exclusion logic\r\n"
            "                dirs[:] = [\r\n"
            "                    d for d in dirs\r\n"
            "                    if d not in EXCL_DIRS\r\n"
            "                    and not any(d.startswith(p) for p in EXCL_PREFIXES)\r\n"
            "                    and not any(d.endswith(s) for s in EXCL_SUFFIXES)\r\n"
            "                ]\r\n"
            "                for fn in files:\r\n"
            "                    if any(fn.startswith(p) for p in EXCL_PREFIXES):\r\n"
            "                        continue\r\n"
            "                    try:\r\n"
            "                        total += os.path.getsize(os.path.join(root, fn))\r\n"
            "                    except OSError:\r\n"
            "                        pass\r\n"
            "        except Exception:\r\n"
            "            pass\r\n"
            "        return total / (1024 * 1024)"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #13: search_workspace grep walk descends into node_modules etc.
    # The actual code uses `for root, _, files in os.walk(s_dir)` — fix it
    # by capturing dirs and pruning excluded ones in-place.
    # -------------------------------------------------------------------------
    patch_file(
        "file_ops_agent.py",
        "Bug #13: prune EXCL_DIRS/PREFIXES/SUFFIXES in search_workspace grep walk",
        find=(
            "            # os.walk handles nested folders automatically\n"
            "            for root, _, files in os.walk(s_dir):\n"
            "                for filename in files:"
        ),
        replace=(
            "            # os.walk handles nested folders automatically\n"
            "            # Bug #13 fix: capture dirs and prune heavy/excluded subtrees\n"
            "            # (node_modules, .git, venv etc.) so grep can't flood context.\n"
            "            from utils import EXCL_DIRS, EXCL_PREFIXES, EXCL_SUFFIXES\n"
            "            for root, dirs, files in os.walk(s_dir):\n"
            "                dirs[:] = [\n"
            "                    d for d in dirs\n"
            "                    if d not in EXCL_DIRS\n"
            "                    and not any(d.startswith(p) for p in EXCL_PREFIXES)\n"
            "                    and not any(d.endswith(s) for s in EXCL_SUFFIXES)\n"
            "                ]\n"
            "                for filename in files:"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #18: rerun deletes T{target}'s archive BEFORE _run_turn succeeds.
    # Fix: move the delete to the success branch (after _run_turn returns a
    # non-None result), so a cancelled/failed rerun leaves the archive intact.
    # -------------------------------------------------------------------------
    patch_file(
        "start.py",
        "Bug #18: defer archive deletion to after _run_turn succeeds in cmd_rerun",
        find=(
            "    # Delete only T{target}'s old archive \u2014 not the tail's.\r\n"
            "    tsm.delete_turn_archives(target)\r\n"
            "\r\n"
            "    rerun_filenames = [img[\"filename\"] for img in rerun_images] if rerun_images else None"
        ),
        replace=(
            "    # Bug #18 fix: DON'T delete T{target}'s archive here - it must\r\n"
            "    # survive until _run_turn completes successfully. Deletion is deferred\r\n"
            "    # to the success branch so a cancelled/failed rerun leaves the\r\n"
            "    # original turn's archive intact and /restore still works.\r\n"
            "\r\n"
            "    rerun_filenames = [img[\"filename\"] for img in rerun_images] if rerun_images else None"
        ),
    )

    patch_file(
        "start.py",
        "Bug #18: add delete_turn_archives call in the success branch of cmd_rerun",
        find=(
            "    # Success: re-append tail after the newly generated T{target}.\r\n"
            "    _reappend_tail(tail_history, tail_ledger, target, current)"
        ),
        replace=(
            "    # Success: the new T{target} is committed - now safe to delete the old archive.\r\n"
            "    # Bug #18 fix: deletion happens here (post-success), not before _run_turn.\r\n"
            "    tsm.delete_turn_archives(target)\r\n"
            "\r\n"
            "    # Re-append tail after the newly generated T{target}.\r\n"
            "    _reappend_tail(tail_history, tail_ledger, target, current)"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #19: /edit strips the [SYSTEM: Images attached ...] marker from the
    # user message before saving, so the ledger and /turns display lose the
    # image reference permanently. Fix: re-add the marker after stripping it,
    # using the snapshot filenames from the ledger entry.
    # -------------------------------------------------------------------------
    patch_file(
        "start.py",
        "Bug #19: re-add image-attachment marker after /edit strips it",
        find=(
            "            # new_content is the edited text of old_content, which still\r\n"
            "            # has the marker baked in (it was the pre-filled editor text) \u2014\r\n"
            "            # strip it so _run_turn regenerates it correctly from the real images.\r\n"
            "            new_content = re.sub(r'\\n\\[SYSTEM: Images attached.*?\\]\\s*$', '', new_content)"
        ),
        replace=(
            "            # new_content is the edited text of old_content, which still\r\n"
            "            # has the marker baked in (it was the pre-filled editor text) \u2014\r\n"
            "            # strip it so _run_turn regenerates it correctly from the real images.\r\n"
            "            new_content = re.sub(r'\\n\\[SYSTEM: Images attached.*?\\]\\s*$', '', new_content)\r\n"
            "            # Bug #19 fix: re-add the marker from the snapshot filenames so\r\n"
            "            # the stored message still references the images that were attached.\r\n"
            "            # Use U+2014 em-dash to match the regex in cmd_rerun's image marker.\r\n"
            "            if edit_images:\r\n"
            "                _snap_names = ', '.join(img['filename'] for img in edit_images)\r\n"
            "                new_content = f\"{new_content}\\n[SYSTEM: Images attached \\u2014 {_snap_names}]\""
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #22: ingest_chat builds out_name from the stem alone, so two
    # different files with the same basename (e.g. two "chat.txt" exports)
    # silently overwrite each other. Fix: include first 8 chars of MD5.
    # -------------------------------------------------------------------------
    patch_file(
        "tool_handlers.py",
        "Bug #22: include content hash in ingest_chat output filename",
        find=(
            "        stem     = os.path.splitext(os.path.basename(filename))[0]\n"
            "        out_name = f\"{stem}_imported.jsonl\"\n"
            "        out_path = os.path.join(config.GLOBAL_HISTORIES_DIR, out_name)\n"
            "        sidecar_path = out_path + \".import.json\""
        ),
        replace=(
            "        stem     = os.path.splitext(os.path.basename(filename))[0]\n"
            "        # Bug #22 fix: include the first 8 hex chars of the content hash\n"
            "        # so two different files with the same basename don't collide.\n"
            "        out_name = f\"{stem}_{file_hash[:8]}_imported.jsonl\"\n"
            "        out_path = os.path.join(config.GLOBAL_HISTORIES_DIR, out_name)\n"
            "        sidecar_path = out_path + \".import.json\""
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #23: clear_all() doesn't wait for in-flight zip threads before
    # deleting, so a just-committed turn's zip can land AFTER the clear and
    # ghost an archive that /reset was supposed to have removed.
    # Fix: iterate _pending_zips and join each thread before deleting.
    # -------------------------------------------------------------------------
    patch_file(
        "turn_state_manager.py",
        "Bug #23: wait for all in-flight zip threads before clear_all deletes",
        find=(
            "    def clear_all(self):\r\n"
            "        \"\"\"\r\n"
            "        Wipe all backup archives and the ledger.\r\n"
            "        Called by /reset to keep backups/ in sync with the cleared session.\r\n"
            "        \"\"\"\r\n"
            "        try:\r\n"
            "            for item in os.listdir(config.BACKUPS_DIR):"
        ),
        replace=(
            "    def clear_all(self):\r\n"
            "        \"\"\"\r\n"
            "        Wipe all backup archives and the ledger.\r\n"
            "        Called by /reset to keep backups/ in sync with the cleared session.\r\n"
            "        \"\"\"\r\n"
            "        # Bug #23 fix: wait for every in-flight background zip thread to\r\n"
            "        # finish before deleting, so a just-committed turn's zip can't\r\n"
            "        # materialise after the clear and leave ghost archives.\r\n"
            "        for t in list(self._pending_zips.keys()):\r\n"
            "            self._await_zip(t, timeout=20.0)\r\n"
            "        try:\r\n"
            "            for item in os.listdir(config.BACKUPS_DIR):"
        ),
    )

    # -------------------------------------------------------------------------
    # Bug #25: delete_turn's ledger rebuild doesn't adjust rerun_of pointers,
    # so after deleting T3 the entry that said "rerun_of=3" still reads 3 even
    # though the old T4 is now T3. Fix: decrement rerun_of when it's > turn_num,
    # and drop it when it equals turn_num (the original turn was deleted).
    # -------------------------------------------------------------------------
    patch_file(
        "turn_state_manager.py",
        "Bug #25: adjust rerun_of pointers in delete_turn ledger rebuild",
        find=(
            "            if t > turn_num:\r\n"
            "                new_e = dict(entry)\r\n"
            "                new_e[\"turn\"] = t - 1\r\n"
            "\r\n"
            "                # Patch scratch_zip filename if it follows the standard pattern"
        ),
        replace=(
            "            if t > turn_num:\r\n"
            "                new_e = dict(entry)\r\n"
            "                new_e[\"turn\"] = t - 1\r\n"
            "\r\n"
            "                # Bug #25 fix: adjust rerun_of alongside the turn number so\r\n"
            "                # the /turns display stays consistent after a delete.\r\n"
            "                if \"rerun_of\" in new_e:\r\n"
            "                    ro = new_e[\"rerun_of\"]\r\n"
            "                    if ro == turn_num:\r\n"
            "                        del new_e[\"rerun_of\"]   # original turn was deleted\r\n"
            "                    elif ro > turn_num:\r\n"
            "                        new_e[\"rerun_of\"] = ro - 1\r\n"
            "\r\n"
            "                # Patch scratch_zip filename if it follows the standard pattern"
        ),
    )

    print()
    print("Batch 2 complete.")


if __name__ == "__main__":
    main()
