# turn_state_manager.py — Per-turn backup archives and restore pipeline.
#
# Called by start.py after every successful turn (commit_turn) and when the
# user types /restore N or /turns. Never exposed as a model tool.
#
# What is archived per turn:
#   • scratch/  →  backups/turn_N_scratch.zip  (ephemeral enhanced_*/backup dirs excluded)
#   • workspace_registry.json  →  backups/turn_N_registry.json
#   • ledger entry  →  backups/turn_ledger.json  (filenames, bash cmds, uploads, outputs)
#
# uploads/ and outputs/ are NOT archived — see design notes in plan.

import json
import os
import shutil
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import config
from utils import robust_rmtree, load_json, save_json_atomic, EXCL_PREFIXES, EXCL_SUFFIXES, EXCL_DIRS, _force_remove, console
from rich.markup import escape as _esc_markup

# ── Tool-to-attribution mapping ───────────────────────────────────────────────
# IMPORTANT: These must match the ACTUAL active tools in core_tool_definitions.py
# and _build_dispatch() in tool_handlers.py.
#
# Active file-modifying tools in this project:
#   bash       — primary way the model creates/modifies files (cat >, cp, python, etc.)
#   str_replace — surgical edits to existing files
#
_OP_MODIFY = {"str_replace"}
_OP_BASH   = {"bash"}

# Bash subcommand patterns that indicate file creation/modification in scratch
# Used to produce useful model_created hints in the ledger even though bash is opaque.
_BASH_CREATE_PATTERNS = (
    "> ", ">> ",       # shell redirect writes
    "cat >", "cat>>",  # heredoc/pipe writes
    "cp ", "mv ",      # copy/move into scratch
    "touch ",          # file creation
)


def _count_scratch_files(scratch_path: str) -> int:
    """
    Count files in scratch_path, pruning heavy/excluded directories so we
    never descend into node_modules, venv, .git, etc.  Replaces bare
    Path(x).rglob('*') calls that can freeze the REPL on large workspaces.
    """
    count = 0
    try:
        for root, dirs, files in os.walk(scratch_path):
            dirs[:] = [
                d for d in dirs
                if d not in EXCL_DIRS
                and not any(d.endswith(s) for s in EXCL_SUFFIXES)
            ]
            count += len(files)
    except Exception:
        pass
    return count


class TurnStateManager:
    """
    Manages per-turn backup archives and the turn ledger.

    Public API (called from start.py):
        commit_turn(turn_num, tool_call_log, uploaded_filenames, prompt_preview)
        restore_turn(turn_num, agent)  →  status string
        list_turns()                   →  formatted timeline string
        clear_all()                    →  wipes backups/ (called by /reset)
    """

    def __init__(self):
        os.makedirs(config.BACKUPS_DIR, exist_ok=True)
        self._ledger_path = os.path.join(config.BACKUPS_DIR, "turn_ledger.json")
        self._ledger      = self._load_ledger()

        # turn_num → Thread, for scratch zips currently being written in the
        # background by commit_turn(). Without this, restore_turn()/list_turns()/
        # delete_turn() have no way to know a zip is still in-flight and will
        # treat a just-committed turn's not-yet-written archive as "pruned"
        # (see _await_zip()).
        self._pending_zips: dict = {}

    def _await_zip(self, turn_num: int, timeout: float = 15.0) -> None:
        """
        Block until turn_num's background scratch-zip thread (if any) finishes,
        bounded by timeout. commit_turn() zips scratch/ asynchronously so the
        REPL isn't blocked after every turn — but that means restore_turn(),
        list_turns(), and delete_turn() must not check os.path.exists(zip_path)
        without first waiting for a same-turn zip that may still be writing.
        Call this before any archive-existence check for a specific turn_num.
        """
        t = self._pending_zips.get(turn_num)
        if t is not None:
            t.join(timeout=timeout)
            if not t.is_alive():
                self._pending_zips.pop(turn_num, None)

    def _await_all_zips(self, timeout: float = 30.0) -> None:
        """
        Block until ALL pending background scratch-zip threads finish.
        Called by restore_turn() before wiping scratch/ to prevent PermissionError
        on Windows when a background zip thread still holds file handles open.
        """
        for _turn_num, _t in list(self._pending_zips.items()):
            try:
                _t.join(timeout=timeout)
            except Exception:
                pass
            if not _t.is_alive():
                self._pending_zips.pop(_turn_num, None)

    # ══════════════════════════════════════════════════════════════════════════
    # LEDGER PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def _load_ledger(self) -> list:
        data = load_json(self._ledger_path, default=[])
        return data if isinstance(data, list) else []

    def _save_ledger(self):
        save_json_atomic(self._ledger_path, self._ledger)

    # ══════════════════════════════════════════════════════════════════════════
    # ZIP HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _zip_scratch(self, zip_path: str):
        """
        Zip config.SCRATCH_DIR into zip_path, skipping ephemeral files/dirs.
        Uses ZIP_DEFLATED level-6 — good compression, fast enough for text/code.
        """
        scratch = config.SCRATCH_DIR
        with zipfile.ZipFile(zip_path, "w",
                             compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for root, dirs, files in os.walk(scratch):
                # Prune excluded directories in-place so os.walk doesn't descend
                dirs[:] = [
                    d for d in dirs
                    if not any(d.startswith(p) for p in EXCL_PREFIXES)
                    and not any(d.endswith(s) for s in EXCL_SUFFIXES)
                    and d not in EXCL_DIRS
                ]
                for fname in files:
                    if any(fname.startswith(p) for p in EXCL_PREFIXES):
                        continue
                    fp      = os.path.join(root, fname)
                    arcname = os.path.relpath(fp, scratch).replace("\\", "/")
                    try:
                        zf.write(fp, arcname)
                    except Exception:
                        pass  # skip locked / unreadable files silently

    def _scratch_size_mb(self) -> float:
        """Return total size of config.SCRATCH_DIR in megabytes."""
        total = 0
        try:
            for root, _, files in os.walk(config.SCRATCH_DIR):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
        except Exception:
            pass
        return total / (1024 * 1024)

    # ══════════════════════════════════════════════════════════════════════════
    # ATTRIBUTION — free from _current_tool_call_log
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_attribution(self, tool_call_log: list) -> dict:
        """
        Extract per-turn file attribution from emperor._current_tool_call_log.

        Active file-modifying tools in this project:
          - bash       → primary creation/modification path (opaque but we flag it)
          - str_replace → surgical edits; args["file"] is the target path

        No filesystem watching needed — the log records every executed tool call.
        """
        modified  = set()
        bash_cmds = []
        outputs   = set()

        for entry in (tool_call_log or []):
            fn   = entry.get("fn", "")
            args = entry.get("args", {})

            if fn in _OP_MODIFY:
                # str_replace: args["file"] is the target path
                raw = args.get("file", "")
                if not raw:
                    console.print("[dim yellow]attribution: str_replace call missing 'file' key — skipping[/dim yellow]")
                    continue
                host = config.container_to_host_path(raw)
                name = os.path.basename(host)
                # Classify: outputs/ or scratch/
                if os.path.normcase(host).startswith(
                        os.path.normcase(config.OUTPUTS_DIR)):
                    outputs.add(name)
                else:
                    modified.add(name)

            elif fn in _OP_BASH:
                cmd = (args.get("command") or "")[:80].strip()
                if cmd:
                    bash_cmds.append(cmd)

        return {
            "model_modified":  sorted(modified),   # files edited via str_replace
            "model_bash":      bash_cmds[:5],       # bash commands (file creation lives here)
            "outputs_created": sorted(outputs),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # COMMIT — called at end of each successful turn
    # ══════════════════════════════════════════════════════════════════════════

    def commit_turn(
        self,
        turn_num:          int,
        tool_call_log:     list,
        uploaded_filenames: list,
        prompt_preview:    str,
        rerun_of:          int  = None,
        edited:            bool = False,
        images:            list = None,
    ):
        """
        Archive the current state and write a ledger entry for turn_num.

        Should be called AFTER emperor.save_history() so the history on disk
        matches what we're snapshotting.

        Args:
            rerun_of: If this turn is a rerun, the original turn number it replaces.
            edited:   True if this turn was generated from an edited user message.
            images:   The already-processed images list for this turn (the
                      output of load_images_for_vision — [{filename, src, mime}]),
                      if any were attached. Persisted as a small sidecar JSON so
                      a later /rerun can resend the exact same images without
                      depending on the original file/path still existing —
                      local-file images are already base64-encoded into `src`
                      at this point, so this makes them just as durable as a
                      raw base64/data-URI image was from the start.
        """
        zip_name = f"turn_{turn_num}_scratch.zip"
        reg_name = f"turn_{turn_num}_registry.json"
        zip_path = os.path.join(config.BACKUPS_DIR, zip_name)
        reg_path = os.path.join(config.BACKUPS_DIR, reg_name)

        # ── Zip scratch/ (background thread, non-blocking) ───────────────────
        # The zip runs in a daemon thread so the REPL is free for the user's
        # next message immediately. Each turn zips to its own uniquely-named
        # file so concurrent zips across turns can't collide; restore_turn(),
        # list_turns(), and delete_turn() call _await_zip() before touching a
        # specific turn's archive, so a still-writing zip is waited for
        # instead of being misreported as pruned/missing.
        tmp_zip_path = zip_path + ".tmp"
        scratch_mb   = self._scratch_size_mb()
        max_mb       = getattr(config, "BACKUP_MAX_SCRATCH_MB", 150)

        if scratch_mb > max_mb:
            console.print(
                f"[yellow]⚠ T{turn_num} scratch backup skipped — "
                f"workspace is {scratch_mb:.0f} MB (limit {max_mb} MB).[/yellow]"
            )
            zip_name = None
        else:
            def _do_zip():
                try:
                    self._zip_scratch(tmp_zip_path)
                    os.replace(tmp_zip_path, zip_path)
                except Exception as e:
                    console.print(f"[yellow]⚠ T{turn_num} scratch backup failed (bg): {e}[/yellow]")
                    try:
                        if os.path.exists(tmp_zip_path):
                            os.remove(tmp_zip_path)
                    except Exception:
                        pass

            _zip_thread = threading.Thread(target=_do_zip, daemon=True, name=f"zip-turn-{turn_num}")
            self._pending_zips[turn_num] = _zip_thread
            _zip_thread.start()

        # ── Snapshot workspace registry ───────────────────────────────────────
        tmp_reg_path = reg_path + ".tmp"
        try:
            if os.path.exists(config.WORKSPACE_REGISTRY_FILE):
                shutil.copy2(config.WORKSPACE_REGISTRY_FILE, tmp_reg_path)
                os.replace(tmp_reg_path, reg_path)
            else:
                reg_name = None
        except Exception as e:
            console.print(f"[yellow]⚠ T{turn_num} registry backup failed: {e}[/yellow]")
            if os.path.exists(tmp_reg_path):
                os.remove(tmp_reg_path)
            reg_name = None

        # ── Snapshot images (if this turn had any) ────────────────────────────
        img_name = None
        if images:
            img_name = f"turn_{turn_num}_images.json"
            img_path = os.path.join(config.BACKUPS_DIR, img_name)
            tmp_img_path = img_path + ".tmp"
            try:
                # Write-then-replace here to match the atomic-write pattern
                # used for the zip/registry snapshots above.
                with open(tmp_img_path, "w", encoding="utf-8") as f:
                    json.dump(images, f)
                os.replace(tmp_img_path, img_path)
            except Exception as e:
                console.print(f"[yellow]⚠ T{turn_num} image snapshot failed: {e}[/yellow]")
                img_name = None
                if os.path.exists(tmp_img_path):
                    try:
                        os.remove(tmp_img_path)
                    except Exception:
                        pass

        # ── Build ledger entry ────────────────────────────────────────────────
        attr  = self._parse_attribution(tool_call_log)
        entry = {
            "turn":           turn_num,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "prompt_preview": (prompt_preview or "")[:120],
            "user_uploads":   list(uploaded_filenames or []),
            "scratch_zip":    zip_name,
            "registry_snap":  reg_name,
            "images_snap":    img_name,
            "rerun_of":       rerun_of,   # None for normal turns
            "edited":         edited,     # False for normal turns
            **attr,
        }

        # Replace any existing entry for this turn (handles post-restore re-commit)
        self._ledger = [e for e in self._ledger if e.get("turn") != turn_num]
        self._ledger.append(entry)
        self._ledger.sort(key=lambda e: e.get("turn", 0))
        self._save_ledger()

        # ── Prune archives beyond rolling window ──────────────────────────────
        # Only prune if the current turn's backup succeeded — otherwise older
        # restorable archives would be deleted with nothing to replace them.
        if zip_name is not None:
            self._prune_old_archives(turn_num)


    # ══════════════════════════════════════════════════════════════════════════
    # PRUNE — rolling archive window
    # ══════════════════════════════════════════════════════════════════════════

    def _prune_old_archives(self, current_turn: int):
        """
        Delete zip/registry files for turns older than config.MAX_BACKUP_TURNS.
        Ledger entries are kept forever (they're tiny text).
        """
        cutoff = current_turn - config.MAX_BACKUP_TURNS
        for entry in self._ledger:
            if entry.get("turn", 0) <= cutoff:
                for key in ("scratch_zip", "registry_snap", "images_snap"):
                    fname = entry.get(key)
                    if fname:
                        fpath = os.path.join(config.BACKUPS_DIR, fname)
                        try:
                            if os.path.exists(fpath):
                                os.remove(fpath)
                        except Exception:
                            pass

    def _delete_orphan_archives_after(self, turn_num: int):
        """
        Delete archive files (.zip, registry snaps, image snaps) for every
        ledger entry whose turn number is strictly greater than turn_num.

        MUST be called BEFORE trimming the ledger — once entries are removed
        their filenames are lost and the files become permanently orphaned on disk.

        Used by restore_turn() (full restore path). NOT called by /rerun or /edit
        when keep_tail=True — those use delete_turn_archives() instead.
        """
        for entry in self._ledger:
            if entry.get("turn", 0) > turn_num:
                self._await_zip(entry.get("turn", 0))  # don't delete a zip mid-write
                for key in ("scratch_zip", "registry_snap", "images_snap"):
                    fname = entry.get(key)
                    if fname:
                        fpath = os.path.join(config.BACKUPS_DIR, fname)
                        try:
                            if os.path.exists(fpath):
                                os.remove(fpath)
                        except Exception:
                            pass

    def delete_turn_archives(self, turn_num: int):
        """
        Delete archive files (scratch zip, registry snap, image snap) for
        exactly one turn and remove its ledger entry.

        Used by /rerun and /edit user when keep_tail=True is passed to
        truncate_history_only() — the blanket _delete_orphan_archives_after()
        is skipped in that path, so this handles cleanup of only the single
        turn being replaced, leaving T(N+1)..T(current) archives intact so
        /restore still works for those turns.
        """
        self._await_zip(turn_num)
        entry = next((e for e in self._ledger if e.get("turn") == turn_num), None)
        if entry:
            for key in ("scratch_zip", "registry_snap", "images_snap"):
                fname = entry.get(key)
                if fname:
                    fpath = os.path.join(config.BACKUPS_DIR, fname)
                    try:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                    except Exception:
                        pass
            # Remove only this turn's ledger entry — tail entries are preserved.
            self._ledger = [e for e in self._ledger if e.get("turn") != turn_num]
            self._save_ledger()

    # ══════════════════════════════════════════════════════════════════════════
    # DELETE — remove a single turn pair and renumber everything after it
    # ══════════════════════════════════════════════════════════════════════════

    def delete_turn(self, turn_num: int, agent) -> str:
        """
        Remove a single user+assistant pair from conversation history and keep
        all data stores consistent.

        What changes:
          • chat_history — pair at index (N-1)*2 and (N-1)*2+1 is spliced out.
          • Archives for deleted turn — scratch.zip and registry.json are deleted.
          • Archives for turns N+1..max — renamed to N..max-1 (ascending order
            to avoid collisions: turn_4→turn_3 before turn_5→turn_4, etc.).
          • Ledger — entry for N is removed; entries for N+1..max get turn-=1
            and updated filenames.

        What does NOT change:
          • live scratch/ workspace — completely untouched.
          • uploads/, outputs/ — untouched.
          • Global .jsonl history — append-only; never mutated.

        Returns a Rich-formatted status string.
        """
        current_turns = len(agent.chat_history) // 2

        if turn_num < 1 or turn_num > current_turns:
            return f"[red]T{turn_num} out of range (1–{current_turns}).[/red]"

        # ── Step 1: Splice out the two messages from chat_history ─────────────
        user_idx = (turn_num - 1) * 2
        del agent.chat_history[user_idx : user_idx + 2]
        agent.save_history(skip_global=True)

        # ── Step 2: Delete orphaned archives for the deleted turn ─────────────
        self._await_zip(turn_num)  # don't delete a zip that's still being written
        deleted_entry = next(
            (e for e in self._ledger if e.get("turn") == turn_num), None
        )
        if deleted_entry:
            for key in ("scratch_zip", "registry_snap", "images_snap"):
                fname = deleted_entry.get(key)
                if fname:
                    fpath = os.path.join(config.BACKUPS_DIR, fname)
                    try:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                    except Exception as e:
                        console.print(
                            f"[yellow]⚠ could not delete archive {fname}: {e}[/yellow]"
                        )

        # ── Step 3: Rename subsequent archives (ascending order is critical) ──
        # Renaming turn_4→turn_3 before turn_5→turn_4 prevents overwriting.
        # Missing files are silently skipped — not all turns have archives
        # (e.g. pure-chat turns or turns whose archives were already pruned).
        for t in range(turn_num + 1, current_turns + 1):
            self._await_zip(t)  # don't rename a zip that's still being written
            for suffix in ("_scratch.zip", "_registry.json", "_images.json"):
                old_name = f"turn_{t}{suffix}"
                new_name = f"turn_{t - 1}{suffix}"
                old_path = os.path.join(config.BACKUPS_DIR, old_name)
                new_path = os.path.join(config.BACKUPS_DIR, new_name)
                if os.path.exists(old_path):
                    try:
                        os.replace(old_path, new_path)
                    except Exception as e:
                        console.print(
                            f"[yellow]⚠ could not rename {old_name} → {new_name}: {e}[/yellow]"
                        )

        # ── Step 4: Rebuild ledger ────────────────────────────────────────────
        # Drop the deleted entry; decrement turn number and patch filenames for
        # every entry that came after the deleted turn.
        new_ledger = []
        for entry in self._ledger:
            t = entry.get("turn", 0)
            if t == turn_num:
                continue  # dropped

            if t > turn_num:
                new_e = dict(entry)
                new_e["turn"] = t - 1

                # Patch scratch_zip filename if it follows the standard pattern
                old_zip = entry.get("scratch_zip")
                if old_zip == f"turn_{t}_scratch.zip":
                    new_e["scratch_zip"] = f"turn_{t - 1}_scratch.zip"

                # Patch registry_snap filename similarly
                old_reg = entry.get("registry_snap")
                if old_reg == f"turn_{t}_registry.json":
                    new_e["registry_snap"] = f"turn_{t - 1}_registry.json"

                # Patch images_snap filename similarly
                old_img = entry.get("images_snap")
                if old_img == f"turn_{t}_images.json":
                    new_e["images_snap"] = f"turn_{t - 1}_images.json"

                new_ledger.append(new_e)
            else:
                new_ledger.append(entry)

        self._ledger = sorted(new_ledger, key=lambda e: e.get("turn", 0))
        self._save_ledger()

        # ── Step 5: Reconcile workspace tracker ───────────────────────────────
        # The live workspace is untouched, but this keeps the in-memory index
        # accurate in case anything changed underneath between this and the
        # previous reconcile.
        try:
            agent.workspace_tracker.reconcile_workspace()
        except Exception:
            pass

        new_total = len(agent.chat_history) // 2
        if new_total == 0:
            suffix = " History is now empty."
        elif turn_num < current_turns:
            suffix = f" Turns T{turn_num}–T{current_turns - 1} renumbered."
        else:
            suffix = ""

        return (
            f"[green]✓ T{turn_num} deleted. "
            f"{new_total} turn(s) remain.{suffix}[/green]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RESTORE — full pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def restore_turn(self, turn_num: int, agent) -> str:
        """
        Revert workspace and agent memory to end-of-turn N.

        Steps:
          1. Validate archives exist in ledger
          2. Clear scratch/, extract turn_N_scratch.zip
          3. Restore workspace_registry.json + reload tracker
          4. Truncate agent.chat_history to N*2 messages
          5. Save history to disk
          6. Trim ledger to turns <= N
          7. Reconcile workspace tracker

        Returns a Rich-formatted status string for console.print().
        """
        # ── Find ledger entry ─────────────────────────────────────────────────
        entry = next((e for e in self._ledger if e.get("turn") == turn_num), None)
        if not entry:
            available = sorted(e["turn"] for e in self._ledger)
            return (
                f"[red]T{turn_num} not found in ledger.[/red]\n"
                f"[dim]Available turns: {available}[/dim]"
            )

        zip_name = entry.get("scratch_zip")
        reg_name = entry.get("registry_snap")

        if not zip_name:
            return (
                f"[red]T{turn_num} has no scratch archive "
                f"(backup may have failed at that turn).[/red]"
            )

        # If this turn's zip is still being written in the background
        # (commit_turn() is async), wait for it instead of misreporting the
        # archive as pruned just because it isn't on disk yet.
        self._await_zip(turn_num)

        zip_path = os.path.join(config.BACKUPS_DIR, zip_name)
        reg_path = os.path.join(config.BACKUPS_DIR, reg_name) if reg_name else None

        if not os.path.exists(zip_path):
            # Find the nearest turn whose archive is still on disk so the user
            # has an actionable /restore target instead of a dead end.
            _available = sorted(
                e["turn"] for e in self._ledger
                if e.get("scratch_zip")
                and os.path.exists(os.path.join(config.BACKUPS_DIR, e["scratch_zip"]))
            )
            if _available:
                _nearest = min(_available, key=lambda t: abs(t - turn_num))
                _hint = f"\n[dim]  Nearest restorable turn: T{_nearest}  →  /restore {_nearest}[/dim]"
            else:
                _hint = "\n[dim]  No restorable turns currently available.[/dim]"
            return (
                f"[red]T{turn_num} archive not found "
                f"(pruned — only last {config.MAX_BACKUP_TURNS} turns retained).[/red]{_hint}"
            )

        # ── 1. Restore scratch/ ───────────────────────────────────────────────
        scratch = config.SCRATCH_DIR
        try:
            # Bug 28: await ALL pending zip threads, not just the target turn's,
            # so no background thread holds file handles open during the wipe.
            self._await_all_zips()

            # Wipe current scratch contents (keep the directory itself)
            _locked = []
            for item in os.listdir(scratch):
                if item in EXCL_DIRS:
                    continue  # preserve heavy dependency folders across restores

                item_path = os.path.join(scratch, item)
                try:
                    if os.path.isdir(item_path):
                        robust_rmtree(item_path)
                    else:
                        _force_remove(item_path)  # Bug 35: handles read-only files on Windows
                except Exception:
                    _locked.append(item_path)  # track instead of silently skipping

            # One retry after a brief pause — Windows file locks often release quickly
            if _locked:
                import time as _t; _t.sleep(0.3)
                _still_locked = []
                for _lp in _locked:
                    try:
                        if os.path.isdir(_lp):
                            robust_rmtree(_lp)
                        else:
                            _force_remove(_lp)  # Bug 35: handles read-only files on Windows
                    except Exception:
                        _still_locked.append(os.path.basename(_lp))
                if _still_locked:
                    console.print(
                        f"[yellow]⚠  {len(_still_locked)} locked file(s) could not be removed "
                        f"before restore: {', '.join(_still_locked[:5])}. "
                        f"Workspace may contain stale files.[/yellow]"
                    )

            # Extract archived scratch
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(scratch)

        except Exception as e:
            return f"[red]Scratch restore failed: {e}[/red]"

        # ── 2. Restore workspace registry ─────────────────────────────────────
        if reg_path and os.path.exists(reg_path):
            try:
                shutil.copy2(reg_path, config.WORKSPACE_REGISTRY_FILE)
                agent.workspace_tracker._load_registry()
                # Bug 11: also purge the on-disk bm25_cache.pkl, not just the
                # in-memory index — otherwise the stale pickle (newer mtime
                # than the just-restored, older registry) gets reloaded as
                # if valid, returning phantom search results from turns that
                # no longer exist post-restore.
                agent.workspace_tracker.invalidate_bm25_cache()
            except Exception as e:
                console.print(f"[yellow]⚠ registry restore partial: {e}[/yellow]")
        else:
            # No registry snap → wipe existing registry so tracker starts clean
            try:
                if os.path.exists(config.WORKSPACE_REGISTRY_FILE):
                    os.remove(config.WORKSPACE_REGISTRY_FILE)
                agent.workspace_tracker._load_registry()
                agent.workspace_tracker.invalidate_bm25_cache()
            except Exception:
                pass

        # ── 3 & 4. Truncate agent memory + trim ledger ─────────────────────────
        # Factored into its own method so /rerun can call ONLY this part —
        # see truncate_history_only() below.
        self.truncate_history_only(turn_num, agent)

        # ── 5. Reconcile workspace tracker ────────────────────────────────────
        try:
            agent.workspace_tracker.reconcile_workspace()
        except Exception:
            pass

        # ── Summary ───────────────────────────────────────────────────────────
        n_files = _count_scratch_files(scratch)
        return (
            f"[green]✓ Restored to end of T{turn_num}. "
            f"{n_files} file(s) in scratch, memory at {turn_num} turn(s).[/green]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY-ONLY TRUNCATION  (used by /rerun — does NOT touch scratch/)
    # ══════════════════════════════════════════════════════════════════════════

    def truncate_history_only(self, turn_num: int, agent, keep_tail: bool = False) -> str:
        """
        Truncate agent.chat_history to end-of-turn N, WITHOUT touching
        scratch/ or the workspace registry.

        keep_tail=False (default) — also deletes all archives and ledger
            entries for turns > turn_num. Used by restore_turn() and the
            old /rerun behaviour.

        keep_tail=True — skips archive deletion and ledger trimming for
            turns beyond turn_num. Used by /rerun and /edit user when the
            tail turns (T(N+1)..T(current)) are being preserved and
            re-appended after regeneration. The caller is responsible for
            cleaning up only the single replaced turn via
            delete_turn_archives().
        """
        agent.chat_history            = agent.chat_history[:turn_num * 2]
        agent.last_tool_summary       = ""    # stale — would confuse next turn's context
        # Bug 14 follow-up: also clear the cancelled-turn summary here. Without
        # this, a turn cancelled just before a /restore or /rerun would have
        # its "[PREVIOUS TURN WAS CANCELLED...]" note survive the truncation
        # and get injected into the regenerated/restored turn's prompt, even
        # though that cancelled attempt has nothing to do with the turn being
        # truncated back to.
        agent.last_cancelled_summary  = ""
        agent._partial_messages       = None  # clear any mid-turn interrupt state
        agent._current_tool_call_log  = []
        agent.save_history(skip_global=True)

        if not keep_tail:
            # Delete orphaned archive files (scratch zip, registry snap, image
            # snap) for turns > turn_num BEFORE trimming the ledger, so we
            # still have the filenames to look up — they become permanently
            # unreachable once their ledger entry is gone.
            self._delete_orphan_archives_after(turn_num)
            self._ledger = [e for e in self._ledger if e.get("turn", 0) <= turn_num]
            self._save_ledger()

        n_files = _count_scratch_files(config.SCRATCH_DIR)
        return (
            f"[green]✓ History truncated to end of T{turn_num}. "
            f"Scratch/workspace files were NOT reverted "
            f"({n_files} file(s) currently in scratch).[/green]"
        )

    def load_images_snapshot(self, turn_num: int) -> list | None:
        """
        Load the persisted images list for turn_num (see commit_turn's
        `images` param), for use by /rerun.

        Returns None if the turn had no images, its snapshot was pruned by
        the rolling backup window, or the file is missing/corrupt — /rerun
        treats None (when the turn DID have images per the ledger) as a hard
        stop rather than silently proceeding text-only.
        """
        entry = next((e for e in self._ledger if e.get("turn") == turn_num), None)
        if not entry:
            return None
        img_name = entry.get("images_snap")
        if not img_name:
            return None
        img_path = os.path.join(config.BACKUPS_DIR, img_name)
        if not os.path.exists(img_path):
            return None
        try:
            with open(img_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, list) and loaded else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # /turns — formatted timeline
    # ══════════════════════════════════════════════════════════════════════════

    def list_turns(self) -> str:
        """Return a Rich-formatted turn timeline for /turns command."""
        if not self._ledger:
            return "[dim]No turns recorded yet.[/dim]"

        lines = ["─" * 72]
        for entry in self._ledger:
            t        = entry.get("turn", "?")
            ts       = entry.get("timestamp", "")[:16].replace("T", "  ")
            # Bug 14: escape user-controlled strings so Rich doesn't interpret
            # brackets in prompt text as markup tags, which would cause MarkupError.
            preview  = _esc_markup(entry.get("prompt_preview", "")[:52])
            uploads  = [_esc_markup(u) for u in entry.get("user_uploads",   [])]
            modified = [_esc_markup(m) for m in entry.get("model_modified", [])]
            outputs  = [_esc_markup(o) for o in entry.get("outputs_created", [])]
            bash     = entry.get("model_bash", [])

            # Revision markers — shown when turn was a rerun or from an edit
            rerun_of = entry.get("rerun_of")
            edited   = entry.get("edited", False)
            markers  = ""
            if rerun_of is not None:
                markers += f"  [bold yellow][↺ rerun of T{rerun_of}][/bold yellow]"
            if edited:
                markers += f"  [bold cyan][✏ edited][/bold cyan]"

            # Archive availability tag
            zip_name    = entry.get("scratch_zip")
            if zip_name:
                self._await_zip(t, timeout=2.0)  # short — this is just a display check
            has_archive = bool(
                zip_name and
                os.path.exists(os.path.join(config.BACKUPS_DIR, zip_name))
            )
            archive_tag = "" if has_archive else "  [dim][archive pruned][/dim]"

            lines.append(
                f" [bold cyan]T{t}[/bold cyan]  [dim]{ts}[/dim]  \"{preview}\"{markers}{archive_tag}"
            )

            # Attribution row
            parts = []
            if uploads:
                parts.append(f"[magenta]↑[/magenta] {', '.join(uploads)}")
            if modified:
                parts.append(f"[yellow]~[/yellow] {', '.join(modified)}")
            if outputs:
                parts.append(f"[blue]→[/blue] outputs/{', '.join(outputs)}")
            if bash:
                # Always show bash summary — it's the primary file creation path
                parts.append(f"[dim]$ {bash[0][:50]}{'…' if len(bash[0]) > 50 else ''}[/dim]")
                if len(bash) > 1:
                    parts.append(f"[dim]  (+{len(bash)-1} more bash cmd(s))[/dim]")

            if parts:
                lines.append(f"           {'  ·  '.join(parts)}")
            lines.append("")

        lines.append("─" * 72)
        lines.append("  /restore [bold]N[/bold]      revert scratch + memory to end of turn N")
        lines.append("  /restore [bold]last[/bold]   undo the last completed turn")
        lines.append("─" * 72)
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # CLEAR ALL — called by /reset
    # ══════════════════════════════════════════════════════════════════════════

    def clear_all(self):
        """
        Wipe all backup archives and the ledger.
        Called by /reset to keep backups/ in sync with the cleared session.
        """
        try:
            for item in os.listdir(config.BACKUPS_DIR):
                if item == "turn_ledger.json" or (item.startswith("turn_") and (item.endswith(".zip") or item.endswith(".json"))):
                    item_path = os.path.join(config.BACKUPS_DIR, item)
                    if os.path.isfile(item_path):
                        try:
                            os.remove(item_path)
                        except Exception:
                            pass
            self._ledger = []
            self._save_ledger()
        except Exception as e:
            console.print(f"[yellow]⚠ backup clear failed: {e}[/yellow]")