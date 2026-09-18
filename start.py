#!/usr/bin/env python3
# start.py - single entry point: Docker sandbox setup + agent loop.
#
# Run this file to launch Emperor for this project.
#
# Docker is optional - if the engine is not running the agent starts
# normally but bash-tool mode (/tool) is blocked with a clear warning.

import os
import re
import json
import glob
import shlex
import shutil
import atexit
import subprocess
import tempfile
import threading
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style as PTStyle

# prompt_toolkit's CheckboxList/RadioList hardcode ScrollbarMargin(display_arrows=True)
# with no way to pass custom symbols through checkboxlist_dialog(). Its library
# defaults are the raw ASCII characters "^" and "v" - a caret sits small and high
# in the cell while "v" spans near full letter-height, so the two arrows never
# look the same size even though they share one style class. Patch the class
# default so every dialog in the app (current and future) gets a matched,
# equal-weight triangle pair instead.
from prompt_toolkit.layout.margins import ScrollbarMargin as _ScrollbarMargin
_original_scrollbar_init = _ScrollbarMargin.__init__
def _patched_scrollbar_init(self, display_arrows=False, up_arrow_symbol="▲", down_arrow_symbol="▼"):
    _original_scrollbar_init(self, display_arrows, up_arrow_symbol, down_arrow_symbol)
_ScrollbarMargin.__init__ = _patched_scrollbar_init

class SlashCommandCompleter(Completer):
    def __init__(self):
        self.commands = [
            ("/tool", "Open tool group selector (web, files, pdf, bash, research)"),
            ("/google", "Switch to Google Gemini backend"),
            ("/nim", "Switch to NVIDIA NIM backend"),
            ("/cf", "Switch to Cloudflare Workers AI backend"),
            ("/local", "Switch to local llama.cpp server"),
            ("/tools", "Display current tool mode and active backend"),
            ("/history", "Print the current conversation history"),
            ("/turns", "Display the turn timeline and revision markers"),
            ("/edit", "Edit a past message (e.g. /edit 2 user or /edit 2 agent)"),
            ("/restore", "Revert workspace and memory to a turn"),
            ("/rerun", "Regenerate a turn from history"),
            ("/delete", "Delete a turn pair from history (e.g. /delete 3)"),
            ("/reset", "Clear the current session, scratch files, and backups"),
            ("/review", "Self-review last response (optional: /review <focus>"),
        ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd, desc in self.commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd, 
                        start_position=-len(text), 
                        display=cmd, 
                        display_meta=desc
                    )

slash_completer = SlashCommandCompleter()

import config
from image_ocr_agent import ImageOCRAgent
from utils import TokenCounter, robust_rmtree, EXCL_DIRS, save_json_atomic, _force_remove
from emperor_agent import EmperorAgent
from web_agent import WebAgent
from turn_state_manager import TurnStateManager
from renderer import (
    console, rule, _flatten_content, show_image_in_terminal,
    load_images_for_vision, process_images_via_ocr, print_smart_response,
    RULE_STYLE, META, WARN, ERR,
)
from rich.markup import escape as _esc

# Project / container identity
# Sourced from config.py to avoid symlink-related divergence.
_PROJECT_DIR  = config.WORKSPACE_ROOT
_PROJECT_NAME = Path(_PROJECT_DIR).name
_CONTAINER    = config.CONTAINER_NAME
_IMAGE_NAME   = "emperor-base:latest"

# /reset cleanup patterns (item 20 / SP-2)
# Previously /reset only removed top-level *.json files from database/, one by one
# with os.remove(). That missed:
#   - bm25_corpus.pkl        (workspace_tracker's BM25 index cache)
#   - *.tmp orphans          (left behind if a crash interrupts save_json_atomic's
#                             write-then-os.replace sequence)
# Both now get swept up too. Still top-level / non-recursive, same as before -
# subfolders like backups/ and chat_histories/ are handled separately (tsm.clear_all()
# / preserved intentionally) and are not touched here.
_RESET_PATTERNS = ["*.json", "*.pkl", "*.tmp"]


# ----
# DOCKER SANDBOX SETUP  (non-fatal)
# ----

def _setup_sandbox() -> bool:
    """
    Ensure the Docker sandbox container is running.

    Returns True if the container is up and ready, False if Docker is
    unavailable or setup failed. Never raises - the agent always starts.
    """
    scratch_dir = config.SCRATCH_DIR
    uploads_dir = config.UPLOADS_FOLDER
    outputs_dir = config.OUTPUTS_DIR

    for d in (scratch_dir, uploads_dir, outputs_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[dim]🏛️ Emperor — {_PROJECT_NAME}[/dim]")
    console.print(f"[dim]   container : {_CONTAINER}[/dim]")
    console.print(f"[dim]   scratch   : {scratch_dir}[/dim]")

    try:
        ping = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=6
        )
        if ping.returncode != 0:
            raise RuntimeError("Docker engine is not responding.")

        exists = subprocess.run(
            ["docker", "inspect", "--type=container", _CONTAINER],
            capture_output=True, text=True
        ).returncode == 0

        if not exists:
            console.print(f"[dim]   creating container '{_CONTAINER}'...[/dim]")
            cmd = [
                "docker", "run",
                "--name", _CONTAINER,
                "--detach",
                "--volume", f"{scratch_dir}:/workspace/scratch",
                "--volume", f"{uploads_dir}:/uploads:ro",
                "--volume", f"{outputs_dir}:/outputs",
                _IMAGE_NAME,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
            console.print(f"[dim]   container created.[/dim]\n")
        else:
            running = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", _CONTAINER],
                capture_output=True, text=True
            ).stdout.strip() == "true"

            if not running:
                console.print(f"[dim]   starting container '{_CONTAINER}'...[/dim]")
                r = subprocess.run(["docker", "start", _CONTAINER],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"docker start failed: {r.stderr.strip()}")
                console.print(f"[dim]   container started.[/dim]\n")
            else:
                console.print(f"[dim]   container already running.[/dim]\n")

        return True

    except Exception as e:
        console.print(f"[{WARN}]   Docker offline — bash tool disabled. Start Docker Engine and restart to enable.[/{WARN}]\n")
        return False


# Run sandbox setup once at startup.
DOCKER_AVAILABLE = _setup_sandbox()
def edit_in_external_editor(old_content: str) -> str:
    """
    Open old_content in the user's $EDITOR (default: nano) for in-place editing.

    Writes old_content to a temp file, blocks while the editor runs, reads the
    saved result back, and returns it. Used by /edit so users get a real cursor,
    full paste support, and no blank-line-terminated input() footguns.

    Falls back to the plain input() loop (returns None) if no editor could be
    launched - e.g. $EDITOR/nano missing, or running in a non-interactive shell.
    """
    editor_raw = os.environ.get("EDITOR", "nano")
    try:
        editor_cmd = shlex.split(editor_raw)
    except ValueError:
        editor_cmd = editor_raw.split()
    if not editor_cmd:
        editor_cmd = ["nano"]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(old_content)
            tmp_path = tf.name

        result = subprocess.call([*editor_cmd, tmp_path])
        if result != 0:
            console.print(f"[{WARN}]editor '{_esc(editor_raw)}' exited with code {result} — edit may be incomplete.[/{WARN}]")

        with open(tmp_path, "r", encoding="utf-8") as f:
            new_content = f.read()

        return new_content.rstrip("\n")

    except FileNotFoundError:
        console.print(f"[{WARN}]editor '{_esc(editor_raw)}' not found. Set $EDITOR or install nano.[/{WARN}]")
        return None
    except Exception as e:
        console.print(f"[{WARN}]external editor failed: {_esc(str(e))}[/{WARN}]")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _save_partial_history(agent, prompt_payload):
    try:
        partial_summary = agent.summarizer.summarize_turn(agent._current_tool_call_log)

        # Bug 14: this used to set agent.last_tool_summary, which
        # generate_with_history() injects into the NEXT turn's system prompt
        # under "[LAST TURN ACTIONS]" - implying those actions were
        # committed. But a cancelled turn's workspace changes are rolled
        # back, so that label would describe rolled-back actions as if they
        # actually happened. Store it under a distinctly-labelled field
        # instead so the model still gets the useful "this was attempted and
        # undone" context, without being told it happened for real.
        # last_tool_summary itself must only ever reflect committed turns.
        if partial_summary:
            agent.last_cancelled_summary = partial_summary

        console.print(f"[{META}]turn cancelled. not saved to history.[/{META}]\n")
    except Exception as e:
        console.print(f"[{WARN}]partial history save error: {_esc(str(e))}[/{WARN}]")


# ---- Initialization ----

image_ocr     = ImageOCRAgent()
web_agent     = WebAgent()
emperor      = EmperorAgent(ocr_agent=image_ocr, docker_available=DOCKER_AVAILABLE)
token_counter = TokenCounter()
tsm           = TurnStateManager()

atexit.register(lambda: emperor.save_history(skip_global=True))

# Path for the rerun Write-Ahead Log (WAL) - tail turns persisted here
# before history is truncated, so a process kill mid-rerun can be recovered.
_RERUN_TAIL_FILE = os.path.join(config.DATABASE_DIR, "rerun_pending_tail.json")


def _save_rerun_tail(target: int, current: int,
                    tail_history: list, tail_ledger: list) -> None:
    """Persist the rerun tail to the WAL file so it survives a crash."""
    try:
        os.makedirs(config.DATABASE_DIR, exist_ok=True)
        save_json_atomic(_RERUN_TAIL_FILE, {
            "target":       target,
            "current":      current,
            "tail_history": tail_history,
            "tail_ledger":  tail_ledger,
        })
    except Exception as e:
        console.print(
            f"[{WARN}]could not write rerun WAL — crash recovery disabled: "
            f"{_esc(str(e))}[/{WARN}]"
        )


# ----
# GENERATION HELPER  - shared by normal turns, /rerun, and /edit user
# ----

def _run_turn(prompt_payload: str, images=None, uploaded_filenames=None,
              prompt_preview: str = None, rerun_of: int = None, edited: bool = False):
    """
    Run one full generation turn:
      backup → generate (with interrupt/resume) → save history → commit archive.

    Args:
        prompt_payload:  The full prompt sent to the model (may include OCR content).
        images:          Vision images list, or None.
        uploaded_filenames: Filenames for history stub labelling.
        prompt_preview:  Short preview stored in the ledger (defaults to
                         prompt_payload[:120]). Pass `user_text` from the
                         normal flow so OCR-expanded payloads don’t pollute
                         the /turns timeline.
        rerun_of:        Turn number this is a rerun of, for ledger markers.
        edited:          True if this turn was generated from an edited user message.

    Returns the raw response string, or None if cancelled by the user.
    """
    uploaded_filenames = uploaded_filenames or []

    # Turn Setup & Rollback Backup
    # Backup is now created lazily, on the first actual tool call, not
    # eagerly here before every tool-mode turn (item 4 / SP-1). We just
    # reset the per-turn flags; emperor._start_scratch_backup() (triggered
    # from _call_tool) does the real work on a background thread.
    is_resuming     = False
    guidance        = ""
    thinking_paste  = ""
    cancelled       = False
    backup_dir      = config.SCRATCH_DIR + "_backup"
    backup_registry = config.WORKSPACE_REGISTRY_FILE + "_backup"

    if emperor._active_groups:
        emperor._backup_created = False
        emperor._backup_success = False
        emperor._backup_thread  = None

    # Generate
    raw_response = None
    while True:
        try:
            if not is_resuming:
                raw_response = emperor.generate_with_history(
                    prompt=prompt_payload,
                    web_agent=web_agent,
                    token_counter=token_counter,
                    images=images or None,
                )
            else:
                raw_response = emperor.resume_from_interrupt(
                    guidance=guidance,
                    thinking_paste=thinking_paste,
                    web_agent=web_agent,
                    token_counter=token_counter,
                )

            console.print()
            rule(style=RULE_STYLE)
            print_smart_response(raw_response)
            rule(style=RULE_STYLE)
            console.print()
            break

        except KeyboardInterrupt:
            console.print(f"\n[bold yellow]⚡ interrupted[/bold yellow]")
            try:
                thinking_paste = prompt(HTML('<b> ❯ paste thinking trace (or Enter to skip): </b>')).strip()
            except (KeyboardInterrupt, EOFError):
                thinking_paste = ""
            try:
                guidance = prompt(HTML('<b> ❯ guidance (or enter to cancel): </b>')).strip()
            except (KeyboardInterrupt, EOFError):
                guidance = ""

            if not guidance:
                emperor._partial_messages = None
                cancelled = True

                # Rollback workspace changes
                # A backup only exists if a tool was actually called this turn
                # (lazy backup - item 4). If it's still copying in the
                # background, give it a moment to finish before deciding.
                if emperor._active_groups and emperor._backup_created and emperor._backup_thread is not None:
                    emperor._backup_thread.join(timeout=10)
                if emperor._active_groups and emperor._backup_created and emperor._backup_success and os.path.exists(backup_dir):
                    console.print(f"[{META}]rolling back workspace changes...[/{META}]")
                    # NOTE: the backup itself excludes EXCL_DIRS (node_modules, venv,
                    # .git, etc. - see emperor_agent._start_scratch_backup), so those
                    # dirs must be preserved here rather than wiped, exactly like
                    # TurnStateManager.restore_turn() already does. Wiping everything
                    # and copying back only from backup_dir would otherwise silently
                    # delete them for good.
                    for _item in os.listdir(config.SCRATCH_DIR):
                        if _item in EXCL_DIRS:
                            continue
                        _item_path = os.path.join(config.SCRATCH_DIR, _item)
                        try:
                            if os.path.isdir(_item_path):
                                robust_rmtree(_item_path)
                            else:
                                os.remove(_item_path)
                        except Exception:
                            pass
                    try:
                        if os.path.exists(backup_registry):
                            shutil.copy(backup_registry, config.WORKSPACE_REGISTRY_FILE)
                        elif os.path.exists(config.WORKSPACE_REGISTRY_FILE):
                            os.remove(config.WORKSPACE_REGISTRY_FILE)

                        emperor.workspace_tracker._load_registry()
                        # Same primitive /restore uses (Bug 11): drop the in-memory
                        # BM25 index AND delete bm25_corpus.pkl. Clearing only
                        # memory leaves the pickle on disk; _ensure_index_loaded()
                        # below then reloads it whenever cache_mtime >=
                        # registry_mtime. The no-backup branch deletes the
                        # registry (mtime=0), so the pickle always wins and
                        # phantom mid-turn paths survive the rollback.
                        emperor.workspace_tracker.invalidate_bm25_cache()

                        # scratch/ is back to pre-turn state, so any embeddings
                        # built mid-turn now describe code that is gone or
                        # reverted. mark_stale() (not invalidate_index()) is
                        # enough: the next semantic query runs the incremental
                        # hash-diff rebuild and re-embeds only files that
                        # actually changed. Wiping the whole index would
                        # re-embed the entire workspace after every Ctrl+C,
                        # including turns that never searched. /restore still
                        # uses invalidate_index() because a zip-unpack is a
                        # full snapshot replace.
                        if getattr(emperor, "code_search_agent", None):
                            emperor.code_search_agent.mark_stale()

                        # Bug #37 fix: uploads cache may reference a PDF that was
                        # ingested mid-turn and then reverted - clear it so the
                        # model doesn't see a stale "ALREADY INGESTED" status.
                        emperor._uploads_cache = None

                        shutil.copytree(backup_dir, config.SCRATCH_DIR, dirs_exist_ok=True)
                        emperor.workspace_tracker._ensure_index_loaded()
                        console.print(f"[{META}]✓ workspace restored.[/{META}]")
                    except Exception as e:
                        console.print(f"[{WARN}]rollback partially failed due to locked files: {_esc(str(e))}[/{WARN}]")
                    finally:
                        try:
                            emperor.workspace_tracker.reconcile_workspace()
                        except Exception as _reconcile_err:
                            console.print(f"[{WARN}]post-rollback reconcile failed: {_reconcile_err}[/{WARN}]")
                        robust_rmtree(backup_dir)
                        if os.path.exists(backup_registry):
                            try:
                                os.remove(backup_registry)
                            except Exception:
                                pass
                # Persist partial history so the interrupted turn is not silently lost.
                _save_partial_history(emperor, prompt_payload)
                break

            console.print(f"[{META}]resuming with guidance...[/{META}]\n")
            is_resuming = True

    # Clean up backups if turn finished successfully
    if emperor._backup_thread is not None:
        emperor._backup_thread.join(timeout=10)
    if not cancelled and os.path.exists(backup_dir):
        robust_rmtree(backup_dir)
        if os.path.exists(backup_registry):
            os.remove(backup_registry)

    if cancelled:
        return None

    # Save history
    turn_num = len(emperor.chat_history) // 2 + 1
    try:
        # Use prompt_preview (raw user_text) as the history base so that
        # OCR-expanded payloads (vision=False path) never bloat the permanent
        # chat history. prompt_preview is always user_text in the normal flow
        # (passed explicitly from the REPL); fall back to prompt_payload only
        # for rerun/edit paths that don't set it.
        _history_base = prompt_preview if prompt_preview else prompt_payload
        if uploaded_filenames:
            names = ", ".join(uploaded_filenames)
            history_user_content = f"{_history_base}\n[SYSTEM: Images attached — {names}]"
        else:
            history_user_content = _history_base

        emperor.chat_history.append({"role": "user",      "content": history_user_content})
        # Build the full assistant history entry: any intermediate narrative text
        # (explanations/commentary written alongside tool calls) prepended to the
        # final response, separated by ---. Narratives were already printed to
        # terminal mid-turn; storing them here so /history shows the complete turn.
        _narratives = getattr(emperor, "_pending_narratives", [])
        _hist_parts  = [n for n in _narratives if n]
        if raw_response:
            _hist_parts.append(raw_response)
        _hist_content = "\n\n---\n\n".join(_hist_parts) if _hist_parts else (raw_response or "")
        emperor._pending_narratives = []   # reset for next turn
        # Append a single SYSTEM block listing all images shown this turn so
        # future turns and the reviewer both know what was displayed.
        _shown_imgs = getattr(emperor, "_images_shown_this_turn", [])
        if _shown_imgs:
            _hist_content += f"\n\n[SYSTEM: Images that were shown to user this turn (image visible in UI, not stored in history): {', '.join(_shown_imgs)}]"
            emperor._images_shown_this_turn = []
        emperor.chat_history.append({"role": "assistant", "content": _hist_content})
        emperor.save_history()
    except Exception as e:
        console.print(f"[{WARN}]history save error: {_esc(str(e))}[/{WARN}]")

    # Per-turn backup
    # NOTE: this used to check `"tool" in emperor._active_groups`, but
    # _active_groups only ever holds {"web","files","pdf","bash"} - "tool" is
    # never a member, so this branch was silently dead and every turn fell
    # through to the lightweight else-branch below, even turns that called
    # tools. Fixed to check whether any tool was actually invoked this turn.
    if emperor._current_tool_call_log:
        try:
            tsm.commit_turn(
                turn_num           = turn_num,
                tool_call_log      = emperor._current_tool_call_log,
                uploaded_filenames = uploaded_filenames,
                prompt_preview     = (prompt_preview or prompt_payload)[:120],
                rerun_of           = rerun_of,
                edited             = edited,
                images             = images,
            )
            console.print(f"[{META}]✓ T{turn_num} saved.[/{META}]")
        except Exception as e:
            console.print(f"[{WARN}]T{turn_num} backup error: {_esc(str(e))}[/{WARN}]")
    else:
        # Pure chat turn - no workspace changes to archive.
        # Still record a lightweight ledger entry so /turns shows the turn,
        # but skip the expensive scratch zip.
        try:
            tsm.commit_turn(
                turn_num           = turn_num,
                tool_call_log      = [],        # no tools used
                uploaded_filenames = uploaded_filenames,
                prompt_preview     = (prompt_preview or prompt_payload)[:120],
                rerun_of           = rerun_of,
                edited             = edited,
                images             = images,
            )
        except Exception:
            pass  # non-fatal for chat-only turns

    return raw_response

def get_bottom_toolbar():
    if emperor._active_groups:
        group_parts = []
        for g in ("web", "files", "pdf", "bash", "research"):
            if g in emperor._active_groups:
                group_parts.append(f'<style fg="ansicyan">{g}</style>')
            else:
                group_parts.append(f'<style fg="ansigray">{g}</style>')
        tool_str = ' <style fg="ansiwhite">|</style> '.join(group_parts)
    else:
        tool_str = '<style fg="ansigray">[tools off]</style>'

    tokens = getattr(emperor, "last_token_count", 0)
    if tokens > 0:
        if tokens >= 50000:
            color = "ansired"
            warn = " ⚠️ MAX LIMIT APPROACHING"
        elif tokens >= 20000:
            color = "ansiyellow"
            warn = " ⚠️"
        else:
            color = "ansicyan"
            warn = ""
        context_str = f' <style fg="ansiwhite">|</style> <style fg="{color}">[ Context: {tokens:,} tok{warn} ]</style>'
    else:
        context_str = ""

    return HTML(f'<style bg="ansiblack" fg="ansiwhite"> {tool_str}{context_str} </style>')


# ----
# SLASH COMMANDS
# ----
# Each command below is a small, self-contained function operating on the
# module-level singletons (emperor, tsm, web_agent, image_ocr) - the same
# convention _run_turn() above already uses. Splitting these out of the old
# flat while-loop body means:
#   - each command has real local scope (no more leading-underscore vars used
#     purely to avoid colliding with a sibling if-block in shared scope)
#   - each command can raise and get attributed, per-command error context
#     instead of one generic "error: {e}" for whichever of the nine branches
#     happened to break (see _dispatch_command / the main loop below)
#   - the /rerun <-> /edit tail-preservation duplication collapses into one
#     shared helper, _reappend_tail()

def _reappend_tail(tail_history: list, tail_ledger: list, target: int, current: int):
    """
    Re-append the conversation-tail turns (T{target+1}..T{current}) that were
    saved off before a /rerun or a user /edit truncated history down to
    T{target}. Previously copy-pasted identically at the end of both
    cmd_rerun() and cmd_edit(); now shared.
    """
    if tail_history:
        emperor.chat_history.extend(tail_history)
        # Partial 2 fix: tail_ledger is a pre-truncation SNAPSHOT captured by the
        # caller before truncate_history_only(..., keep_tail=True) ran - and that
        # call deliberately does NOT prune _ledger for keep_tail=True (see its
        # docstring), so entries for turns target+1..current are still sitting in
        # tsm._ledger the whole time this function's caller is running. Blindly
        # extending with tail_ledger therefore re-adds the SAME entries a second
        # time. Filter out anything whose turn number is already present before
        # extending, making this function safe to call from both the success and
        # cancel/failure paths without duplicating ledger entries.
        existing_turns     = {e.get("turn") for e in tsm._ledger}
        new_ledger_entries = [e for e in tail_ledger if e.get("turn") not in existing_turns]
        if new_ledger_entries:
            tsm._ledger.extend(new_ledger_entries)
            tsm._ledger.sort(key=lambda e: e.get("turn", 0))
            tsm._save_ledger()
        emperor.save_history(skip_global=True)
        console.print(f"[{META}]✓ T{target + 1}–T{current} conversation tail preserved.[/{META}]")
        
    # Clean up the WAL file now that the tail is safely on disk (or there was no tail).
    try:
        if os.path.exists(_RERUN_TAIL_FILE):
            os.remove(_RERUN_TAIL_FILE)
    except Exception:
        pass


def cmd_reset():
    """/reset - clear the current session, scratch files, and backups."""
    emperor.chat_history.clear()
    emperor.workspace_tracker.reset_workspace()

    # Code index persists across /reset (uploads survive the reset below, and
    # their embeddings are still valid) - just mark it stale so the next
    # semantic search walks disk again. scratch/ entries drop out naturally
    # (their files are gone); uploads/ hashes still match, so those chunks
    # are kept with zero extra API calls. Do NOT invalidate_index() here -
    # that deletes code_index.json and re-embeds uploads from scratch, which
    # is the opposite of "persists across /reset".
    if getattr(emperor, "code_search_agent", None):
        emperor.code_search_agent.mark_stale()

    # Wipe scratch folder - all model-generated files (code, OCR .txt, etc.)
    # Uploads are intentionally preserved - user placed them there manually.
    if os.path.isdir(config.SCRATCH_DIR):
        for item in os.listdir(config.SCRATCH_DIR):
            item_path = os.path.join(config.SCRATCH_DIR, item)
            try:
                if os.path.isfile(item_path):
                    _force_remove(item_path)  # Bug 36: handles read-only files on Windows
                elif os.path.isdir(item_path):
                    robust_rmtree(item_path)
            except Exception as e:
                console.print(f"[{META}]could not remove {_esc(str(item))}: {_esc(str(e))}[/{META}]")

    console.print(f"[{META}]wiping session data...[/{META}]")
    if os.path.exists(config.DATABASE_DIR):
        seen = set()
        for pattern in _RESET_PATTERNS:
            for filepath in glob.glob(os.path.join(config.DATABASE_DIR, pattern)):
                if not os.path.isfile(filepath):
                    continue
                if os.path.abspath(filepath) == os.path.abspath(config.WEB_CACHE_FILE):
                    continue
                if filepath in seen:
                    continue
                seen.add(filepath)
                try:
                    os.remove(filepath)
                except Exception as e:
                    console.print(f"[{META}]could not remove {_esc(os.path.basename(filepath))}: {_esc(str(e))}[/{META}]")

    tsm.clear_all()
    emperor.last_tool_summary = ""
    emperor.last_cancelled_summary = ""  # Bug 14 follow-up: full reset on /reset
    emperor._partial_messages = None
    emperor._current_tool_call_log = []
    emperor.save_history(skip_global=True)

    console.print(f"[{META}] [/{META}]")


def cmd_tool():
    """/tool - open the tool-group checkbox selector."""
    global DOCKER_AVAILABLE

    dialog_style = PTStyle.from_dict({
        "dialog":             "bg:#1e1e2e",
        "dialog.body":        "bg:#1e1e2e fg:#cdd6f4",
        "dialog frame.label": "bg:#1e1e2e fg:#89b4fa bold",
        "checkbox":           "fg:#cdd6f4",
        "checkbox-checked":   "fg:#a6e3a1 bold",
        "button":             "bg:#313244 fg:#cdd6f4",
        "button.focused":     "bg:#89b4fa fg:#1e1e2e bold",
    })

    all_groups = ["web", "files", "pdf", "bash", "research"]
    descriptions = {
        "web":      "web      — quick_search, url_search",
        "files":    "files    — view_lines, search_in_file, str_replace, workspace_search, search_history, ingest_chat",
        "pdf":      "pdf      — ingest_pdf, ingest_text, doc_search, view_lines",
        "bash":     "bash     — bash sandbox, show_image (requires Docker)",
        "research": "research — search_semantic_scholar (Semantic Scholar academic database)",
    }

    current_selection = [g for g in all_groups if g in emperor._active_groups]

    selected = checkboxlist_dialog(
        title="Select Active Tool Groups",
        text="Space to toggle  |  Tab to switch  |  Enter to confirm",
        values=[(g, descriptions[g]) for g in all_groups],
        default_values=current_selection,
        style=dialog_style,
    ).run()

    if selected is None:  # user pressed Cancel
        console.print(f"[{META}]Tool selection cancelled.[/{META}]\n")
        return

    new_groups = set(selected)

    # Guard: bash requires Docker
    if "bash" in new_groups and not DOCKER_AVAILABLE:
        console.print(f"[{META}]Checking Docker availability...[/{META}]")
        DOCKER_AVAILABLE = _setup_sandbox()
        if not DOCKER_AVAILABLE:
            console.print(f"[{WARN}]⚠  bash group requires Docker, which is still not running.[/{WARN}]")
            console.print(f"[{META}]   bash was removed from selection. Start Docker and try again.[/{META}]")
            new_groups.discard("bash")

    emperor.set_tool_groups(new_groups)
    console.print(emperor.tools_status())

    # Re-warm local system-prompt cache if on local backend
    if config.ACTIVE_BACKEND == "local":
        threading.Thread(
            target=emperor.warm_local_system_prompt,
            args=(emperor._base_system,),
            daemon=True,
        ).start()
        console.print(f"[dim]  (re-warming local cache for new system-prompt shape...)[/dim]")
    console.print()


def cmd_switch_backend(user_text: str):
    """/nim, /google, /cf, /cloudflare, /local - switch the active LLM backend."""
    backend = "cloudflare" if user_text in ("/cf", "/cloudflare") else user_text[1:]
    console.print(emperor.switch_backend(backend))
    console.print()


def cmd_tools_status():
    """/tools - display current tool mode and active backend."""
    console.print(emperor.tools_status())
    console.print()


def cmd_ctx():
    """/ctx - toggle workspace + uploads context injection on/off.
    When off, [WORKSPACE] and [UPLOADS FOLDER] are not sent to the model.
    Last-turn-actions and cancelled-turn blocks are unaffected.
    """
    emperor._ctx_enabled = not emperor._ctx_enabled
    state = "[green]ON[/green]" if emperor._ctx_enabled else "[red]OFF[/red]"
    console.print(f"[{META}]Workspace/uploads context: {state}[/{META}]\n")


def cmd_history():
    """/history - print the full conversation history."""
    if not emperor.chat_history:
        console.print(f"[{META}]no history yet.[/{META}]")
        return

    rule(style=RULE_STYLE)
    turn = 0
    for i, msg in enumerate(emperor.chat_history):
        role = msg.get("role", "")
        content = _flatten_content(msg.get("content", ""))
        if role == "user":
            turn = i // 2 + 1
            console.print(f"[bold bright_magenta][T{turn}] YOU:[/bold bright_magenta] {_esc(content[:500])}")

            matches = re.findall(r'\[SYSTEM: Images attached \u2014\s*(.*?)\]', content)
            for match in matches:
                for filename in match.split(","):
                    fname = filename.strip()
                    if "://" in fname:
                        console.print(f"[dim](remote image: {_esc(fname[:60])})[/dim]")
                        continue
                    img_path = os.path.join(config.SCRATCH_DIR, fname)
                    if os.path.exists(img_path):
                        show_image_in_terminal(img_path)

        elif role == "assistant":
            pad = " " * (len(str(turn)) + 4)
            console.print(f"[bold cyan]{pad}AI:[/bold cyan]")
            print_smart_response(content)
        console.print()
    rule(style=RULE_STYLE)


def cmd_turns():
    """/turns - display the turn timeline and revision markers."""
    console.print()
    console.print(tsm.list_turns())
    console.print()


def cmd_review(user_text: str):
    """/review [focus] - self-review the latest response using the same model + tools.

    /review             → plain review
    /review <focus>     → review with a user-directed focus appended
    """
    if len(emperor.chat_history) < 2:
        console.print(f"[{META}]No previous response to review.[/{META}]\n")
        return

    parts = user_text.split(" ", 1)
    user_focus = parts[1].strip() if len(parts) > 1 else ""

    review_prompt = user_focus if user_focus else "Review."

    try:
        raw = emperor.generate_review(
            review_prompt=review_prompt,
            web_agent=web_agent,
            token_counter=token_counter,
        )
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]⚡ review cancelled[/bold yellow]\n")
        return

    if raw:
        console.print()
        rule(style=RULE_STYLE)
        print_smart_response(raw)
        rule(style=RULE_STYLE)
        console.print()


def cmd_restore(user_text: str):
    """/restore N | /restore last - revert workspace and memory to a turn."""
    parts = user_text.split()
    if len(parts) < 2:
        console.print(f"[{WARN}]Usage: /restore N  or  /restore last[/{WARN}]")
        return

    arg          = parts[1].lower().lstrip("#")
    current_turn = len(emperor.chat_history) // 2

    if arg == "last":
        target = current_turn - 1
    else:
        try:
            target = int(arg)
        except ValueError:
            console.print(f"[{WARN}]Invalid turn: '{parts[1]}'. Use a number or 'last'.[/{WARN}]")
            return

    if target < 1:
        console.print(f"[{WARN}]No previous turn to restore to. Use /reset to wipe the session.[/{WARN}]")
        return

    if target >= current_turn:
        console.print(f"[{WARN}]T{target} is the current or future turn. Only past turns can be restored.[/{WARN}]")
        return

    # Inform user what will change
    console.print()
    console.print(f"[{WARN}]⚠  Restore to end of T{target}[/{WARN}]")
    console.print(f"[{META}]   scratch/ will revert to its T{target} state.[/{META}]")
    console.print(f"[{META}]   Memory (turns {target+1}–{current_turn}) will be erased.[/{META}]")

    # Collect uploads/outputs warnings from future ledger entries
    future = [e for e in tsm._ledger if e.get("turn", 0) > target]
    all_uploads = [f for e in future for f in e.get("user_uploads", [])]
    all_outputs = [f for e in future for f in e.get("outputs_created", [])]
    if all_uploads:
        console.print(f"[{META}]   uploads/ unchanged — {', '.join(all_uploads)} remain (model won't remember them).[/{META}]")
    if all_outputs:
        console.print(f"[{META}]   outputs/ unchanged — {', '.join(all_outputs)} remain.[/{META}]")

    console.print()
    try:
        confirm = prompt(HTML('<b> Type YES to confirm restore: </b>')).strip()
    except (KeyboardInterrupt, EOFError):
        confirm = ""
    if confirm != "YES":
        console.print(f"[{META}]Restore cancelled.[/{META}]\n")
        return

    console.print(f"[{META}]restoring...[/{META}]")
    console.print(tsm.restore_turn(target, emperor))
    console.print()


def cmd_rerun(user_text: str):
    """/rerun N | /rerun last - regenerate the AI response at turn N."""
    parts = user_text.split()
    if len(parts) < 2:
        console.print(f"[{WARN}]Usage: /rerun N[/{WARN}]")
        console.print(f"[{META}]  Retries the AI response at turn N using the same prompt.[/{META}]")
        return
    current = len(emperor.chat_history) // 2

    # Resolve 'last' shorthand
    if parts[1].lower() == "last":
        target = current
    else:
        try:
            target = int(parts[1])
        except ValueError:
            console.print(f"[{WARN}]Invalid turn: '{parts[1]}'. Use a number or 'last'.[/{WARN}]")
            return

    if target < 1 or target > current:
        console.print(f"[{WARN}]T{target} out of range (1–{current}).[/{WARN}]")
        return

    user_idx        = (target - 1) * 2
    original_prompt = _flatten_content(emperor.chat_history[user_idx].get("content", ""))

    # Images: hard-stop if T{target} had images but they're not
    # reloadable, rather than silently proceeding text-only.
    # Bug 16: use ledger images_snap field instead of substring prompt check
    # so a user prompt mentioning the marker string can't spoof this guard.
    _rerun_entry = next((e for e in tsm._ledger if e.get("turn") == target), None)
    had_images   = bool(_rerun_entry and _rerun_entry.get("images_snap"))
    rerun_images = None
    if had_images:
        img_names = re.findall(r'\[SYSTEM: Images attached \u2014\s*(.*?)\]', original_prompt)
        img_list  = img_names[0] if img_names else "unknown"
        rerun_images = tsm.load_images_snapshot(target)
        if not rerun_images:
            console.print()
            console.print(f"[{WARN}]✗ Cannot rerun T{target}: it had images attached ({img_list}), "
                          f"but their snapshot is missing or was pruned.[/{WARN}]")
            console.print(f"[{META}]  Rerunning without them would silently drop images the original "
                          f"turn depended on — refusing instead. Try a more recent turn, "
                          f"or resend this as a new message with the image(s) attached.[/{META}]\n")
            return
        # Strip the stored "[SYSTEM: Images attached - ...]" text marker back off -
        # _run_turn will regenerate it correctly from the real filenames
        # once we pass `images=` back in below.
        original_prompt = re.sub(r'\n\[SYSTEM: Images attached.*?\]\s*$', '', original_prompt)

    console.print()
    console.print(f"[{WARN}]⚠  Rerun T{target}[/{WARN}]")
    console.print(f"[{META}]  prompt: {_esc(original_prompt[:100])}{'...' if len(original_prompt) > 100 else ''}[/{META}]")
    if had_images:
        console.print(f"[{META}]  {len(rerun_images)} image(s) will be re-sent.[/{META}]")
    console.print(f"[{META}]  Scratch/workspace files will NOT be reverted — anything T{target}"
                  f"–T{current} created stays on disk as-is.[/{META}]")
    if current > target:
        console.print(f"[{META}]  Turns T{target + 1}–T{current} will be kept in conversation memory.[/{META}]")

    # Bug 17 fix: persist T{target} itself into the WAL alongside the tail,
    # so a crash mid-regeneration can restore the original turn instead of
    # leaving a gap.
    tail_history = emperor.chat_history[(target - 1) * 2:]
    tail_ledger  = [e for e in tsm._ledger if e.get("turn", 0) >= target]

    # Persist tail to WAL before truncating (crash recovery)
    if tail_history:
        _save_rerun_tail(target, current, tail_history, tail_ledger)

    # Truncate history only - keep_tail=True skips archive/ledger
    # deletion for T(target+1)..T(current), preserving those turns.
    console.print(f"[{META}]truncating history to T{target - 1}...[/{META}]")
    console.print(tsm.truncate_history_only(target - 1, emperor, keep_tail=True))

    # Bug #18 fix: DON'T delete T{target}'s archive here - it must
    # survive until _run_turn completes successfully. Deletion is deferred
    # to the success branch so a cancelled/failed rerun leaves the
    # original turn's archive intact and /restore still works.

    rerun_filenames = [img["filename"] for img in rerun_images] if rerun_images else None

    console.print(f"[{META}]Rerunning T{target}...[/{META}]\n")
    emperor.last_tool_summary = ""  # clear stale tool context from old turn
    # Bug 3 (was Bug 1): _run_turn() catches KeyboardInterrupt/cancellation
    # internally and returns None instead of raising - it does NOT throw an
    # exception on cancel. The old try/except/else here treated "no exception
    # raised" as success, so a cancelled rerun still hit the `else` branch and
    # re-appended the tail at offset `target` as if T{target} had been
    # regenerated and committed. It never was (the cancel path never calls
    # commit_turn()), so T{target} was permanently lost. Check the return
    # value explicitly instead of relying on exception control flow.
    result = None
    try:
        # Strip any [IMAGE OCR CONTENT] block that may exist in old history
        # entries saved before the vision=False history fix. prompt_preview
        # tells _run_turn's history-save to use this clean version instead of
        # re-storing the expanded OCR payload.
        _rerun_preview = re.sub(r'\s*\[IMAGE OCR CONTENT\].*$', '', original_prompt, flags=re.DOTALL).strip()
        result = _run_turn(original_prompt, images=rerun_images, uploaded_filenames=rerun_filenames,
                           rerun_of=target, prompt_preview=_rerun_preview)
    except Exception as e:
        console.print(f"[{ERR}]rerun generation failed: {_esc(str(e))}[/{ERR}]")
        # Re-append at target-1 so the gap left by truncation is filled back in.
        _reappend_tail(tail_history, tail_ledger, target - 1, current)
        return

    if result is None:
        # Cancelled - T{target} was never committed, same as an exception
        # from this function's perspective: re-stitch at target-1 (no gap).
        console.print(f"[{WARN}]rerun of T{target} was cancelled — restoring prior conversation tail.[/{WARN}]")
        _reappend_tail(tail_history, tail_ledger, target - 1, current)
        return

    # Success: the new T{target} is committed - now safe to delete the old archive.
    # Bug #18 fix: deletion happens here (post-success), not before _run_turn.
    tsm.delete_turn_archives(target)

    # Re-append tail after the newly generated T{target}.
    # Slice the old target turn off both the history and ledger so we don't
    # append it immediately after the new target turn.
    _reappend_tail(tail_history[2:], [e for e in tail_ledger if e.get("turn") != target], target, current)


def cmd_delete(user_text: str):
    """/delete N - remove a turn pair from conversation history."""
    parts = user_text.split()
    if len(parts) < 2:
        console.print(f"[{WARN}]Usage: /delete N[/{WARN}]")
        console.print(f"[{META}]  Permanently removes turn N (user+assistant pair) from conversation memory.[/{META}]")
        console.print(f"[{META}]  All subsequent turns are renumbered. The live workspace is NOT changed.[/{META}]")
        return

    current = len(emperor.chat_history) // 2
    if current == 0:
        console.print(f"[{WARN}]No turns to delete.[/{WARN}]")
        return

    try:
        target = int(parts[1])
    except ValueError:
        console.print(f"[{WARN}]Invalid turn number: '{parts[1]}'. Use an integer.[/{WARN}]")
        return

    if target < 1 or target > current:
        console.print(f"[{WARN}]T{target} out of range (1–{current}).[/{WARN}]")
        return

    # Preview the turn being deleted
    user_idx = (target - 1) * 2
    preview  = _flatten_content(emperor.chat_history[user_idx].get("content", ""))

    # Check whether an archive exists for this turn
    del_entry   = next((e for e in tsm._ledger if e.get("turn") == target), None)
    has_archive = bool(
        del_entry and del_entry.get("scratch_zip") and
        os.path.exists(os.path.join(config.BACKUPS_DIR, del_entry["scratch_zip"]))
    )

    # Show what will happen
    console.print()
    console.print(f"[{WARN}]⚠  Delete T{target}[/{WARN}]")
    console.print(
        f"[{META}]  prompt : {preview[:100]}"
        f"{'...' if len(preview) > 100 else ''}[/{META}]"
    )
    console.print(f"[{META}]  This user+assistant pair will be erased from conversation memory.[/{META}]")
    if target < current:
        console.print(
            f"[{META}]  Turns T{target+1}–T{current} will be renumbered "
            f"to T{target}–T{current-1}.[/{META}]"
        )
    if has_archive:
        console.print(
            f"[{META}]  T{target} scratch archive will be deleted; "
            f"subsequent archives renumbered.[/{META}]"
        )
    console.print(
        f"[{WARN}]  The live workspace (scratch/) is NOT changed — "
        f"only conversation memory.[/{WARN}]"
    )

    console.print(f"[{META}]deleting T{target}...[/{META}]")
    console.print(tsm.delete_turn(target, emperor))
    console.print()


def cmd_edit(user_text: str):
    """/edit N [user|agent] - rewrite a past message."""
    parts = user_text.split()
    if len(parts) < 2:
        console.print(f"[{WARN}]Usage: /edit N [user|agent][/{WARN}]")
        console.print(f"[{META}]  Edit a past message. Defaults to editing the user message.[/{META}]")
        return
    current = len(emperor.chat_history) // 2

    # Resolve 'last' shorthand
    if parts[1].lower() == "last":
        target = current
    else:
        try:
            target = int(parts[1])
        except ValueError:
            console.print(f"[{WARN}]Invalid turn: '{parts[1]}'. Use a number or 'last'.[/{WARN}]")
            return

    role = parts[2].lower() if len(parts) > 2 else "user"
    if role not in ("user", "agent"):
        console.print(f"[{WARN}]Role must be 'user' or 'agent'.[/{WARN}]")
        return

    if target < 1 or target > current:
        console.print(f"[{WARN}]T{target} out of range (1–{current}).[/{WARN}]")
        return

    # Index: user message = (N-1)*2, agent message = (N-1)*2 + 1
    msg_idx     = (target - 1) * 2 + (1 if role == "agent" else 0)
    old_content = _flatten_content(emperor.chat_history[msg_idx].get("content", ""))

    # Open the current message in the user's $EDITOR (default: nano) for
    # in-place editing - pre-filled, full cursor movement, no truncation,
    # and no blank-line-terminated input() footgun for multiline content.
    console.print()
    console.print(f"── Editing T{target} {role} message {'─' * (50 - len(str(target)) - len(role))}")
    console.print(f"[{META}]Opening in $EDITOR ({os.environ.get('EDITOR', 'nano')})...[/{META}]")

    new_content = edit_in_external_editor(str(old_content))

    if new_content is None:
        # Editor unavailable - fall back to the original line-by-line
        # prompt so /edit still works, with a clear caveat about blank lines.
        console.print(f"[{META}]Falling back to inline input (blank line ends input — do not include blank lines mid-message):[/{META}]")
        console.print(str(old_content)[:600])
        if len(str(old_content)) > 600:
            console.print(f"[{META}]  ... ({len(str(old_content)) - 600} more chars)[/{META}]")
        console.print("─" * 60)
        console.print(f"Type replacement (empty line to finish, Ctrl+C to cancel):")

        lines = []
        try:
            while True:
                try:
                    line = prompt(HTML('<b>❯ </b>'))
                except EOFError:
                    break
                if line == "":
                    break
                lines.append(line)
        except KeyboardInterrupt:
            console.print(f"\n[{META}]Edit cancelled.[/{META}]\n")
            return

        if not lines:
            console.print(f"[{META}]No input — edit cancelled.[/{META}]\n")
            return

        new_content = "\n".join(lines)

    if not new_content.strip():
        console.print(f"[{META}]Empty content — edit cancelled.[/{META}]\n")
        return

    if new_content == str(old_content):
        console.print(f"[{META}]No changes made — edit cancelled.[/{META}]\n")
        return

    # Images: same hard-stop pattern as /rerun. Only relevant for
    # user edits - an agent edit doesn't regenerate anything, so
    # nothing gets resent and there's no image to preserve.
    edit_images = None
    if role == "user":
        # Bug 16: use ledger images_snap instead of substring prompt check.
        _edit_entry = next((e for e in tsm._ledger if e.get("turn") == target), None)
        had_images  = bool(_edit_entry and _edit_entry.get("images_snap"))
        if had_images:
            img_names = re.findall(r'\[SYSTEM: Images attached \u2014\s*(.*?)\]', str(old_content))
            img_list  = img_names[0] if img_names else "unknown"
            edit_images = tsm.load_images_snapshot(target)
            if not edit_images:
                console.print()
                console.print(f"[{WARN}]✗ Cannot edit T{target}: it had images attached ({img_list}), "
                              f"but their snapshot is missing or was pruned.[/{WARN}]")
                console.print(f"[{META}]  Regenerating without them would silently drop images the "
                              f"original turn depended on — refusing instead.[/{META}]\n")
                return
            # new_content is the edited text of old_content, which still
            # has the marker baked in (it was the pre-filled editor text) -
            # strip it so _run_turn regenerates it correctly from the real images.
            new_content = re.sub(r'\n\[SYSTEM: Images attached.*?\]\s*$', '', new_content)
            # Bug #19 fix: re-add the marker from the snapshot filenames so
            # the stored message still references the images that were attached.
            # Use U+2014 em-dash to match the regex in cmd_rerun's image marker.
            if edit_images:
                _snap_names = ', '.join(img['filename'] for img in edit_images)
                new_content = f"{new_content}\n[SYSTEM: Images attached \u2014 {_snap_names}]"

    # In-place patch (both user and agent)
    emperor.chat_history[msg_idx]["content"] = new_content
    emperor.save_history(skip_global=True)
    emperor.last_tool_summary = ""
    emperor.last_cancelled_summary = ""  # Bug 14 follow-up: stale after an edit too

    console.print(f"[{META}]✓ T{target} {role} message updated.[/{META}]")
    if role == "user":
        console.print(f"[dim]  Response NOT regenerated — run /rerun {target} to regenerate.[/dim]")
    else:
        if current > target:
            console.print(f"[dim]  Turns T{target + 1}–T{current} still in memory. "
                          f"Use /rerun {target} if the new response implies different files.[/dim]")
    console.print()


# Exact-match commands (take no arguments).
_COMMANDS_EXACT = {
    "/reset":   cmd_reset,
    "/tool":    cmd_tool,
    "/tools":   cmd_tools_status,
    "/ctx":     cmd_ctx,
    "/history": cmd_history,
    "/turns":   cmd_turns,
}
# Prefix commands - take a turn number / role argument, so they parse the
# raw user_text themselves (e.g. "/rerun 3", "/edit last agent").
# /review is also a prefix command since it optionally takes a focus string.
_COMMANDS_PREFIX = [
    ("/restore", cmd_restore),
    ("/rerun",   cmd_rerun),
    ("/delete",  cmd_delete),
    ("/edit",    cmd_edit),
    ("/review",  cmd_review),
]
_BACKEND_COMMANDS = ("/nim", "/google", "/cf", "/cloudflare", "/local", "/tr", "/tokenrouter")


def _dispatch_command(user_text: str):
    """
    Resolve user_text to a zero-arg callable, or None if it isn't a
    recognized slash command (i.e. it's a normal chat turn and should fall
    through to the image-handling + _run_turn path below).
    """
    if user_text in _COMMANDS_EXACT:
        return _COMMANDS_EXACT[user_text]
    if user_text in _BACKEND_COMMANDS:
        return lambda: cmd_switch_backend(user_text)
    # Bug 5: require a token boundary after the command word so natural
    # prompts starting with the same letters ("/rerunner my query",
    # "/restore_backup") don't get silently swallowed by a command handler.
    for prefix, fn in _COMMANDS_PREFIX:
        if user_text == prefix or user_text.startswith(prefix + " "):
            return lambda: fn(user_text)
    return None


# Startup header
rule()
_backend_label = config.ACTIVE_BACKEND
_model_label   = config.get_active_model().split("/")[-1]
console.print(
    f"[bold white]🏛[/bold white]  "
    f"[dim cyan]{_backend_label} / {_model_label}[/dim cyan]  "

    f"[dim]   Nav:      /tool  /tools  /ctx  /turns  /history  /reset  /review\n"
    f"   Backends: /google  /nim  /cf  /local  /tr    Revisions: /restore N  /rerun N  /edit N  /delete N[/dim]"
)
rule(style=RULE_STYLE)
console.print()

# Rerun WAL crash recovery
# If the process was killed mid-rerun (after truncate, before _reappend_tail),
# the tail is on disk in _RERUN_TAIL_FILE. Auto-recover it now.
if os.path.exists(_RERUN_TAIL_FILE):
    try:
        with open(_RERUN_TAIL_FILE, encoding="utf-8") as _wf:
            _wal = json.load(_wf)
        _w_target  = _wal.get("target", "?")
        _w_current = _wal.get("current", "?")
        _w_th      = _wal.get("tail_history", [])
        _w_tl      = _wal.get("tail_ledger",  [])
        if _w_th:
            console.print(
                f"[{WARN}]⚠  Incomplete /rerun T{_w_target} detected — process was killed "
                f"before T{_w_target + 1 if isinstance(_w_target, int) else '?'}–T{_w_current} "
                f"were reappended. Auto-recovering {len(_w_th) // 2} tail turn(s)...[/{WARN}]"
            )
            
            # If the crash happened after the regenerated turn was committed to history,
            # slice the old target turn off the tail to prevent duplication.
            if isinstance(_w_target, int) and len(emperor.chat_history) >= _w_target * 2:
                _w_th = _w_th[2:]
                _w_tl = [e for e in _w_tl if e.get("turn") != _w_target]
                
            _reappend_tail(_w_th, _w_tl,
                           _w_target if isinstance(_w_target, int) else 0,
                           _w_current if isinstance(_w_current, int) else 0)
        else:
            os.remove(_RERUN_TAIL_FILE)   # empty / corrupt - discard
    except Exception as _wal_err:
        console.print(f"[{WARN}]rerun WAL recovery failed: {_esc(str(_wal_err))}[/{WARN}]")
        try:
            os.remove(_RERUN_TAIL_FILE)
        except Exception:
            pass

# ---- Main loop ----

while True:
    try:
        turn_num = len(emperor.chat_history) // 2 + 1
        try:
            user_text = prompt(
                HTML(f'<ansimagenta><b> [T{turn_num}] ❯ </b></ansimagenta>'),
                completer=slash_completer,
                complete_while_typing=True,
                bottom_toolbar=get_bottom_toolbar
            )
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() == "exit":
            break

        # Slash-command dispatch
        # Each command owns real function scope now, and a failure inside one
        # is reported against that command specifically instead of a single
        # generic "error: {e}" shared by all nine branches.
        command = _dispatch_command(user_text)
        if command is not None:
            try:
                command()
            except Exception as e:
                console.print(f"[{ERR}]error in {_esc(user_text.split()[0])}: {_esc(str(e))}[/{ERR}]")
            continue

        # Image handling
        image_input = console.input(f"[{META}] images (path/url or enter to skip): [/{META}]").strip()
        images      = []   # for direct vision path
        ocr_text    = ""   # for OCR fallback path
        uploaded_filenames = []

        if image_input:
            try:
                items = shlex.split(image_input, posix=False)
            except Exception:
                items = image_input.split()
            for i in [i.strip().strip('"').strip("'") for i in items if i.strip()]:
                uploaded_filenames.append(os.path.basename(i) if not i.startswith("http") else i.split("/")[-1])
            if config.SUPPORTS_VISION:
                images = load_images_for_vision(image_input)
                if images:
                    names = ", ".join(i["filename"] for i in images)
                    console.print(f"[{META}]  sending {len(images)} image(s) directly to model: {names}[/{META}]")
            else:
                console.print(f"[{META}]  model has no vision — routing through OCR pipeline (NIM)...[/{META}]")
                ocr_text = process_images_via_ocr(image_input, image_ocr)

        # Build prompt
        if ocr_text:
            prompt_payload = f"{user_text}\n\n[IMAGE OCR CONTENT]\n{ocr_text}"
        else:
            prompt_payload = user_text

        # Normal turn
        # Pass user_text as prompt_preview so the ledger always shows the raw
        # question even when prompt_payload has been expanded with OCR content.
        _run_turn(prompt_payload, images=images, uploaded_filenames=uploaded_filenames,
                  prompt_preview=user_text)

    except KeyboardInterrupt:
        break
    except Exception as e:
        console.print(f"[{ERR}]error: {_esc(str(e))}[/{ERR}]")