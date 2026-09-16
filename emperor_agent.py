# emperor_agent.py - Core agent: initialisation, generation loop, history, public API.
#
# Tool definitions  →  core_tool_definitions.py  (ALL tools + ALL system-prompt snippets)
# Tool handlers     →  tool_handlers.py           (ToolHandlersMixin)
#
# EmperorAgent inherits ToolHandlersMixin so its external interface is
# identical to the original single-file version.

import json
import os
import re
import shutil
import threading
from google import genai

import config

from utils import read_api_key, extract_thinking_tags, GlobalHistoryWriter, save_json_atomic, robust_rmtree, EXCL_PREFIXES, EXCL_SUFFIXES, EXCL_DIRS, console
from workspace_tracker import WorkspaceTracker
from file_ops_agent import FileOpsAgent
from tool_summarizer import ToolCallSummarizer
from doc_search_agent import DocSearchAgent
from search_history_agent import SearchHistoryAgent
from code_search_agent import CodeSearchAgent

from core_tool_definitions import (
    PSEUDO_TOOL_FORMAT,
    CORE_PROMPT,
    FILE_OPS_PROMPT,
    GROUP_PROMPTS,
    GROUP_TOOLS,
    GROUP_TOOLS_UNION,
    web_date_context,
)

# Mixins: tool handlers + LLM API backends
from tool_handlers import ToolHandlersMixin
from llm_backends import LLMBackendsMixin
from renderer import print_smart_response, RULE_STYLE

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

_BASE_SYSTEM_PROMPT = """You are an assistant.

# How to think before answering? :

## 1. Handle Ambiguity Directly
If a request is genuinely ambiguous and proceeding on the wrong interpretation would waste the user's time — **ask**. If the ambiguity is minor, state your assumption and proceed.

## 2. Choose Your Posture
Decide how to engage before engaging:
- **Just answer** — clear request, execute it well
- **Answer and enrich** — answer what was asked, add what they need
- **Reframe then answer** — surface the better question, then answer both
- **Push back** — wrong assumption, gently correct it first
- **Ask first** — too ambiguous to proceed usefully

## 3. Correctness Over Comfort
Being right matters more than being agreeable — if the user is wrong, say so plainly instead of softening into agreement. This holds after the first answer too: if they push back or get frustrated, re-check your reasoning, but don't cave just because they're unhappy — emotional pressure isn't evidence you were wrong.
The flip side: when you *are* wrong, say what was wrong, make the correction, move on.

If user asks for any advice like "what to do", factor in real world constraints and give a answer which is relevant to user, not simply emit whatever you know.

## 4. Recall and think more and more — many times with different perspectives.
"""


_REVIEW_SYSTEM_PROMPT = """You have been provided with the conversation history. Your job is to review your most recent response (the last assistant message). 

Look at it with fresh eyes. Doubt each and everything and ask yourself:
- Is anything wrong, inaccurate, hallucinations/typos?
- Did I miss something the user needed?
- If the user is working with coding work, check the codebase and see if there are any bugs, gaps, or things that could go wrong.
- Is there a valuable nuance, caveat, related concept, or important heads-up the user should know — even if they didn't ask? If yes, surface it.
- Recall, think and reason more and more — many times with different perspectives.

If everything looks good — say "All good".
"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class EmperorAgent(LLMBackendsMixin, ToolHandlersMixin):
    """
    Main agent (STEM assistant).

    Inherits all tool-handler methods from ToolHandlersMixin.
    This file contains only initialisation, the generation loop, and
    history / public-API methods.
    """

    def __init__(self, api_key_ignored=None, ocr_agent=None, docker_available: bool = False):
        try:
            # ── Google backend ─────────────────────────────────────────────────
            # Load a pool of keys from GOOGLE_API_KEY_FILE_POOL (one per line).
            # If the pool file is absent or empty, fall back to the single key
            # from GOOGLE_API_KEY_FILE so existing setups are unaffected.
            _pool_keys = []
            if config.GOOGLE_API_KEY_FILE_POOL:
                try:
                    with open(config.GOOGLE_API_KEY_FILE_POOL, "r", encoding="utf-8") as _f:
                        _pool_keys = [ln.strip() for ln in _f if ln.strip()]
                except FileNotFoundError:
                    pass  # pool file missing — fall through to single-key fallback
                except Exception as _e:
                    console.print(f"⚠️  [yellow]Could not read Google key pool: {_e}[/yellow]")

            if not _pool_keys:
                # Single-key fallback (original behaviour)
                _single = read_api_key(config.GOOGLE_API_KEY_FILE, silence_warning=True)
                _pool_keys = [_single] if _single else []

            self._google_key_pool  = _pool_keys
            self._google_key_index = 0
            self.google_client = (
                genai.Client(api_key=_pool_keys[0]) if _pool_keys else None
            )

            # ── NIM backend ────────────────────────────────────────────────────
            self.nim_api_key = read_api_key(config.NVIDIA_API_KEY_FILE, silence_warning=True)
            self.nim_base_url = config.NVIDIA_BASE_URL
            self.chat_history = []
            self.last_token_count = 0
            self._load_history()

            # ── Cloudflare backend ─────────────────────────────────────────────
            self.cf_api_key = read_api_key(config.CLOUDFLARE_API_KEY_FILE, silence_warning=True)
            self.cf_base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.CLOUDFLARE_ACCOUNT_ID}/ai/v1"
            # Bug 7: CLOUDFLARE_ACCOUNT_ID defaults to "" when unset, which silently
            # builds a malformed URL (".../accounts//ai/v1") that 404s with no
            # diagnostic. Track configuredness here so switch_backend() and the
            # request path can fail fast with a clear message instead.
            self.cf_configured = bool(config.CLOUDFLARE_ACCOUNT_ID)
            if not self.cf_configured:
                console.print(
                    "[yellow]⚠️  CLOUDFLARE_ACCOUNT_ID is not set in .env — "
                    "the Cloudflare backend (/cf) will be unavailable until it is.[/yellow]"
                )

            # ── Local backend (llama-server) ────────────────────────────────────
            self.local_api_key = (
                read_api_key(config.LOCAL_API_KEY_FILE, silence_warning=True)
                if config.LOCAL_API_KEY_FILE else None
            )
            self.local_base_url = config.LOCAL_BASE_URL

            # ── Token Router backend ────────────────────────────────────────────
            self.tr_api_key = read_api_key(config.TOKEN_ROUTER_API_KEY_FILE, silence_warning=True)

            self.workspace_tracker = WorkspaceTracker(workspace_path=config.SCRATCH_DIR)

            self.code_search_agent = CodeSearchAgent(
                search_dirs=[config.SCRATCH_DIR, config.UPLOADS_FOLDER],
                index_dir=config.CODE_INDEX_DIR,
            )

            self.ocr_agent    = ocr_agent
            self.file_ops = FileOpsAgent(
                workspace_root=config.SCRATCH_DIR,
                workspace_tracker=self.workspace_tracker,
                code_search_agent=self.code_search_agent,
            )
            self.doc_search_agent = DocSearchAgent(scratch_dir=config.SCRATCH_DIR)
            self.search_history_agent = SearchHistoryAgent()
            self.summarizer = ToolCallSummarizer()
            self.last_tool_summary = ""  # populated after each agentic turn
            # Bug 14: separate field for cancelled/rolled-back turns — never
            # conflated with last_tool_summary, which must only reflect
            # committed turns (see _save_partial_history / generate_with_history).
            self.last_cancelled_summary = ""
            self.global_history = GlobalHistoryWriter()

            # Interrupt-resume state: holds the mid-turn messages list if the
            # user presses Ctrl+C during a tool loop. Survives the stack unwind
            # so resume_from_interrupt() can continue exactly where we left off.
            self._partial_messages = None

            # Accumulates every tool call this turn. Lives on self (not as a
            # local variable) so it survives Ctrl+C and continues accumulating
            # across resume_from_interrupt() calls. Reset at the start of each
            # new non-resume turn. Used by start.py to build partial history on cancel.
            self._current_tool_call_log = []

            # Alien-format detection — classifier loaded lazily on first use.
            # _alien_nudge_sent prevents re-triggering after a nudge is injected.
            self._alien_detector = None
            self._alien_nudge_sent: bool = False

            # Narrative text the model wrote alongside tool calls during a turn
            # (e.g. a long explanation before calling bash). Lives on self so
            # start.py can read it when writing to chat_history after the turn
            # ends, without having to join it into the return value (which would
            # cause it to be printed twice in the terminal).
            self._pending_narratives = []

            # Filenames of images shown to the user via show_image this turn.
            # Collected by _handle_show_image; appended as a single [SYSTEM] block
            # to the assistant chat_history entry in start.py so future turns and
            # the reviewer both know what was displayed.
            self._images_shown_this_turn: list = []

            # All thinking/reasoning traces from the most recent main turn.
            # Collected per-iteration in _generate_with_tools; read by
            # generate_review for the second-pass trace-informed review.
            # Transient — session memory only, never saved to chat_history.
            self.last_think_trace: str = ""

            # Controls whether workspace + uploads context is injected into
            # each turn's user message. Toggled by /ctx command.
            # last-turn-actions and cancelled-turn blocks are unaffected.
            self._ctx_enabled: bool = False


            # Build dispatch table (defined in ToolHandlersMixin)
            self._dispatch = self._build_dispatch()

            # item 22 / SP-20: catch drift between GROUP_TOOLS (core_tool_definitions.py,
            # what the model is TOLD it can call) and _build_dispatch() (tool_handlers.py,
            # what can ACTUALLY be called) at startup rather than silently at use-time.
            _dispatch_tools = frozenset(self._dispatch.keys())
            if _dispatch_tools != GROUP_TOOLS_UNION:
                _missing_handler = GROUP_TOOLS_UNION - _dispatch_tools   # advertised, not dispatchable
                _missing_group   = _dispatch_tools - GROUP_TOOLS_UNION   # dispatchable, never advertised
                _msg = "GROUP_TOOLS / _build_dispatch() drift detected."
                if _missing_handler:
                    _msg += f" In GROUP_TOOLS but no handler: {sorted(_missing_handler)}."
                if _missing_group:
                    _msg += f" Has a handler but not in any GROUP_TOOLS group: {sorted(_missing_group)}."
                raise AssertionError(_msg)

            # /tool toggles groups. Defaults to web + bash on startup.
            # bash is only included if Docker is confirmed available — same guard
            # the /tool dialog applies. start.py passes docker_available after
            # _setup_sandbox() runs; falls back to web-only if not provided.
            _default_groups = {"web", "bash"} if docker_available else {"web"}
            self._active_groups: set = _default_groups

            # ── Lazy pre-turn scratch backup (item 4 / SP-1) ─────────────────────
            # Previously start.py ran a synchronous shutil.copytree() of the whole
            # scratch/ dir before EVERY tool-mode turn, even pure-text turns that
            # never touch a file. Now the backup is created lazily, on a background
            # thread, the first time a tool is actually called this turn (see
            # _start_scratch_backup() and _call_tool() in tool_handlers.py).
            # start.py resets _backup_created to False at the start of each turn.
            self._backup_created = False   # True once a backup attempt has started this turn
            self._backup_success = False   # True once the background copy finishes OK
            self._backup_thread  = None    # Thread handle so callers can join() before rollback

            # Cache for _scan_uploads_folder() (item 11 / SP-6): (mtime, pdf_active, result).
            # Invalidated on UPLOADS_FOLDER mtime change, or explicitly after ingest_pdf.
            self._uploads_cache = None

        except Exception as e:
            console.print(f"🚨 Init failed: {e}")
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # GOOGLE KEY ROTATION
    # ══════════════════════════════════════════════════════════════════════════

    def _rotate_google_key(self) -> None:
        """Advance to the next key in the pool and rebuild google_client.

        Called by _make_request_google (LLMBackendsMixin) every time a 429 /
        RESOURCE_EXHAUSTED error is received.  The index wraps with modulo so
        after the last key we cycle back to key[0] (second round of attempts).
        No-op if the pool has only one key (or is empty).
        """
        pool = getattr(self, "_google_key_pool", [])
        if len(pool) < 2:
            return  # nothing to rotate to
        self._google_key_index = (self._google_key_index + 1) % len(pool)
        new_key = pool[self._google_key_index]
        self.google_client = genai.Client(api_key=new_key)
        console.print(
            f"🔄 [yellow]Google 429 — rotating to key "
            f"#{self._google_key_index + 1}/{len(pool)}[/yellow]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TOOL GROUP TOGGLES
    # ══════════════════════════════════════════════════════════════════════════


    @property
    def _base_system(self) -> str:
        """Build system prompt dynamically from active groups.
        PSEUDO_TOOL_FORMAT preamble is injected when any group is active.
        Each group then appends its own *_TOOLS_PROMPT block."""
        prompt = _BASE_SYSTEM_PROMPT
        if self._active_groups:
            prompt += "\n\n" + PSEUDO_TOOL_FORMAT
            # search_history (HISTORY_TOOLS_PROMPT) is baked only into
            # GROUP_PROMPTS["files"], matching GROUP_TOOLS["files"] — the tool
            # is advertised, and executable, only when 'files' is active.
            # Bug #52 fix: track whether the date block has already been
            # injected so 'web' + 'research' active together don't produce
            # two identical REFERENCE DATE blocks.
            _date_injected = False
            for group in ("web", "files", "pdf", "bash", "research"):  # deterministic order
                if group not in self._active_groups:
                    continue
                prompt += "\n\n" + GROUP_PROMPTS[group]
                if group in ("web", "research") and not _date_injected:
                    # Computed fresh every call (not cached) — a long-running
                    # session needs today's actual date, not the date the
                    # process started. Only injected when 'web' or 'research'
                    # is active; injected at most once per call.
                    prompt += "\n\n" + web_date_context()
                    _date_injected = True
        return prompt

    @property
    def _review_system(self) -> str:
        """Build the reviewer system prompt dynamically.
        Uses _REVIEW_SYSTEM_PROMPT as the base instead of _BASE_SYSTEM_PROMPT
        but appends the exact same tool-format blocks as _base_system so the
        reviewer can call tools if any groups are active."""
        prompt = _REVIEW_SYSTEM_PROMPT
        if self._active_groups:
            prompt += "\n\n" + PSEUDO_TOOL_FORMAT
            # Bug #52 fix: inject date at most once.
            _date_injected = False
            for group in ("web", "files", "pdf", "bash", "research"):  # deterministic order
                if group not in self._active_groups:
                    continue
                prompt += "\n\n" + GROUP_PROMPTS[group]
                if group in ("web", "research") and not _date_injected:
                    prompt += "\n\n" + web_date_context()
                    _date_injected = True
        return prompt


    def _start_scratch_backup(self) -> None:
        """
        Lazily create the rollback backup of scratch/ (item 4 / SP-1).

        Triggered from _call_tool() on the FIRST tool call of a turn — not
        eagerly at turn entry. Most tool-mode turns never modify a file (e.g.
        a pure web search or a read-only view_lines), so this often avoids
        the copytree entirely. Runs on a daemon background thread so it never
        blocks the tool loop; the Ctrl+C rollback path joins the thread before
        deciding whether a backup actually exists to roll back to.
        """
        if self._backup_created:
            return
        self._backup_created = True  # set immediately so only one thread ever starts

        def _do_backup():
            backup_dir = config.SCRATCH_DIR + "_backup"
            backup_registry = config.WORKSPACE_REGISTRY_FILE + "_backup"

            def _ignore(dirpath, names):
                # Skip the same heavy/ephemeral dirs the turn-archive zip
                # excludes (turn_state_manager.py's _zip_scratch) — this backup
                # exists purely for Ctrl+C rollback of model-made edits, not to
                # snapshot node_modules/venv/.git, which don't need reverting.
                ignored = set()
                for name in names:
                    if any(name.startswith(p) for p in EXCL_PREFIXES):
                        ignored.add(name)
                        continue
                    if os.path.isdir(os.path.join(dirpath, name)) and (
                        name in EXCL_DIRS or any(name.endswith(s) for s in EXCL_SUFFIXES)
                    ):
                        ignored.add(name)
                return ignored

            try:
                if os.path.exists(backup_dir):
                    robust_rmtree(backup_dir)
                shutil.copytree(config.SCRATCH_DIR, backup_dir, ignore=_ignore)
                if os.path.exists(config.WORKSPACE_REGISTRY_FILE):
                    shutil.copy(config.WORKSPACE_REGISTRY_FILE, backup_registry)
                self._backup_success = True
            except Exception as e:
                console.print(f"[yellow]⚠️ could not create scratch backup: {e}[/yellow]")

        self._backup_thread = threading.Thread(target=_do_backup, daemon=True)
        self._backup_thread.start()

    def set_tool_groups(self, groups: set) -> None:
        """Replace active groups with a new set. Caller is responsible for validation."""
        self._active_groups = set(groups)

    def toggle_tool_group(self) -> bool:
        """Legacy binary toggle: activates ALL groups or clears all.
        Used by tests and any code that hasn't been updated yet.
        Returns True if now active, False if now inactive."""
        if self._active_groups:
            self._active_groups.clear()
            return False
        else:
            self._active_groups = {"web", "files", "pdf", "bash"}
            return True

    def tools_status(self) -> str:
        """Return a one-line status string."""
        if self._active_groups:
            parts = []
            labels = {"web": "web", "files": "files", "pdf": "pdf", "bash": "bash", "research": "research"}
            for g in ("web", "files", "pdf", "bash", "research"):
                if g in self._active_groups:
                    parts.append(f"[green]{labels[g]}[/green]")
                else:
                    parts.append(f"[dim]{labels[g]}[/dim]")
            state = "tools: " + "  ".join(parts)
        else:
            state = "[dim]tools ✗  (clean chat)[/dim]"
        backend = f"[cyan]{config.ACTIVE_BACKEND}[/cyan] ({config.get_active_model().split('/')[-1]})"
        return f"  {state}  backend: {backend}"

    def switch_backend(self, backend: str) -> str:
        """
        Switch between 'google' and 'nim' backends.
        Updates config.ACTIVE_BACKEND, config.THINKING_TYPE, config.SUPPORTS_VISION.
        Returns a status string for display.
        """
        backend = backend.lower().strip()
        if backend in ("cf", "cloudflare"):
            backend = "cloudflare"
        if backend in ("tr", "tokenrouter", "token_router"):
            backend = "tokenrouter"
        if backend not in ("google", "nim", "cloudflare", "local", "tokenrouter"):
            return f"[red]Unknown backend '{backend}'. Use /google, /nim, /cf, /local, or /tr.[/red]"

        if backend == config.ACTIVE_BACKEND:
            return f"[dim]Already on {backend} backend.[/dim]"

        if backend == "cloudflare":
            # Bug 7: refuse the switch with a clear message instead of flipping
            # ACTIVE_BACKEND and letting every subsequent request 404 on a
            # malformed URL (".../accounts//ai/v1").
            if not self.cf_configured:
                return (
                    "[red]Cannot switch to Cloudflare: CLOUDFLARE_ACCOUNT_ID is not "
                    "set in .env. Get it from the Cloudflare dashboard URL bar and "
                    "set CLOUDFLARE_ACCOUNT_ID=... then restart.[/red]"
                )
            config.ACTIVE_BACKEND   = "cloudflare"
            config.THINKING_TYPE    = config.CLOUDFLARE_THINKING_TYPE
            config.SUPPORTS_VISION  = config.CLOUDFLARE_VISION
            return (
                f"[green]Switched to Cloudflare backend[/green]\n"
                f"  model:   {config.CLOUDFLARE_MODEL}\n"
                f"  thinking: {config.THINKING_TYPE}\n"
                f"  vision:   {config.SUPPORTS_VISION}"
            )

        if backend == "local":
            config.ACTIVE_BACKEND   = "local"
            config.THINKING_TYPE    = config.LOCAL_THINKING_TYPE
            config.SUPPORTS_VISION  = config.LOCAL_VISION

            # Turn-1 cache warm-up only — fires once here at switch time, not
            # on every turn. Runs on a background thread so /local returns to
            # the prompt immediately instead of blocking on prefill. Harmless
            # no-op if llama-server isn't running yet — the real request will
            # just surface a clear connection error when the user sends one.
            threading.Thread(
                target=self.warm_local_system_prompt,
                args=(self._base_system,),
                daemon=True,
            ).start()

            return (
                f"[green]Switched to local backend[/green]\n"
                f"  endpoint: {config.LOCAL_BASE_URL}\n"
                f"  model:    {config.LOCAL_MODEL}\n"
                f"  thinking: {config.THINKING_TYPE}\n"
                f"  vision:   {config.SUPPORTS_VISION}\n"
                f"  [dim]warming system-prompt cache in background...[/dim]"
            )

        if backend == "tokenrouter":
            if not self.tr_api_key:
                return (
                    "[red]Cannot switch to Token Router: TOKEN_ROUTER_API_KEY_FILE is not "
                    "set in .env, or the key file is missing/empty.[/red]"
                )
            config.ACTIVE_BACKEND   = "tokenrouter"
            config.THINKING_TYPE    = config.TOKEN_ROUTER_THINKING_TYPE
            config.SUPPORTS_VISION  = config.TOKEN_ROUTER_VISION
            return (
                f"[green]Switched to Token Router backend[/green]\n"
                f"  model:    {config.TOKEN_ROUTER_MODEL}\n"
                f"  thinking: {config.THINKING_TYPE}\n"
                f"  vision:   {config.SUPPORTS_VISION}"
            )

        if backend == "google":
            config.ACTIVE_BACKEND   = "google"
            config.THINKING_TYPE    = config.GOOGLE_THINKING_TYPE
            config.SUPPORTS_VISION  = config.GOOGLE_VISION
        else:  # nim
            config.ACTIVE_BACKEND   = "nim"
            config.THINKING_TYPE    = config.NIM_THINKING_TYPE
            config.SUPPORTS_VISION  = config.NIM_VISION

        if backend == "nim":
            return (
                f"[green]Switched to nim backend[/green]\n"
                f"  model    : {config.NIM_TEXT_MODEL.split('/')[-1]}\n"
                f"  thinking : {config.THINKING_TYPE}\n"
                f"  vision   : {config.SUPPORTS_VISION}"
            )
        return (
            f"[green]Switched to {backend} backend[/green]\n"
            f"  model:   {config.get_active_model()}\n"
            f"  thinking: {config.THINKING_TYPE}\n"
            f"  vision:   {config.SUPPORTS_VISION}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATION — STATEFUL (main entry point from start.py)
    # ══════════════════════════════════════════════════════════════════════════


    def generate_with_history(self, prompt, web_agent, token_counter=None, images=None):
        """
        Main entry point for a new user turn.
        Constructs the full system prompt (including tool instructions and active workspace state),
        appends the chat history (truncating if necessary based on token limits),
        and kicks off the generation loop.
        """
        # Capture the summary from the PREVIOUS turn before resetting the attribute.
        # Without this, self.last_tool_summary is wiped and reads as ""
        # when the context block tries to inject it, making [LAST TURN
        # ACTIONS] permanently empty regardless of what tools were called last turn.
        _prior_tool_summary = self.last_tool_summary
        self.last_tool_summary = ""
        # Bug 14: cancelled/rolled-back turns are captured separately (see
        # _save_partial_history in start.py) so they can be labelled
        # distinctly here instead of being injected as "[LAST TURN ACTIONS]",
        # which would falsely imply those actions were committed.
        _prior_cancelled_summary = self.last_cancelled_summary
        self.last_cancelled_summary = ""
        self._images_shown_this_turn = []  # reset per-turn image log
        self.last_think_trace = ""          # reset per-turn reasoning trace
        console.print(f"\n[cyan] ✦ [/cyan][dim]{config.ACTIVE_BACKEND} ({config.THINKING_TYPE} thinking)...[/dim]")

        if self._active_groups:
            # Start reconcile in background immediately — before history slicing
            # or message assembly, so those operations overlap with the I/O.
            # We join (wait) only right before the workspace summary is needed.
            _reconcile_thread = threading.Thread(
                target=self.workspace_tracker.reconcile_workspace,
                daemon=True, name="reconcile"
            )
            _reconcile_thread.start()

        messages = [{"role": "system", "content": self._base_system}]

        # FIX #8: if the tail-slice starts on an assistant message (e.g. when
        # len(chat_history) % MAX_HISTORY_TURNS produces an odd-indexed boundary),
        # drop that orphaned entry so the model always receives well-formed history.
        sliced = self.chat_history[-config.MAX_HISTORY_TURNS:]
        if sliced and sliced[0].get("role") != "user":
            sliced = sliced[1:]

        for msg in sliced:
            if msg['role'] in ('user', 'assistant'):
                messages.append(msg)

        if self._active_groups:
            uploads_summary = self._scan_uploads_folder(pdf_active="pdf" in self._active_groups)
            # Join reconcile thread before building workspace summary
            _reconcile_thread.join(timeout=2.0)
            workspace = self.workspace_tracker.get_workspace_summary()
            prior = (
                f"[LAST TURN ACTIONS]\n{_prior_tool_summary}\n\n"
                if _prior_tool_summary else ""
            )
            # Bug 14: distinctly-labelled — these actions were attempted then
            # rolled back, not committed. Kept separate from `prior` above so
            # the model still benefits from knowing what was already tried
            # (e.g. to avoid repeating a failing approach) without being told
            # it actually happened.
            cancelled_prior = (
                f"[PREVIOUS TURN WAS CANCELLED AND ROLLED BACK — the following "
                f"actions were attempted but NOT applied to the workspace:]\n"
                f"{_prior_cancelled_summary}\n\n"
                if _prior_cancelled_summary else ""
            )
            _ws_empty  = workspace.startswith("[WORKSPACE] Empty")
            _upl_empty = uploads_summary == "[UPLOADS FOLDER EMPTY]"
            _show_ws  = self._ctx_enabled and not _ws_empty
            _show_upl = self._ctx_enabled and not _upl_empty
            if prior or cancelled_prior or _show_ws or _show_upl:
                context_block = (
                    "[SYSTEM CONTEXT — injected by runtime, not the user]\n"
                    + prior
                    + cancelled_prior
                    + ("" if not _show_ws  else f"[WORKSPACE]\n{workspace}\n\n")
                    + ("" if not _show_upl else f"[UPLOADS FOLDER]\n{uploads_summary}")
                    + f"\n[END SYSTEM CONTEXT]\n\n---\n\n"
                )
            else:
                context_block = ""
        else:
            context_block = ""


        # Build user message — multimodal when vision is supported and images provided,
        # plain string otherwise (images were pre-processed to OCR text by start.py).
        if images and config.SUPPORTS_VISION:
            content = [{"type": "text", "text": context_block + prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["src"]},
                })
            # Remind the model that images are not persisted across turns.
            # Injected after the images so the model processes visuals first.
            content.append({
                "type": "text",
                "text": "[Note: Images attached here are not saved in chat history — future turns will not have access to them.]"
            })
            messages.append({"role": "user", "content": content})

        else:
            messages.append({"role": "user", "content": context_block + prompt})

        raw_query = prompt

        return self._generate_with_tools(
            messages, web_agent,
            config.EMPEROR_STATEFUL_TEMP, token_counter,
            user_query=raw_query
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SELF-REVIEW
    # ══════════════════════════════════════════════════════════════════════════

    def generate_review(self, review_prompt: str, web_agent, token_counter=None) -> str:
        """
        Self-review mode: call the model with the same conversation history and
        workspace context as a normal turn, but with _REVIEW_SYSTEM_PROMPT so
        it critiques its own latest response instead of answering a new question.

        State-isolated: last_tool_summary, last_cancelled_summary, backup flags,
        _current_tool_call_log, _pending_narratives, and _partial_messages are
        all saved and restored in a finally block so the next real turn sees
        exactly the same state as before this call. Never writes to TurnStateManager.
        """
        # ── State preservation ──────────────────────────────────────────────
        # Read (don't consume) the prior-turn summaries so the reviewer sees
        # the same [LAST TURN ACTIONS] context a real turn would see — but
        # we restore them afterward so the next real turn still gets them.
        _saved_tool_summary      = self.last_tool_summary
        _saved_cancelled_summary = self.last_cancelled_summary
        _saved_trace             = self.last_think_trace   # main model's reasoning; captured before
                                                           # Pass 1 overwrites last_think_trace

        # Pre-set _backup_created so _call_tool never fires the lazy backup.
        # If we let the backup run, the orphaned scratch_backup/ dir would
        # interfere with the next real turn's Ctrl+C rollback path.
        _saved_backup_created = self._backup_created
        _saved_backup_success = self._backup_success
        _saved_backup_thread  = self._backup_thread
        self._backup_created  = True
        self._backup_success  = False
        self._backup_thread   = None

        console.print(f"\n[cyan] ✦ [/cyan][dim]{config.ACTIVE_BACKEND} ({config.THINKING_TYPE} thinking)...[/dim]")

        try:
            # ── Build messages ──────────────────────────────────────────────
            if self._active_groups:
                _reconcile_thread = threading.Thread(
                    target=self.workspace_tracker.reconcile_workspace,
                    daemon=True, name="reconcile-review"
                )
                _reconcile_thread.start()

            messages = [{"role": "system", "content": self._review_system}]

            # Identical slice + orphan-drop as generate_with_history
            sliced = self.chat_history[-config.MAX_HISTORY_TURNS:]
            if sliced and sliced[0].get("role") != "user":
                sliced = sliced[1:]
            for msg in sliced:
                if msg["role"] in ("user", "assistant"):
                    messages.append(msg)

            # ── Context block — identical to generate_with_history ──────────
            if self._active_groups:
                uploads_summary = self._scan_uploads_folder(pdf_active="pdf" in self._active_groups)
                _reconcile_thread.join(timeout=2.0)
                workspace = self.workspace_tracker.get_workspace_summary()
                prior = (
                    f"[LAST TURN ACTIONS]\n{_saved_tool_summary}\n\n"
                    if _saved_tool_summary else ""
                )
                cancelled_prior = (
                    f"[PREVIOUS TURN WAS CANCELLED AND ROLLED BACK — the following "
                    f"actions were attempted but NOT applied to the workspace:]\n"
                    f"{_saved_cancelled_summary}\n\n"
                    if _saved_cancelled_summary else ""
                )
                _ws_empty  = workspace.startswith("[WORKSPACE] Empty")
                _upl_empty = uploads_summary == "[UPLOADS FOLDER EMPTY]"
                _show_ws  = self._ctx_enabled and not _ws_empty
                _show_upl = self._ctx_enabled and not _upl_empty
                if prior or cancelled_prior or _show_ws or _show_upl:
                    context_block = (
                        "[SYSTEM CONTEXT — injected by runtime, not the user]\n"
                        + prior
                        + cancelled_prior
                        + ("" if not _show_ws  else f"[WORKSPACE]\n{workspace}\n\n")
                        + ("" if not _show_upl else f"[UPLOADS FOLDER]\n{uploads_summary}")
                        + f"\n[END SYSTEM CONTEXT]\n\n---\n\n"
                    )
                else:
                    context_block = ""
            else:
                context_block = ""

            messages.append({"role": "user", "content": context_block + review_prompt})

            pass1_response = self._generate_with_tools(
                messages, web_agent,
                config.EMPEROR_REVIEW_TEMP, token_counter,
                user_query=review_prompt,
            )

            # ── Pass 2: trace-informed follow-up ─────────────────────────────
            # Feed the main model's reasoning trace back so the reviewer can
            # spot things it explored in thinking but dropped from the answer —
            # omissions that are invisible from the final response alone.
            if _saved_trace:
                # Pass 1 final text isn't printed by _generate_with_tools — cmd_review
                # would normally handle it, but we're returning Pass 2 instead.
                # Print Pass 1 here so the user sees both responses.
                console.print()
                console.rule(style=RULE_STYLE)
                print_smart_response(pass1_response)
                console.rule(style=RULE_STYLE)
                console.print()

                console.rule("reviewing with reasoning trace", style="dim cyan")
                _pass2_prompt = (
                    "[SYSTEM: Here is the internal reasoning the model went through before "
                    "writing that response \u2014 thinking before and after each power use.]\n\n"
                    f"[POWERS USED]\n{_saved_tool_summary or 'None'}\n\n"
                    f"[REASONING TRACE]\n{_saved_trace}\n\n"
                    "---\n\n"
                    "Review the response. Use the reasoning trace above to catch things "
                    "invisible from the final answer alone: something the model explored in "
                    "thinking but dropped, or a wrong early commitment it couldn't recover from."
                )
                # Fresh call — replace the pass-1 user message rather than
                # appending a continuation, so pass 2 has no memory of pass 1.
                messages2 = messages[:-1] + [{"role": "user", "content": context_block + _pass2_prompt}]
                return self._generate_with_tools(
                    messages2, web_agent,
                    config.EMPEROR_REVIEW_TEMP, token_counter,
                    user_query=_pass2_prompt,
                )
            return pass1_response

        finally:
            # ── State restoration ───────────────────────────────────────────
            # Always runs — success, cancellation (Ctrl+C), or error.
            self.last_tool_summary      = _saved_tool_summary
            self.last_cancelled_summary = _saved_cancelled_summary
            self._backup_created        = _saved_backup_created
            self._backup_success        = _saved_backup_success
            self._backup_thread         = _saved_backup_thread
            self._current_tool_call_log = []
            self._pending_narratives    = []
            self._partial_messages      = None
            self._images_shown_this_turn = []   # review is not a real turn; discard any images it showed
            self.last_think_trace = ""           # discard any reviewer thinking; main model trace already consumed

    # ══════════════════════════════════════════════════════════════════════════
    # INTERRUPT RESUME
    # ══════════════════════════════════════════════════════════════════════════

    def resume_from_interrupt(self, guidance: str, web_agent, token_counter=None,
                               thinking_paste: str = "") -> str:
        """
        Continue a turn that was interrupted by Ctrl+C.

        Picks up the mid-turn messages list saved by _generate_with_tools
        (which snapshots `self._partial_messages` before every blocking API call).
        It appends the guidance to the existing user message's content block 
        (safely handling text blocks within multimodal lists) to prevent empty 
        message crashes, then re-enters the tool loop from that exact point.

        Key differences from calling generate_with_history() again:
          - Does NOT rebuild messages from chat history — keeps all tool calls
            that already executed this turn.
          - Falls back to generate_with_history() if _partial_messages is None
            (e.g. interrupt happened before the first API call).

        Args:
            thinking_paste: Optional text the user pasted describing the model's
                own reasoning/thinking trace right before the interrupt (e.g.
                copied from a UI panel that doesn't otherwise reach the model).
                When empty, the injected block is identical to before this
                feature existed.
        """
        if not self._partial_messages:
            # Interrupted before the first tool call — nothing to resume from.
            # Treat as a fresh turn with the guidance as the prompt.
            console.print("[dim]No mid-turn state to resume — starting fresh.[/dim]")
            return self.generate_with_history(guidance, web_agent, token_counter=token_counter)

        messages = self._partial_messages

        # Inject guidance (and, if provided, the pasted thinking trace) into the
        # live message thread so the model sees: everything it already did +
        # what it was thinking + the course correction.
        thinking_block = (
            f"[SYSTEM: Your reasoning trace before the interrupt]\n{thinking_paste}\n\n"
            if thinking_paste else ""
        )
        guidance_text = (
            f"\n\n{thinking_block}"
            f"[SYSTEM: Turn interrupted. User guidance follows]\n{guidance}\n\n"
            f"[SYSTEM: Resume from where you left off, incorporating the above guidance. Do not restart from scratch.]"
        )

        if messages and messages[-1].get("role") == "user":
            if isinstance(messages[-1]["content"], list):
                # Multimodal message: first element is always the text block
                messages[-1]["content"][0]["text"] += guidance_text
            else:
                messages[-1]["content"] += guidance_text
        else:
            messages.append({
                "role": "user",
                "content": guidance_text.strip()
            })

        # Update the snapshot so a subsequent interrupt resumes from here safely
        self._partial_messages = list(messages)

        console.print("[dim]Resuming interrupted turn with your guidance...[/dim]")

        return self._generate_with_tools(
            messages, web_agent,
            config.EMPEROR_STATEFUL_TEMP, token_counter,
            user_query=guidance,
            resume_mode=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PSEUDO-TOOL PARSER
    # ══════════════════════════════════════════════════════════════════════════

    # Every tool name the runtime knows about. Used to scope the regex scan so
    # arbitrary XML in user files never gets misidentified as a tool call.
    def _get_active_known_tools(self) -> set:
        """Return active tool names by unioning all active group tool-sets.
        Empty set when no group is active — stale history tags never execute."""
        if not self._active_groups:
            return set()
        active = set()
        for group in self._active_groups:
            active |= GROUP_TOOLS.get(group, set())
        return active

    def _parse_pseudo_tools(self, content: str) -> list:
        """
        Scan model output for pseudo-tool XML tags and return ordered list of
        {"fn": str, "args": dict, "pos": int} dicts ready for _call_tool().

        Handles three tag formats:
          1. Self-closing:  <tool_name attr="val"/>
          2. Simple block:  <tool_name>inner text</tool_name>
          3. Structured:    <tool_name><child>val</child>...</tool_name>

        Intentionally regex-based (not xml.etree) so that raw code or file
        content inside tags never causes a parse failure.
        """
        calls = []

        for tool_name in self._get_active_known_tools():
            # ── Format 1: self-closing  <tool_name attr="val" attr2="val2"/>
            for m in re.finditer(
                rf'<{tool_name}((?:\s+[^>]*?)?)/>',
                content, re.DOTALL
            ):
                args = self._parse_xml_attrs(m.group(1))
                calls.append({"fn": tool_name, "args": args, "pos": m.start()})

            # ── Formats 2 & 3: block  <tool_name ...>...</tool_name>
            # Use a stack-based parser to gracefully handle nested identical tags
            open_pattern = rf'<{tool_name}((?:\s[^>]*)??)(?<!/)>'
            close_pattern = rf'</{tool_name}>'
            
            tokens = []
            for m in re.finditer(open_pattern, content):
                tokens.append(('open', m.start(), m.end(), m.group(1)))
            for m in re.finditer(close_pattern, content):
                tokens.append(('close', m.start(), m.end(), None))
            tokens.sort(key=lambda x: x[1])

            stack = []
            for t_type, start, end, attr_str in tokens:
                if t_type == 'open':
                    stack.append((start, end, attr_str))
                elif t_type == 'close':
                    if stack:
                        o_start, o_end, o_attr_str = stack.pop()
                        if not stack: # outer-most tag completed
                            inner = content[o_end:start]
                            args = {**self._parse_xml_attrs(o_attr_str),
                                    **self._parse_inner_args(tool_name, inner)}
                            calls.append({"fn": tool_name, "args": args, "pos": o_start})

        # Execute in the order they appear in the response
        calls.sort(key=lambda c: c["pos"])
        return calls

    @staticmethod
    def _parse_xml_attrs(attr_str: str) -> dict:
        """Extract key="value" or key='value' pairs from an XML attribute string."""
        return {
            m.group(1): m.group(3)
            for m in re.finditer(r'(\w+)\s*=\s*([\'"])(.*?)\2', attr_str or "")
        }

    @staticmethod
    def _parse_inner_args(tool_name: str, inner: str) -> dict:
        """
        Extract args from the inner body of a block tag.

        Structured tools have named child elements:
            <str_replace><file>x</file><old>...</old><new>...</new></str_replace>

        Simple tools treat the whole inner text as their primary param:
            <quick_search>my query</quick_search>

        The mapping of tool → primary param key lives here so handlers stay
        untouched (they all do args.get("key")).
        """
        # Tools that carry named child elements — list every child tag expected
        STRUCTURED = {
            "str_replace":   ["file", "old_str", "new_str", "count"],
            "view_lines":    ["file", "start", "end", "context"],
            "search_in_file":["file", "pattern", "regex", "context_lines", "max_results"],
            "workspace_search": ["query", "file_filter", "semantic"],
            "doc_search":    ["query", "top_k"],
            # ingest_pdf is in SIMPLE_PARAM below (simple inner-text format)
            "search_history":             ["query", "top_k"],
            "search_semantic_scholar":    ["query", "limit", "year"],
            # Bug #45 fix: url_search is self-closing in the documented/prompted
            # format (<url_search url="..." context="..."/>) but if a model emits
            # it as a block tag with named children, SIMPLE_PARAM would dump the
            # entire raw inner XML into 'url'. Moving to STRUCTURED handles both.
            "url_search":    ["url", "context"],
        }

        # Primary param for simple (inner-text) tools
        SIMPLE_PARAM = {
            "quick_search": "query",
            # url_search moved to STRUCTURED (Bug #45 fix above)
            "bash":         "command",   # <bash>command here</bash>
            "ingest_pdf":   "filename",  # <ingest_pdf>paper.pdf</ingest_pdf>
            "ingest_chat":  "filename",  # <ingest_chat>gemini_chat.txt</ingest_chat>
            "ingest_text":  "filename",  # <ingest_text>notes.txt</ingest_text>
            "show_image":   "file",      # <show_image>chart.png</show_image>
        }

        if tool_name in STRUCTURED:
            result = {}
            for key in STRUCTURED[tool_name]:
                m = re.search(rf'<{key}(?:\s+[^>]*)?>(.*?)</{key}\s*>', inner, re.DOTALL)
                if m:
                    val = m.group(1)
                    if val.startswith("\n"):
                        val = val[1:]
                    if val.endswith("\n"):
                        val = val[:-1]
                    result[key] = val
            return result

        if tool_name in SIMPLE_PARAM:
            return {SIMPLE_PARAM[tool_name]: inner.strip()}

        # Generic fallback — inner text → "query"
        stripped = inner.strip()
        return {"query": stripped} if stripped else {}

    # ══════════════════════════════════════════════════════════════════════════
    # INNER GENERATION LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_kimi_native_tools(self, content: str, offset: int = 0) -> list:
        """
        Parse Kimi's native tool-call format emitted in the content field:

            <|tool_calls_section_begin|>
            <|tool_call_begin|> functions.quick_search:0
            <|tool_call_argument_begin|> {"query": "..."} <|tool_call_end|>
            <|tool_calls_section_end|>

        Maps function names → our pseudo-tool names and extracts JSON args.
        Returns list of {"fn", "args", "pos"} dicts compatible with _call_tool.
        """
        import json as _json

        calls = []
        # Match each <|tool_call_begin|> ... <|tool_call_end|> block
        for m in re.finditer(
            r'<\|tool_call_begin\|>\s*functions\.([\w]+)(?::[\d]+)?\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>',
            content, re.DOTALL
        ):
            # Kimi's function names already match our pseudo-tool names 1:1 —
            # no mapping needed, just filter against known tools below.
            fn_name  = m.group(1)
            arg_text = m.group(2).strip()

            if fn_name not in self._get_active_known_tools():
                continue

            try:
                args = _json.loads(arg_text)
                if not isinstance(args, dict):
                    args = {"query": str(args)}
            except Exception:
                # Fallback: treat raw text as primary arg
                args = {"query": arg_text}

            calls.append({"fn": fn_name, "args": args, "pos": m.start() + offset})

        if calls:
            console.print(f"[dim]Kimi native tool calls parsed: {[c['fn'] for c in calls]}[/dim]")

        return calls

    def _load_alien_detector(self):
        """
        Lazily loads the alien-format classifier from alien_format_clf.joblib.
        Returns the detector on success, None if the file is missing or any
        dependency (joblib, sklearn) is unavailable.
        """
        # Already loaded successfully
        if self._alien_detector is not None:
            return self._alien_detector
        # Previous load attempt failed — don't retry every call
        if getattr(self, "_alien_detector_failed", False):
            return None
        model_path = os.path.join(os.path.dirname(__file__), "alien_format_clf.joblib")
        if not os.path.exists(model_path):
            console.print("[dim]⚠️  alien_format_clf.joblib not found — alien detection disabled[/dim]")
            self._alien_detector_failed = True
            return None
        try:
            from train_alien_format_classifier import AlienFormatDetector
            self._alien_detector = AlienFormatDetector.load(model_path)
            console.print("[dim]✓ Alien format detector loaded[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Alien detector load failed: {e}[/yellow]")
            self._alien_detector_failed = True
        return self._alien_detector

    def _generate_with_tools(self, messages, web_agent, temp, token_counter, user_query: str = "", resume_mode: bool = False):

        """
        The core generation and execution loop for the agent.
        
        Handles API calls, tracks token counts, parses pseudo-XML tool tags from 
        the response, and executes tools sequentially. It snapshots the message 
        state to `self._partial_messages` before every blocking API call, ensuring
        that `resume_from_interrupt` can safely recover from early terminations.
        """
        tool_call_count = 0
        # Accumulates narrative text (model prose alongside tool calls) so the
        # full turn — explanation + tools + final answer — is preserved in
        # permanent history. Stored on self so start.py can read it after the
        # turn ends without re-printing it (which would cause double output).
        # Reset here for new turns; preserved across resume to keep pre-interrupt
        # narratives for the combined history entry.
        _turn_thinking: list = []   # collects per-iteration thinking traces for last_think_trace
        if not resume_mode:
            self._pending_narratives = []
            self._current_tool_call_log = []
            self._alien_nudge_sent = False   # allow one nudge per fresh turn

        for loop in range(45):
            # Snapshot BEFORE the blocking API call so that if Ctrl+C fires
            # during the request OR during any subsequent tool execution,
            # resume_from_interrupt has the correct messages state for this iteration.
            self._partial_messages = list(messages)
            response_data = self._make_request(messages, temp)

            if response_data.get("error"):
                self._partial_messages = None
                self.last_tool_summary = self.summarizer.summarize_turn(self._current_tool_call_log)
                return f"[API ERROR] {response_data['error']}"

            if "usage" in response_data and response_data["usage"] > 0:
                self.last_token_count = response_data["usage"]

            thinking_content = response_data.get("thinking")
            if thinking_content and config.THINKING_TYPE != "native":
                console.print(f"[dim cyan]Thinking: {thinking_content[:150]}...[/dim cyan]")

            raw_content  = response_data.get("content", "")
            think_extracted, clean_content = extract_thinking_tags(raw_content)
            if think_extracted:
                console.print(f"[dim cyan]{think_extracted[:550]}...[/dim cyan]")

            # Collect thinking for the trace-informed second-pass review.
            _cur_thinking = thinking_content or think_extracted or ""
            if _cur_thinking:
                _turn_thinking.append(_cur_thinking)



            # ── Parse pseudo-tool tags from model output ───────────────────
            # Scan clean_content ONLY — the model's actual visible answer,
            # not its thinking/reasoning text. Tags inside <think> blocks or
            # a native reasoning field are NOT executed.
            #
            # Previously this also scanned raw_content (pre-strip) plus the
            # native reasoning field, on the theory that a model might
            # "decide" to call a tool purely while thinking and never repeat
            # it in the visible answer. In practice this backfired: a
            # regex tag-scanner can't tell "I am calling this tool" apart
            # from "here's an example of what that tool call looks like" or
            # "I should have called X" inside natural-language reasoning —
            # all three produce identical-looking tags. The result was
            # illustrative/hypothetical tags in reasoning getting silently
            # executed as real actions (including against mutating tools
            # like str_replace/bash, not just read-only ones). Requiring the
            # tag to appear in the visible answer is the standard contract
            # anyway — PSEUDO_TOOL_FORMAT tells the model to write tags "in
            # your response," meaning the answer it gives you, not private
            # scratch reasoning.
            pseudo_calls = self._parse_pseudo_tools(clean_content or raw_content or "")


            # Kimi also emits its own native <|tool_call_begin|> format in
            # the content field alongside (or instead of) our XML tags.
            # Parse and merge those so nothing gets missed. Scans
            # clean_content for the same reason as above — a native-format
            # tag inside reasoning is just as executable-by-accident as an
            # XML one would be. Both parsers now share clean_content's
            # coordinate space, so no offset is needed to keep positions
            # comparable when interleaving/sorting the two call lists.
            if config.THINKING_TYPE == "native":
                native_calls = self._parse_kimi_native_tools(clean_content or raw_content or "")
                if native_calls:
                    # Merge — avoid duplicating calls already found via XML.
                    # Use json.dumps with sort_keys=True for the dedup key instead of str(args),
                    # because dict stringification is sensitive to key insertion order, which
                    # can differ between our XML parser and Kimi's native JSON output.
                    def _dedup_key(fn: str, call_args: dict) -> tuple:
                        try:
                            canon = json.dumps(call_args, sort_keys=True, default=str)
                        except Exception:
                            canon = str(call_args)
                        return (fn, canon)

                    existing_fns = {_dedup_key(c["fn"], c["args"]) for c in pseudo_calls}
                    for nc in native_calls:
                        key = _dedup_key(nc["fn"], nc["args"])
                        if key not in existing_fns:
                            pseudo_calls.append(nc)
                            # FIX #10: update the set so duplicate entries within
                            # native_calls itself are also caught on subsequent
                            # iterations (not just duplicates vs pseudo_calls).
                            existing_fns.add(key)
                    pseudo_calls.sort(key=lambda x: x["pos"])

            # ── Terminal condition: no tool tags found ────────────────────────
            if not pseudo_calls:
                # ── Alien format detection ────────────────────────────────────
                # Gate: only check when no tool has already run this turn
                # (_current_tool_call_log empty = this is the first/only response)
                # and we haven't already sent a nudge (prevents infinite loop).
                # NIM + Google: the classifier catches alien formats (JSON blobs,
                # YAML, || tags, etc.) on both backends. Cloudflare is still excluded
                # — its native tool format differs enough to risk false positives.
                if (not self._current_tool_call_log
                        and not self._alien_nudge_sent
                        and config.ACTIVE_BACKEND in ("nim", "google")):
                    _response_text = clean_content or raw_content or ""
                    det = self._load_alien_detector()
                    if det is not None:
                        is_alien, _prob = det.predict(_response_text)
                        if is_alien:
                            console.print(
                                f"[yellow]⚠️  Alien tool format detected "
                                f"(p={_prob:.2f}) — nudging model to retry[/yellow]"
                            )
                            self._alien_nudge_sent = True
                            messages.append({
                                "role": "user",
                                "content": (
                                    "[SYSTEM — format correction] Your last response "
                                    "appears to contain a tool invocation in your "
                                    "model's native format instead of the required "
                                    "XML powers format. Do NOT use your native tool "
                                    "syntax. Use the XML powers format exactly as "
                                    "shown in the system prompt, e.g. "
                                    "<quick_search>your query</quick_search>. "
                                    "Please retry your response now using the "
                                    "correct format."
                                ),
                            })
                            continue   # re-enter generation loop
                # ── Normal terminal path ──────────────────────────────────────
                self._partial_messages = None
                self.last_tool_summary = self.summarizer.summarize_turn(self._current_tool_call_log)
                self.last_think_trace = "\n\n---\n\n".join(_turn_thinking)
                return clean_content or raw_content

            # ── Execute pseudo-tool calls in order ───────────────────────────
            if len(pseudo_calls) > config.MAX_TOOL_CALLS_PER_TURN:
                console.print(f"[yellow]⚠️ Truncating to {config.MAX_TOOL_CALLS_PER_TURN} tool calls[/yellow]")
                pseudo_calls = pseudo_calls[:config.MAX_TOOL_CALLS_PER_TURN]

            # ── Print narrative text that accompanied tool calls ───────────────
            # When the model writes prose alongside tool tags (e.g. "Let me
            # check the file first.\n<view_lines>..."), that text is stored in
            # history but was never shown in the terminal. Strip all tool XML
            # blocks from the content and print whatever remains so the user
            # sees the model's reasoning/commentary before tool results arrive.
            _narrative = self._strip_tool_tags(clean_content or raw_content or "")
            if _narrative:
                console.print()
                console.rule(style=RULE_STYLE)
                print_smart_response(_narrative)
                console.rule(style=RULE_STYLE)
                console.print()
                self._pending_narratives.append(_narrative)

            # ── Append assistant turn early for interrupt safety ──────────────
            # This ensures that if the user hits Ctrl+C during a tool call, the
            # assistant's decision to call tools is already saved in history.
            #
            # Thinking injection: fold whatever reasoning the model produced
            # (native field OR extracted <think> tags) into the assistant
            # content as a <reasoning> block. This way the model re-reads its
            # own prior chain-of-thought when it receives power results, so it
            # remembers *why* it called the power and *what it was checking*.
            # Works identically for all backends — it's just text in content.
            # Guard: only added when non-empty so silent models are unaffected.
            _thinking = thinking_content or think_extracted or ""
            _base_content = clean_content or raw_content or ""
            asst_msg = {
                "role": "assistant",
                "content": (
                    f"<think>\n{_thinking}\n</think>\n\n{_base_content}"
                    if _thinking else _base_content
                ),
            }
            if config.THINKING_TYPE == "native":
                _kimi_thinking = response_data.get("thinking")
                if _kimi_thinking:
                    # Echo back with the same key NIM/CF uses on output
                    asst_msg["reasoning"] = _kimi_thinking
            # Google thought_signature: attach in-memory only so _convert_messages_to_gemini
            # can emit a proper thought Part on the next API call (when Google reads the tool
            # result). Only set on the Google backend; stripped by the dispatcher for all
            # others. NOT saved to chat_history — purely transient within this turn loop.
            if config.ACTIVE_BACKEND == "google":
                _g_sig = response_data.get("thought_signature")
                if _g_sig:
                    asst_msg["_google_thought_sig"] = _g_sig
            messages.append(asst_msg)

            user_msg = {
                "role": "user",
                "content": "[SYSTEM: Power execution started — awaiting results...]"
            }
            messages.append(user_msg)

            result_parts = []
            for call in pseudo_calls:
                if tool_call_count >= config.MAX_TOOL_CALLS_PER_TURN:
                    console.print(f"[yellow]⚠️ Tool call limit reached ({config.MAX_TOOL_CALLS_PER_TURN})[/yellow]")
                    break

                fn_name = call["fn"]
                fn_args = call["args"]
                console.print(f"[dim cyan]│ [/dim cyan][cyan]{fn_name}[/cyan]")

                result = self._call_tool(fn_name, fn_args, web_agent)

                text_result = str(result)

                self._current_tool_call_log.append({"fn": fn_name, "args": fn_args, "result": text_result})

                # Echo the primary arg in the result label for tools whose result text
                # doesn't already include the input (e.g. quick_search returns only
                # snippets; bash returns only stdout). Tools like str_replace, view_lines,
                # search_in_file already include "file" / "pattern" in their JSON, so they
                # are intentionally omitted here.
                _ECHO_ARG = {
                    "quick_search":              ("query",   150),
                    "bash":                      ("command", 300),
                    "search_history":            ("query",   150),
                    "doc_search":                ("query",   150),
                    "workspace_search":          ("query",   150),
                    "search_semantic_scholar":   ("query",   150),
                }
                _pkey, _plimit = _ECHO_ARG.get(fn_name, (None, 0))
                if _pkey:
                    _raw = (fn_args.get(_pkey) or "").replace("\n", " ")
                    _snippet = _raw[:_plimit] + ("..." if len(_raw) > _plimit else "")
                    _label = f"{fn_name}[{_snippet}]"
                else:
                    _label = fn_name

                result_parts.append(f"[SYSTEM — {_label} result:]\n{text_result}")
                
                # Update the user message live. If Ctrl+C happens mid-loop,
                # the history contains the results of all finished tools.
                user_msg["content"] = "\n\n".join(result_parts) + "\n\n[SYSTEM: All power results above. Continue.]"
                
                tool_call_count += 1



            # If we hit the per-turn limit, stop the outer loop too.
            if tool_call_count >= config.MAX_TOOL_CALLS_PER_TURN:
                self._partial_messages = None
                self.last_tool_summary = self.summarizer.summarize_turn(self._current_tool_call_log)
                self.last_think_trace = "\n\n---\n\n".join(_turn_thinking)
                return clean_content or raw_content or "[MAX TOOL CALLS REACHED]"

        # Max iterations reached
        self._partial_messages = None
        self.last_tool_summary = self.summarizer.summarize_turn(self._current_tool_call_log)
        self.last_think_trace = "\n\n---\n\n".join(_turn_thinking)
        return "[MAX ITERATIONS REACHED]"

    def _strip_tool_tags(self, content: str) -> str:
        """
        Remove all pseudo-tool XML blocks from content and return the
        surrounding narrative text (prose, explanations, comments).

        Used to surface text the model wrote alongside tool calls that would
        otherwise be swallowed — only the final response (no tool calls) gets
        printed via start.py; intermediate responses need this.
        """
        result = content
        for tool_name in self._get_active_known_tools():
            # Block tags: <tool_name ...>...</tool_name>
            result = re.sub(
                rf'<{tool_name}(?:\s[^>]*)?>.*?</{tool_name}>',
                '', result, flags=re.DOTALL
            )
            # Self-closing tags: <tool_name attr="val"/>
            result = re.sub(
                rf'<{tool_name}(?:\s[^>]*?)?/>',
                '', result, flags=re.DOTALL
            )
        return result.strip()

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY & MEMORY
    # ══════════════════════════════════════════════════════════════════════════

    def _load_history(self):
        if os.path.exists(config.CHAT_HISTORY_FILE):
            try:
                with open(config.CHAT_HISTORY_FILE, "r") as f:
                    self.chat_history = json.load(f)
                console.print(f"📜 [bold cyan]Loaded {len(self.chat_history)} messages from history.[/bold cyan]")
            except Exception as e:
                console.print(f"⚠️ History load error: {e}")
                self.chat_history = []

    def save_history(self, skip_global: bool = False):
        """
        Save the current ephemeral chat history to the local state file.
        If not skipped, also writes the last completed turn to the global persistent .jsonl ledger.
        """
        save_json_atomic(config.CHAT_HISTORY_FILE, self.chat_history)

        # Mirror last turn to permanent global .jsonl file — UNLESS caller
        # says to skip (e.g. partial/interrupt saves, /restore truncations).
        if skip_global:
            return

        if len(self.chat_history) >= 2:
            last_user = self.chat_history[-2]
            last_asst = self.chat_history[-1]
            if last_user.get("role") == "user" and last_asst.get("role") == "assistant":
                user_text = last_user.get("content", "")
                asst_text = last_asst.get("content", "")
                if isinstance(user_text, list):
                    user_text = " ".join(
                        b.get("text", "") for b in user_text if isinstance(b, dict)
                    )
                self.global_history.write_turn(user_text, asst_text)