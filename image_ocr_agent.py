# image_ocr_agent.py
#
# OCR pipeline with adaptive AIMD concurrency + circuit breakers + a
# provider dispatcher, each provider backed by an ordered model list:
#
#   NIM:    config.OCR_NIM_MODELS     (e.g. ["stepfun-ai/step-3.7-flash"])
#   Google: config.OCR_GOOGLE_MODELS  (e.g. ["gemma-4-31b-it", "gemma-4-26b-a4b-it"])
#
#   A model that gets rate-limited/overloaded just moves to the next model
#   in its own provider's list (see _try_provider_models) - add or remove
#   models by editing the config list, no code change needed. A provider's
#   circuit breaker only trips once its ENTIRE model list has failed.
#
# Dispatcher (_choose_order):
#   Each page picks whichever of NIM / Google-tier2 is currently healthy
#   (circuit closed) and has more spare AIMD capacity, preferring NIM on a
#   tie (NIM is the tested/known-good default). This lets both providers'
#   capacity get used concurrently instead of Google sitting idle as a
#   fallback-only path - previously ~4 concurrent OCR calls total even
#   though both gates allow up to their own ceiling in parallel.
#
# Adaptive concurrency (_ConcurrencyGate / AIMD):
#   - Starts at INITIAL_CONCURRENCY parallel API calls (moderate, not maxed -
#     an aggressive starting concurrency risks a synchronized 429 burst
#     before the first rate-limit signal even comes back).
#   - 429 received    -> slots = max(MIN_CONCURRENCY, slots // 2)  (true
#     multiplicative decrease - a single 429 usually means capacity was
#     already meaningfully exceeded, so back off hard, not by 1).
#   - After INCREASE_AFTER consecutive successes -> slots += 1.
#   - NIM's ceiling (MAX_CONCURRENCY) is the tested, known-safe value (4).
#     Google's ceiling (GOOGLE_MAX_CONCURRENCY) is set higher since its real
#     limit is unknown - AIMD is left room to discover it empirically rather
#     than being capped at a number tuned for a different provider.
#   The gates are shared across process_images() calls so rate-limit memory
#   persists between PDF page batches.
#
# Circuit breakers (_CircuitBreaker), one per provider:
#   Distinct from the AIMD gate - the gate throttles per-call concurrency,
#   the breaker decides whether to attempt a provider AT ALL right now.
#   After CONSEC_FAILURES_TO_OPEN consecutive full-chain failures, the
#   breaker opens: new pages skip that provider entirely (no wasted
#   retries/timeouts) for a cooldown window, then a single half-open probe
#   tests recovery before resuming normal routing. Without this, a fully-down
#   provider would make every single page pay its full retry+timeout cost
#   before falling through, even though page #1 already proved it's down.
#
# Outer retry rounds (_call_with_rounds):
#   If ALL tiers fail in one round, sleep ROUND_COOLDOWN_BASE*round seconds
#   then restart from the dispatcher.  Up to MAX_ROUNDS (3) rounds per image,
#   so no image is silently dropped during a temporary provider outage.
#
# Design:
#   - ONE image per API call, always.
#   - All images queued to a thread pool; gates control actual API concurrency.
#   - Base64 inline -- no proxy.
#   - Non-streaming for both NIM and Google.
#   - Return type: [{"label": str, "source": str, "markdown": str}]

import os
import re
import time
import base64
import shlex
import threading
import requests
import concurrent.futures
from pathlib import Path
from google import genai
from google.genai import types
import config
from utils import read_api_key, MIME_MAP, backoff_wait, mime_type_from_url, console

# Concurrency / retry constants
# Bug fix: / BUG-A1: deterministic client-side errors (bad request / payload too
# large) indicate a problem with the specific image/request, not a provider
# outage - subsequent pages should still be attempted against the same
# provider, so these codes must NOT trip the circuit breaker. Shared by both
# _try_provider_models (single-image) and _try_provider_models_group
# (multi-image) so the two sibling methods can't drift out of parity again.
_CLIENT_SIDE_CODES = {400, 413, 422}

RETRIES_PER_TIER    = 2   # NIM retry attempts before falling to next tier
MAX_ROUNDS          = 3   # full Tier1→2→3 restarts before permanent failure
ROUND_COOLDOWN_BASE = 30  # seconds; sleep = base * (round_number - 1) between rounds
INITIAL_CONCURRENCY = 4   # starting parallel API call slots (moderate - see header)
MAX_CONCURRENCY        = 4    # ceiling for NIM's AIMD increase (tested/known-safe)
GOOGLE_MAX_CONCURRENCY = 16   # ceiling for Google's AIMD increase - deliberately
                              # higher than NIM's since Google's real RPM limit is
                              # unknown; AIMD needs room above 4 to discover it.
MIN_CONCURRENCY     = 1   # floor for AIMD decrease
INCREASE_AFTER      = 3   # consecutive successes needed to bump concurrency by 1

# Circuit breaker constants
CONSEC_FAILURES_TO_OPEN = 4     # consecutive full-chain failures before a
                                 # provider's circuit opens (stops being tried)
BREAKER_BASE_COOLDOWN   = 30    # seconds before the first half-open probe
BREAKER_MAX_COOLDOWN    = 300   # cooldown ceiling (doubles on each failed probe)




# Strict transcription prompt
_OCR_SYSTEM_PROMPT = r"""\
You are a transcription engine. Read the image. Write down exactly what is there in it. 

CORE RULE: Your output is a mirror of the image.

You may receive zoomed in sections of the page for clarity, if image is big, along with the full image. Produce a single unified transcription of the complete image.

━━━ TEXT AND MATH ━━━

Write what is written. Every symbol, every coefficient, every operator.
- Math in LaTeX: inline $...$ or display $$...$$
- Plain prose stays plain — do NOT wrap it in $...$
- Struck-out content: skip entirely
- Ambiguous symbol: write your best read + [unclear: X or Y]
- No added steps, labels, or structure that isn't in the image

SYMBOLS THAT LOOK ALIKE — read carefully, do not assume:
$\sec$ vs $\cos$ · $\tan$ vs $\tau$ · $\ln$ vs $\log$
$\partial$ vs $d$ · $\pi$ vs $\Pi$ · $\equiv$ vs $=$
$\rightarrow$ vs $\Rightarrow$ · $\tan^{-1}$ vs $\tan$
Greek letters ($\theta$, $\lambda$, $\omega$, $\alpha$, $\beta$, $\phi$) —
identify exactly, never substitute

EXPONENTS — look twice:
$e^{2t}$ and $e^t$ are different. Count every term inside the exponent.

━━━ VISUALS ━━━

For each graph, diagram, figure, or scene — insert one block after the
surrounding text, in order of appearance:

Mathematical graph →
[GRAPH CODE]
matplotlib code reproducing it: correct axes, labels, ranges, curves,
marked points, annotations — everything visible
[/GRAPH CODE]

Everything else →
[DIAGRAM:
Type: geometric figure | scene/photo | flowchart | etc.

Describe completely enough that someone who cannot see the image can
reconstruct or use it. Include all labels, values, structure, and
spatial relationships. For scenes, cover board/screen content first.
]

━━━ OUTPUT ━━━

No preamble. No commentary. No solving.
Transcription and visual blocks only.
"""

_OCR_USER_PROMPT = (
    "Transcribe the image"
)


# ----
# ADAPTIVE CONCURRENCY GATE  (AIMD)
# ----

class _ConcurrencyGate:
    """
    AIMD (Additive Increase, Multiplicative Decrease) concurrency controller.

    Controls how many API calls are in-flight simultaneously across all
    parallel image threads. Thread-safe via threading.Condition.

    DECREASE: On 429 rate-limit → desired = max(minimum, desired // 2).
              True multiplicative decrease - halves rather than -=1, since a
              429 usually means the limit was already meaningfully exceeded.
              Waiting threads see the lower limit immediately; inflight calls
              already running complete normally and then free their slot.

    INCREASE: After INCREASE_AFTER consecutive successes → desired += 1
              (ceiling: this instance's `maximum`, set per-provider - see
              MAX_CONCURRENCY vs GOOGLE_MAX_CONCURRENCY). Waiting threads are
              notified so the new slot is filled immediately.

    Usage:
        gate.acquire()   # blocks until inflight < desired
        try:
            result = api_call()
            gate.report_success()
            return result
        except RateLimitError:
            gate.report_rate_limit()
            raise
        finally:
            gate.release()
    """

    def __init__(self, initial: int = INITIAL_CONCURRENCY,
                 minimum: int = MIN_CONCURRENCY,
                 maximum: int = MAX_CONCURRENCY):
        self._cond      = threading.Condition(threading.Lock())
        self._desired   = initial
        self._inflight  = 0
        self._min       = minimum
        self._max       = maximum
        self._successes = 0   # consecutive successful API calls

    # Public gate interface

    def acquire(self):
        """Block until a concurrency slot is available, then claim it."""
        with self._cond:
            while self._inflight >= self._desired:
                self._cond.wait()
            self._inflight += 1

    def release(self):
        """Free a concurrency slot and wake any threads waiting to acquire."""
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    # Feedback signals

    def report_success(self):
        """Call after every successful API response. Increases desired after INCREASE_AFTER streak."""
        with self._cond:
            self._successes += 1
            if self._successes >= INCREASE_AFTER and self._desired < self._max:
                self._desired   += 1
                self._successes  = 0
                self._cond.notify_all()   # wake a waiting thread to fill new slot
                console.print(
                    f"   [dim green]⬆ Concurrency → {self._desired} "
                    f"(after {INCREASE_AFTER} successes)[/dim green]"
                )

    def report_rate_limit(self):
        """Call when a 429 / RESOURCE_EXHAUSTED is received. Halves desired
        (floored at minimum) - true multiplicative decrease. A single 429
        usually means capacity was already meaningfully exceeded by the time
        it arrives (other in-flight calls were sent before this signal came
        back), so back off hard rather than grinding down by 1 through a
        storm of repeated rate-limit hits."""
        with self._cond:
            self._successes = 0
            new_desired = max(self._min, self._desired // 2)
            if new_desired < self._desired:
                self._desired = new_desired
                console.print(
                    f"   [yellow]⬇ Concurrency → {self._desired} (rate-limited)[/yellow]"
                )

    @property
    def current(self) -> int:
        return self._desired

    def free_slots(self) -> int:
        """Non-blocking peek: how many slots are free right now.
        Used only for the dispatcher's soft tie-break preference between
        providers - not a reservation, so a race against acquire() is fine
        here (worst case the dispatcher's guess is stale by one slot)."""
        with self._cond:
            return max(0, self._desired - self._inflight)


# ----
# CIRCUIT BREAKER  (per-provider, distinct from the AIMD gate)
# ----

class _CircuitBreaker:
    """
    Per-provider circuit breaker for FULL-CHAIN failures - distinct from the
    AIMD gate above, which throttles per-call concurrency. This decides
    whether to attempt a provider AT ALL right now.

    Without this: if a provider is fully down, every single page still pays
    its full retry+timeout cost before falling through to the other
    provider - even though page #1 already proved this provider is dead.
    The breaker lets later pages skip it entirely until it's likely back.

    States:
      CLOSED    - normal; provider is tried as usual.
      OPEN      - provider tripped after CONSEC_FAILURES_TO_OPEN consecutive
                  full-chain failures; skipped entirely until cooldown elapses.
      HALF_OPEN - cooldown elapsed; exactly one probe page is allowed through
                  to test recovery. Success → CLOSED. Failure → OPEN again,
                  with cooldown doubled (capped at BREAKER_MAX_COOLDOWN) so a
                  still-dead provider isn't hammered with probes.

    Thread-safe via a plain Lock (no waiting/blocking semantics needed here,
    unlike the gate - allow() is a quick non-blocking check).
    """

    def __init__(self, name: str,
                 consec_failures_to_open: int = CONSEC_FAILURES_TO_OPEN,
                 base_cooldown: float = BREAKER_BASE_COOLDOWN,
                 max_cooldown: float = BREAKER_MAX_COOLDOWN):
        self._name           = name
        self._lock            = threading.Lock()
        self._threshold        = consec_failures_to_open
        self._base_cooldown    = base_cooldown
        self._max_cooldown     = max_cooldown
        self._consec_failures  = 0
        self._state             = "closed"   # closed | open | half_open
        self._opened_at         = 0.0
        self._cooldown          = base_cooldown
        self._half_open_inflight = False

    def allow(self) -> bool:
        """Non-blocking check: may this provider be attempted right now?"""
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._cooldown:
                    self._state = "half_open"
                    self._half_open_inflight = True
                    console.print(
                        f"   [dim cyan]⟳ {self._name} circuit half-open — probing...[/dim cyan]"
                    )
                    return True
                return False
            # half_open: only the single in-flight probe is allowed through
            return False

    def record_success(self):
        with self._lock:
            was_tripped = self._state != "closed"
            self._state             = "closed"
            self._consec_failures   = 0
            self._cooldown          = self._base_cooldown
            self._half_open_inflight = False
            if was_tripped:
                console.print(
                    f"   [green]✓ {self._name} circuit closed — provider recovered[/green]"
                )

    def record_inconclusive(self):
        # Bug fix: unlock the half_open slot without touching failure counters
        with self._lock:
            if self._state == "half_open":
                self._state = "open"          # let the next call re-probe after cooldown
                self._half_open_inflight = False

    def record_failure(self):
        with self._lock:
            if self._state == "half_open":
                # Probe failed - reopen with a longer cooldown before retrying.
                self._cooldown          = min(self._max_cooldown, self._cooldown * 2)
                self._state              = "open"
                self._opened_at          = time.monotonic()
                self._half_open_inflight = False
                console.print(
                    f"   [red]✗ {self._name} circuit re-opened — cooling down {self._cooldown:.0f}s[/red]"
                )
                return

            self._consec_failures += 1
            if self._state == "closed" and self._consec_failures >= self._threshold:
                self._state     = "open"
                self._opened_at = time.monotonic()
                console.print(
                    f"   [red]✗ {self._name} circuit opened after {self._consec_failures} "
                    f"consecutive failures — cooling down {self._cooldown:.0f}s[/red]"
                )



class ImageOCRAgent:
    """
    OCR images via a dispatcher that picks whichever of NIM / Google is
    healthy and has spare capacity, each provider trying its own ordered
    model list (config.OCR_NIM_MODELS / config.OCR_GOOGLE_MODELS) before
    the dispatcher moves on to the other provider.

    Adaptive concurrency (per provider, independent):
      - Starts at INITIAL_CONCURRENCY (4) parallel API call slots.
      - 429 received → slots = max(1, slots // 2)  (multiplicative decrease).
      - INCREASE_AFTER (3) consecutive successes → slots += 1.
      - NIM ceiling: MAX_CONCURRENCY (4, tested). Google ceiling:
        GOOGLE_MAX_CONCURRENCY (16) - deliberately higher since Google's real
        limit is unknown; AIMD needs room to discover it.
      The gates are shared across process_images() calls so rate-limit
      history from one batch carries into the next batch.

    Circuit breakers (per provider):
      - After CONSEC_FAILURES_TO_OPEN consecutive full-chain failures, a
        provider is skipped entirely (not retried per-page) until a cooldown
        elapses, then a single probe tests recovery. See _CircuitBreaker.

    Outer retry rounds:
      - If all tiers fail in one round, sleep then restart from the dispatcher.
      - Up to MAX_ROUNDS (3) full rounds per image.
      - Guarantees no image is silently dropped during a temporary outage.
    """

    def __init__(self):
        try:
            self._nvidia_key = read_api_key(config.NVIDIA_API_KEY_FILE, silence_warning=True)
            if not self._nvidia_key:
                console.print("⚠️  [yellow]ImageOCRAgent: NVIDIA key not found — NIM disabled[/yellow]")

            google_api_key      = read_api_key(config.GOOGLE_API_KEY_FILE, silence_warning=True)
            self._google_client = genai.Client(api_key=google_api_key) if google_api_key else None
            if not self._google_client:
                console.print("⚠️  [yellow]ImageOCRAgent: Google key not found — Google tiers disabled[/yellow]")

            # Two independent AIMD gates - NIM and Google have separate RPM quotas.
            # A NIM 429 must NOT throttle Google slots, and vice versa. Google's
            # ceiling is set higher than NIM's (unknown real limit - see header).
            self._nim_gate    = _ConcurrencyGate(maximum=MAX_CONCURRENCY)
            self._google_gate = _ConcurrencyGate(maximum=GOOGLE_MAX_CONCURRENCY)

            # Two independent circuit breakers - see _CircuitBreaker docstring.
            self._nim_breaker    = _CircuitBreaker("NIM")
            self._google_breaker = _CircuitBreaker("Google")

            self._thread_local = threading.local()

        except Exception as e:
            console.print(f"🚨 [bold red]ImageOCRAgent init failed: {e}[/bold red]")
            raise

    # Public entry point

    def process_images(self, image_input: str) -> list:
        """
        Parse image_input, OCR each image independently, return result dicts.

        Args:
            image_input: Space-separated image paths / URLs / data URIs.

        Returns:
            List of {"label", "source", "markdown"} dicts in original order.
        """
        if not image_input or not image_input.strip():
            return []

        try:
            items = shlex.split(image_input, posix=False)
        except Exception:
            items = image_input.split()

        items = [i.strip().strip('"').strip("'") for i in items if i.strip()]
        if not items:
            return []

        console.print(
            f"\n🔍 [cyan]OCR: {len(items)} image(s) — "
            f"{'parallel' if len(items) > 1 else 'single'} call(s) "
            f"[NIM gate: {self._nim_gate.current} | Google gate: {self._google_gate.current}][/cyan]"
        )

        # Resolve all images before spawning threads
        resolved = []
        for item in items:
            try:
                src, url = self._resolve(item)
                resolved.append((src, url))
            except Exception as e:
                console.print(f"   [yellow]⚠️  skipping '{item}': {e}[/yellow]")
                resolved.append((item, None))

        results = [None] * len(resolved)

        def _ocr_one(idx: int, display_source: str, image_url: str):
            label = f"Image {idx + 1}" if len(resolved) > 1 else "Image"
            if image_url is None:
                results[idx] = {
                    "label":    label,
                    "source":   display_source,
                    "markdown": "[Image could not be loaded]",
                }
                return
            markdown, used_model = self._call_with_rounds(image_url, label)
            _is_error = used_model.startswith("[ERROR") or markdown.startswith("[OCR ERROR")
            if not _is_error:
                console.print(f"   ✅ [green]{label} OCR complete[/green] [dim]({used_model})[/dim]")
            # Bug fix: set error=True on failed results so callers can
            # skip embedding the error message as document content.
            results[idx] = {
                "label":    label,
                "source":   display_source,
                "markdown": markdown,
                "error":    _is_error,
            }

        # Spawn one thread per image - the _ConcurrencyGate throttles how many
        # actually make API calls at once, so we don't need to limit max_workers here.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(resolved)) as pool:
            futures = [pool.submit(_ocr_one, i, src, url)
                       for i, (src, url) in enumerate(resolved)]
            concurrent.futures.wait(futures)

        return [r for r in results if r is not None]

    # Outer retry rounds wrapper

    def _call_with_rounds(self, image_url: str, label: str) -> tuple:
        self._thread_local.last_failure_was_client_error = False
        """
        Outer loop: attempt the full provider dispatch (_call_with_fallback)
        up to MAX_ROUNDS times.

        If a round returns a result (any provider succeeds), return immediately.
        If a round exhausts every available provider, sleep
        ROUND_COOLDOWN_BASE*(round-1) seconds then restart. The increasing
        sleep (30s, 60s) is enough time for NIM and Google RPM windows to
        fully reset after a simultaneous rate-limit event.

        Returns (markdown_text, model_name_or_error_tag).
        """
        for round_num in range(1, MAX_ROUNDS + 1):
            if round_num > 1:
                cooldown = ROUND_COOLDOWN_BASE * (round_num - 1)
                console.print(
                    f"   ⏳ [yellow]{label}: all providers failed in round {round_num - 1}. "
                    f"Waiting {cooldown}s before round {round_num}...[/yellow]"
                )
                time.sleep(cooldown)

            console.print(f"   [dim]{label}: OCR round {round_num}/{MAX_ROUNDS}[/dim]")
            result = self._call_with_fallback(image_url, label)
            if result is not None:
                return result
            # Bug fix: if all failed and the cause was client-side, don't sleep
            # and retry - it will fail again. Break immediately.
            if getattr(self._thread_local, "last_failure_was_client_error", False):
                break

        console.print(
            f"   🚨 [bold red]{label}: permanently failed after {MAX_ROUNDS} round(s).[/bold red]"
        )
        return "[OCR ERROR — all tiers exhausted across all retry rounds]", "[ERROR]"

    # Dispatcher: pick provider order for this round

    def _choose_order(self) -> list:
        """
        Decide which provider(s) to try this round, and in what order.
        Skips a provider entirely if its circuit is open (see _CircuitBreaker),
        it has no API key, or it has no models configured.
        Prefers NIM on a tie in free AIMD capacity (tested/known-good
        default), otherwise sends to whichever has more spare slots right
        now - this is what lets both providers' capacity get used
        concurrently instead of Google sitting idle as fallback-only.
        """
        nim_ok    = bool(self._nvidia_key) and bool(config.OCR_NIM_MODELS) and self._nim_breaker.allow()
        google_ok = bool(self._google_client) and bool(config.OCR_GOOGLE_MODELS) and self._google_breaker.allow()

        if nim_ok and google_ok:
            nim_free    = self._nim_gate.free_slots()
            google_free = self._google_gate.free_slots()
            return ["google", "nim"] if google_free > nim_free else ["nim", "google"]
        if nim_ok:
            return ["nim"]
        if google_ok:
            return ["google"]
        return []

    # Try every model in one provider's list before giving up on it

    def _try_provider_models(self, provider: str, image_url: str, label: str) -> tuple | None:
        """
        Try every model in this provider's configured model list, in order.
        A model that exhausts its retries just moves on to the next model in
        the list (e.g. a newer/less-loaded model) - the provider's circuit
        breaker only records a failure once EVERY model in its list has
        failed this round, not on each individual model's exhaustion.
        Returns (text, model) on success, None if the whole list failed.
        """
        if provider == "nim":
            models, call_fn, breaker = config.OCR_NIM_MODELS, self._call_nim_ocr, self._nim_breaker
            retries, base_429_wait   = RETRIES_PER_TIER, 5
        else:
            models, call_fn, breaker = config.OCR_GOOGLE_MODELS, self._call_google_ocr, self._google_breaker
            retries, base_429_wait   = 2, 3

        last_err = None
        for model_idx, model in enumerate(models):
            for attempt in range(1, retries + 1):
                try:
                    text = call_fn(model, image_url)
                    breaker.record_success()
                    return text, model
                except Exception as e:
                    last_err = f"{e.__class__.__name__}: {str(e)[:80]}"
                    is_429   = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                    # Gate already called report_rate_limit inside call_fn
                    if attempt < retries:
                        wait = base_429_wait if is_429 else 2 * attempt
                        console.print(
                            f"   [yellow]⚠️  {label} {provider}/{model} attempt {attempt}/{retries} "
                            f"failed ({last_err}). Retrying in {wait}s...[/yellow]"
                        )
                        time.sleep(wait)

            more_models_left = model_idx < len(models) - 1
            if more_models_left:
                console.print(
                    f"   [yellow]⚠️  {label} {provider}/{model} exhausted ({last_err}). "
                    f"Trying next {provider} model: {models[model_idx + 1]}...[/yellow]"
                )
                time.sleep(2)
            else:
                console.print(
                    f"   [yellow]⚠️  {label} {provider} exhausted all {len(models)} "
                    f"model(s) ({last_err}).[/yellow]"
                )

        # Bug fix: don't trip the circuit breaker for deterministic client-side
        # errors (bad request / payload too large) - those indicate a problem
        # with this specific image, not a provider outage, so subsequent pages
        # should still be attempted against the same provider.
        _is_client_error = any(
            f" {_code}" in str(last_err) or f"({_code})" in str(last_err)
            for _code in _CLIENT_SIDE_CODES
        )
        if not _is_client_error:
            breaker.record_failure()
        else:
            breaker.record_inconclusive()
            # Bug fix: communicate client error up to break out of rounds loop
            self._thread_local.last_failure_was_client_error = True
        return None

    # Single-round dispatch across providers

    def _call_with_fallback(self, image_url: str, label: str) -> tuple | None:
        """
        One attempt of the dispatcher: try providers in _choose_order()'s
        preference, each exhausting its own model list before moving to the
        next provider. Returns (text, model) on first success; None if every
        available provider's full model list fails this round.
        """
        order = self._choose_order()
        if not order:
            console.print(
                f"   [dim]{label}: no providers available this round "
                f"(circuits open or no keys/models configured)[/dim]"
            )
            return None

        for provider in order:
            result = self._try_provider_models(provider, image_url, label)
            if result is not None:
                return result

        return None

    # NVIDIA NIM call  (gated)

    def _call_nim_ocr(self, model: str, image_url: str) -> str:
        """
        POST OCR request to NVIDIA NIM with no-think mode.
        Wraps the HTTP call with the AIMD gate - 429 reduces concurrency slots
        immediately; success increments the streak counter.
        Attaches _status_code to exceptions so callers can detect 429s.
        """
        payload = {
            "model":             model,
            "messages": [
                {"role": "system", "content": _OCR_SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text",      "text": _OCR_USER_PROMPT},
                    ],
                },
            ],
            "max_tokens":        config.OCR_MAX_TOKENS,
            "temperature":       1.0,
            "top_p":             0.95,
            "frequency_penalty": 0.0,
            "presence_penalty":  0.0,
            "stream":            False,
            # Enable internal thinking for OCR calls.
            "chat_template_kwargs": {"enable_thinking": True},
        }

        self._nim_gate.acquire()
        try:
            resp = requests.post(
                f"{config.NVIDIA_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._nvidia_key}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
                json=payload,
                timeout=600,
            )

            if resp.status_code == 429:
                self._nim_gate.report_rate_limit()
                exc = RuntimeError(f"NIM API error 429: {resp.text[:200]}")
                exc._status_code = 429
                raise exc

            if resp.status_code != 200:
                exc = RuntimeError(f"NIM API error {resp.status_code}: {resp.text[:200]}")
                exc._status_code = resp.status_code
                raise exc

            text = resp.json()["choices"][0]["message"]["content"] or ""
            text = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text,
                          flags=re.DOTALL | re.IGNORECASE).strip()
            self._nim_gate.report_success()
            return text

        finally:
            self._nim_gate.release()

    # Google API call  (gated)

    def _call_google_ocr(self, model: str, image_url: str) -> str:
        """
        OCR via Google Gemini API with thinking_level=NONE (zero scratchpad).
        NONE: thinking budget = 0 - no scratchpad at all.
        MINIMAL would still generate a small scratchpad; NONE is correct for OCR.
        Wraps the call with the AIMD gate.
        Handles data-URI and http(s) images.
        """
        if image_url.startswith("data:"):
            try:
                header, b64data = image_url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                raw  = base64.b64decode(b64data)
                image_part = types.Part.from_bytes(data=raw, mime_type=mime)
            except Exception as e:
                raise RuntimeError(f"Could not decode data URI: {e}")
        elif image_url.startswith("http"):
            # Bug fix: Google FileData only accepts File API URIs, not plain HTTP
            # URLs.  Download the image and pass as inline bytes instead.
            try:
                import urllib.request as _urllib_req
                # Bug fix: plain urlopen sends 'Python-urllib/x.y' which
                # Cloudflare, Imgur, Wikipedia etc. block with 403. Use a
                # browser-style User-Agent to avoid the block.
                _req = _urllib_req.Request(
                    image_url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ImageOCR/1.0)"},
                )
                with _urllib_req.urlopen(_req, timeout=15) as _r:
                    _raw = _r.read(20 * 1024 * 1024 + 1)
                if len(_raw) > 20 * 1024 * 1024:
                    raise ValueError("Remote image exceeds 20 MB limit")
                image_part = types.Part.from_bytes(
                    data=_raw, mime_type=mime_type_from_url(image_url)
                )
            except Exception as e:
                raise RuntimeError(f"Could not download image URL for Google OCR: {e}")
        else:
            raise ValueError(f"Unsupported image_url for Google OCR: {image_url[:60]}")

        contents = [
            types.Content(role="user", parts=[
                image_part,
                types.Part.from_text(text=_OCR_USER_PROMPT),
            ])
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=_OCR_SYSTEM_PROMPT.strip(),
            max_output_tokens=config.OCR_MAX_TOKENS,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )

        self._google_gate.acquire()
        try:
            response = self._google_client.models.generate_content(
                model=model, contents=contents, config=cfg,
            )
            text = response.text or ""
            text = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text,
                          flags=re.DOTALL | re.IGNORECASE).strip()
            self._google_gate.report_success()
            return text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                self._google_gate.report_rate_limit()
            raise

        finally:
            self._google_gate.release()

    # Multi-image NIM call  (gated)

    def _call_nim_ocr_multi(self, model: str, image_urls: list) -> str:
        """
        POST a single OCR request to NVIDIA NIM with MULTIPLE images in one
        content array (overview + tile crops). The model sees all images at
        once and returns one unified transcription.
        Same AIMD gating and error handling as _call_nim_ocr.
        """
        content = []
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": _OCR_USER_PROMPT})

        payload = {
            "model":             model,
            "messages": [
                {"role": "system", "content": _OCR_SYSTEM_PROMPT.strip()},
                {"role": "user",   "content": content},
            ],
            "max_tokens":        config.OCR_MAX_TOKENS,
            "temperature":       1.0,
            "top_p":             0.95,
            "frequency_penalty": 0.0,
            "presence_penalty":  0.0,
            "stream":            False,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        self._nim_gate.acquire()
        try:
            resp = requests.post(
                f"{config.NVIDIA_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._nvidia_key}",
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
                json=payload,
                timeout=600,
            )

            if resp.status_code == 429:
                self._nim_gate.report_rate_limit()
                exc = RuntimeError(f"NIM API error 429: {resp.text[:200]}")
                exc._status_code = 429
                raise exc

            if resp.status_code != 200:
                exc = RuntimeError(f"NIM API error {resp.status_code}: {resp.text[:200]}")
                exc._status_code = resp.status_code
                raise exc

            text = resp.json()["choices"][0]["message"]["content"] or ""
            text = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text,
                          flags=re.DOTALL | re.IGNORECASE).strip()
            self._nim_gate.report_success()
            return text

        finally:
            self._nim_gate.release()

    # Multi-image Google call  (gated)

    def _call_google_ocr_multi(self, model: str, image_urls: list) -> str:
        """
        OCR via Google Gemini API with MULTIPLE images in one call.
        Builds one Part per image followed by the text prompt Part.
        Same AIMD gating as _call_google_ocr.
        """
        parts = []
        for image_url in image_urls:
            if image_url.startswith("data:"):
                try:
                    header, b64data = image_url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    raw  = base64.b64decode(b64data)
                    parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                except Exception as e:
                    raise RuntimeError(f"Could not decode data URI: {e}")
            elif image_url.startswith("http"):
                # Bug fix: same fix as _call_google_ocr - download to bytes.
                try:
                    import urllib.request as _urllib_req
                    # Bug fix: + #15 fix: browser User-Agent to bypass CDN 403s;
                    # 20 MB read cap to prevent OOM on unexpectedly large images.
                    _req = _urllib_req.Request(
                        image_url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; ImageOCR/1.0)"},
                    )
                    with _urllib_req.urlopen(_req, timeout=15) as _r:
                        _raw = _r.read(20 * 1024 * 1024 + 1)
                    if len(_raw) > 20 * 1024 * 1024:
                        raise ValueError("Remote image exceeds 20 MB limit")
                    parts.append(types.Part.from_bytes(
                        data=_raw, mime_type=mime_type_from_url(image_url)
                    ))
                except Exception as e:
                    raise RuntimeError(f"Could not download image URL for Google OCR multi: {e}")
            else:
                raise ValueError(f"Unsupported image_url for Google OCR: {image_url[:60]}")
        parts.append(types.Part.from_text(text=_OCR_USER_PROMPT))

        contents = [types.Content(role="user", parts=parts)]
        cfg = types.GenerateContentConfig(
            system_instruction=_OCR_SYSTEM_PROMPT.strip(),
            max_output_tokens=config.OCR_MAX_TOKENS,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )

        self._google_gate.acquire()
        try:
            response = self._google_client.models.generate_content(
                model=model, contents=contents, config=cfg,
            )
            text = response.text or ""
            text = re.sub(r"<(?:think|reasoning)>.*?</(?:think|reasoning)>", "", text,
                          flags=re.DOTALL | re.IGNORECASE).strip()
            self._google_gate.report_success()
            return text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                self._google_gate.report_rate_limit()
            raise

        finally:
            self._google_gate.release()

    # Group retry chain (mirrors single-image chain, uses multi-image calls)

    def _try_provider_models_group(self, provider: str, image_urls: list,
                                   label: str) -> tuple | None:
        """
        Like _try_provider_models but calls the multi-image variants.
        Tries every model in the provider's list before giving up.
        """
        if provider == "nim":
            models, call_fn, breaker = (config.OCR_NIM_MODELS,
                                        self._call_nim_ocr_multi, self._nim_breaker)
            retries, base_429_wait   = RETRIES_PER_TIER, 5
        else:
            models, call_fn, breaker = (config.OCR_GOOGLE_MODELS,
                                        self._call_google_ocr_multi, self._google_breaker)
            retries, base_429_wait   = 2, 3

        last_err = None
        for model_idx, model in enumerate(models):
            for attempt in range(1, retries + 1):
                try:
                    text = call_fn(model, image_urls)
                    breaker.record_success()
                    return text, model
                except Exception as e:
                    last_err = f"{e.__class__.__name__}: {str(e)[:80]}"
                    is_429   = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                    if attempt < retries:
                        wait = base_429_wait if is_429 else 2 * attempt
                        console.print(
                            f"   [yellow]⚠️  {label} {provider}/{model} attempt {attempt}/{retries} "
                            f"failed ({last_err}). Retrying in {wait}s...[/yellow]"
                        )
                        time.sleep(wait)

            more_models_left = model_idx < len(models) - 1
            if more_models_left:
                console.print(
                    f"   [yellow]⚠️  {label} {provider}/{model} exhausted ({last_err}). "
                    f"Trying next {provider} model: {models[model_idx + 1]}...[/yellow]"
                )
                time.sleep(2)
            else:
                console.print(
                    f"   [yellow]⚠️  {label} {provider} exhausted all {len(models)} "
                    f"model(s) ({last_err}).[/yellow]"
                )

        # BUG-A1 fix: mirror the same client-side-error guard used in
        # _try_provider_models (single-image) - a 400/413/422 from an
        # oversized multi-image request is a request-specific error, not a
        # provider outage, so it must not trip the circuit breaker and skip
        # the provider for subsequent pages.
        _is_client_error = any(
            f" {_code}" in str(last_err) or f"({_code})" in str(last_err)
            for _code in _CLIENT_SIDE_CODES
        )
        if not _is_client_error:
            breaker.record_failure()
        else:
            breaker.record_inconclusive()
            # Bug fix: communicate client error up to break out of rounds loop
            self._thread_local.last_failure_was_client_error = True
        return None

    def _call_with_fallback_group(self, image_urls: list, label: str) -> tuple | None:
        """Single-round dispatch across providers for a multi-image group call."""
        order = self._choose_order()
        if not order:
            console.print(
                f"   [dim]{label}: no providers available this round "
                f"(circuits open or no keys/models configured)[/dim]"
            )
            return None

        for provider in order:
            result = self._try_provider_models_group(provider, image_urls, label)
            if result is not None:
                return result

        return None

    def _call_with_rounds_group(self, image_urls: list, label: str) -> tuple:
        self._thread_local.last_failure_was_client_error = False
        """
        Outer retry loop for multi-image group calls.
        Mirrors _call_with_rounds exactly — up to MAX_ROUNDS full attempts.
        """
        for round_num in range(1, MAX_ROUNDS + 1):
            if round_num > 1:
                cooldown = ROUND_COOLDOWN_BASE * (round_num - 1)
                console.print(
                    f"   ⏳ [yellow]{label}: all providers failed in round {round_num - 1}. "
                    f"Waiting {cooldown}s before round {round_num}...[/yellow]"
                )
                time.sleep(cooldown)

            console.print(f"   [dim]{label}: OCR round {round_num}/{MAX_ROUNDS}[/dim]")
            result = self._call_with_fallback_group(image_urls, label)
            if result is not None:
                return result
            # Bug fix: if all failed and the cause was client-side, don't sleep
            # and retry - it will fail again. Break immediately.
            if getattr(self._thread_local, "last_failure_was_client_error", False):
                break

        console.print(
            f"   🚨 [bold red]{label}: permanently failed after {MAX_ROUNDS} round(s).[/bold red]"
        )
        return "[OCR ERROR — all tiers exhausted across all retry rounds]", "[ERROR]"

    # Public group entry point

    def process_image_group(self, image_input: str) -> dict:
        """
        Parse image_input (same shlex format as process_images), resolve all
        images, then send them ALL in ONE API call - overview + tile crops
        together so the model sees full context AND zoomed detail at once.

        This is the tiling companion to process_images: where process_images
        sends each image independently, process_image_group sends the whole
        group as a single multi-image message and returns ONE result dict.

        Returns: {"label": str, "source": str, "markdown": str}
        """
        if not image_input or not image_input.strip():
            return {"label": "Image", "source": "", "markdown": "[No images provided]"}

        try:
            items = shlex.split(image_input, posix=False)
        except Exception:
            items = image_input.split()

        items = [i.strip().strip('"').strip("'") for i in items if i.strip()]
        if not items:
            return {"label": "Image", "source": "", "markdown": "[No images provided]"}

        console.print(
            f"\n🔍 [cyan]OCR group: {len(items)} image(s) in one call "
            f"[NIM gate: {self._nim_gate.current} | Google gate: {self._google_gate.current}][/cyan]"
        )

        # Resolve all images to data URIs / URLs
        image_urls = []
        sources    = []
        for item in items:
            try:
                src, url = self._resolve(item)
                image_urls.append(url)
                sources.append(src)
            except Exception as e:
                console.print(f"   [yellow]⚠️  skipping '{item}': {e}[/yellow]")

        if not image_urls:
            return {
                "label":    "Image",
                "source":   ", ".join(sources),
                "markdown": "[All images failed to load]",
            }

        label = f"Page ({len(image_urls)} tile(s))"
        markdown, used_model = self._call_with_rounds_group(image_urls, label)
        if not used_model.startswith("[ERROR"):
            console.print(f"   ✅ [green]{label} OCR complete[/green] [dim]({used_model})[/dim]")

        return {
            "label":    label,
            "source":   sources[0] if sources else "",
            "markdown": markdown,
        }

    # Image resolver

    def _resolve(self, item: str) -> tuple:
        """Resolve an image input to (display_source, url_or_data_uri)."""
        if item.startswith("http://") or item.startswith("https://"):
            return item, item
        if item.startswith("data:"):
            return "data-uri", item
        if os.path.isfile(item):
            ext = Path(item).suffix.lower()
            if ext not in MIME_MAP:
                raise ValueError(
                    f"Unsupported format '{ext}'. "
                    f"Supported: {', '.join(MIME_MAP.keys())}"
                )
            size_mb = os.path.getsize(item) / (1024 * 1024)
            if size_mb > 20:
                raise ValueError(f"Image too large: {size_mb:.1f} MB (limit 20 MB)")
            with open(item, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            console.print(f"   [dim]encoded {Path(item).name} ({size_mb:.2f} MB)[/dim]")
            return Path(item).name, f"data:{MIME_MAP[ext]};base64,{b64}"
        raise FileNotFoundError(
            f"Cannot resolve '{item}'. "
            "Must be a local file path, http(s) URL, or data URI."
        )