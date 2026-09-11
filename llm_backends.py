# llm_backends.py - LLM API backends mixin: NIM, Google Gemini, Cloudflare
#
# Extracted from emperor_agent.py to keep that file at a manageable size.
# EmperorAgent inherits this mixin:
#   class EmperorAgent(LLMBackendsMixin, ToolHandlersMixin): ...
#
# The mixin accesses instance attributes set by EmperorAgent.__init__:
#   self.nim_api_key, self.nim_base_url  — for NIM
#   self.google_client                   — for Google Gemini
#   self.cf_api_key, self.cf_base_url    — for Cloudflare

import base64
import json
import time
import requests
from google.genai import types

import config
from utils import backoff_wait, mime_type_from_url, console


class LLMBackendsMixin:
    """
    HTTP/streaming API backends for NIM, Google Gemini, and Cloudflare.

    Mixed into EmperorAgent. Never instantiated directly.
    All `self.*` attributes referenced here are set in EmperorAgent.__init__.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # DISPATCHER
    # ══════════════════════════════════════════════════════════════════════════

    def _make_request(self, messages, temp) -> dict:
        """Sanitize messages and dispatch to the active backend."""
        # Strip fields that are only valid in API *responses*, not requests.
        # 'reasoning': stored on assistant messages for NIM/Kimi native thinking;
        #              must be removed before sending — backends reject it.
        # '_google_thought_sig': encrypted thought_signature from Google, kept
        #              only for the Google backend (where _convert_messages_to_gemini
        #              reads it to emit proper thought Parts).  Stripped everywhere
        #              else so non-Google backends never see an unknown field.
        _STRIP_ALL = {"reasoning", "_google_thought_sig"}
        _STRIP_GOOGLE = {"reasoning"}  # keep _google_thought_sig for Google

        if config.ACTIVE_BACKEND == "nim":
            clean = [{k: v for k, v in m.items() if k not in _STRIP_ALL} for m in messages]
            return self._make_request_nim(clean, temp)
        elif config.ACTIVE_BACKEND == "cloudflare":
            clean = [{k: v for k, v in m.items() if k not in _STRIP_ALL} for m in messages]
            return self._make_request_cloudflare(clean, temp)
        elif config.ACTIVE_BACKEND == "local":
            clean = [{k: v for k, v in m.items() if k not in _STRIP_ALL} for m in messages]
            return self._make_request_local(clean, temp)
        elif config.ACTIVE_BACKEND == "tokenrouter":
            clean = [{k: v for k, v in m.items() if k not in _STRIP_ALL} for m in messages]
            return self._make_request_tokenrouter(clean, temp)
        # Google: keep _google_thought_sig; _convert_messages_to_gemini reads it
        clean = [{k: v for k, v in m.items() if k not in _STRIP_GOOGLE} for m in messages]
        return self._make_request_google(clean, temp)

    # ══════════════════════════════════════════════════════════════════════════
    # GOOGLE GEMINI
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _convert_messages_to_gemini(messages: list) -> tuple:
        """
        Convert OpenAI-style message list to Gemini format.
        Returns (system_instruction_str, gemini_contents_list).
        Images: data URIs → inline_data Parts, http URLs → FileData Parts.

        Google thought_signature handling:
          If an assistant message carries a '_google_thought_sig' field (set
          in-memory by emperor_agent during a tool loop — never persisted to
          chat_history), we emit a thought Part with the signature BEFORE the
          regular text Part.  This lets Google restore its internal reasoning
          state when it reads the subsequent tool-result message, exactly the
          way the SDK's built-in chat history does it automatically.
          The dispatcher (_make_request) keeps this field intact for the Google
          backend while stripping it for all other backends.
        """
        system_instructions = []
        contents = []

        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str) and content:
                    system_instructions.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(content, str):
                parts = [types.Part.from_text(text=content)]
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block.get("text", "")))
                    elif block.get("type") == "image_url":
                        src = block.get("image_url", {}).get("url", "")
                        if src.startswith("data:"):
                            try:
                                header, b64data = src.split(",", 1)
                                mime = header.split(":")[1].split(";")[0]
                                raw  = base64.b64decode(b64data)
                                parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                            except Exception:
                                pass
                        elif src.startswith("http"):
                            # Bug 4: Google FileData only accepts File API URIs, not plain
                            # HTTP URLs.  Download the image and pass as inline bytes instead.
                            try:
                                import urllib.request as _urllib_req
                                with _urllib_req.urlopen(src, timeout=15) as _r:
                                    _raw = _r.read()
                                parts.append(types.Part.from_bytes(
                                    data=_raw, mime_type=mime_type_from_url(src)
                                ))
                            except Exception:
                                pass  # skip undownloadable URLs silently
            else:
                parts = [types.Part.from_text(text=str(content))]

            # ── Google thought_signature injection ────────────────────────────
            # For assistant messages produced mid-turn by the Google backend,
            # prepend a thought Part carrying the encrypted signature so Google
            # can restore its reasoning state when it processes the tool result.
            # Only present on in-memory mid-turn messages; never on messages
            # loaded from chat_history (those don't have this key).
            if gemini_role == "model":
                _sig = msg.get("_google_thought_sig")
                if _sig:
                    try:
                        thought_part = types.Part(
                            thought=True,
                            thought_signature=_sig,
                        )
                        parts = [thought_part] + parts
                    except Exception:
                        pass  # SDK version doesn't support thought_signature — skip

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        system_instruction = "\n\n".join(system_instructions) if system_instructions else None
        return system_instruction, contents

    def _make_request_google(self, messages, temp) -> dict:
        # max_retries = pool_size * 2  →  each key tried twice before giving up.
        # Falls back to 8 if pool is not set (e.g. single-key fallback mode).
        _pool = getattr(self, "_google_key_pool", [])
        max_retries = max(len(_pool) * 2, 2) if _pool else 8
        base_delay  = 3

        for attempt in range(max_retries):
            try:
                system_instruction, contents = self._convert_messages_to_gemini(messages)

                cfg = types.GenerateContentConfig(
                    max_output_tokens=config.EMPEROR_MAX_TOKENS,
                    temperature=temp,
                    thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
                    system_instruction=system_instruction,
                )

                full_thinking   = []
                full_content    = []
                thinking_active = False
                last_usage      = 0
                last_thought_sig = None   # encrypted reasoning state blob from Google

                for chunk in self.google_client.models.generate_content_stream(
                    model=config.GOOGLE_MODEL,
                    contents=contents,
                    config=cfg,
                ):
                    try:
                        parts = chunk.candidates[0].content.parts if chunk.candidates else []
                    except Exception:
                        parts = []

                    for part in parts:
                        if getattr(part, "thought", False):
                            # Capture the thought_signature (arrives on the last
                            # thought chunk, as the encrypted reasoning state blob).
                            _sig = getattr(part, "thought_signature", None)
                            if _sig:
                                last_thought_sig = _sig
                            text = getattr(part, "text", "") or ""
                            if text:
                                if not thinking_active:
                                    console.print("\n[dim cyan]Thinking:[/dim cyan]")
                                    thinking_active = True
                                console.print(text, end="", highlight=False, markup=False, style="cyan")
                                full_thinking.append(text)
                        else:
                            text = getattr(part, "text", "") or ""
                            if text:
                                full_content.append(text)

                    # usage_metadata is populated on the final streaming chunk
                    _um = getattr(chunk, "usage_metadata", None)
                    if _um:
                        last_usage = getattr(_um, "total_token_count", 0) or 0

                if thinking_active:
                    console.print()

                return {
                    "content":          "".join(full_content),
                    "thinking":         "".join(full_thinking) or None,
                    "usage":            last_usage,
                    "thought_signature": last_thought_sig,
                }

            except AttributeError as e:
                # Bug 25: client not initialised (missing API key) — no point retrying.
                return {"error": f"Google backend not initialised (missing API key?): {e}"}
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    # Rotate to the next key in the pool before backing off.
                    # _rotate_google_key() is a no-op when the pool has only 1 key.
                    self._rotate_google_key()
                    backoff_wait(attempt, base_delay, reason="Rate limit", max_attempts=max_retries)
                    continue
                if attempt == max_retries - 1:
                    return {"error": f"Request failed after {max_retries} attempts: {err}"}
                console.print(f"⚠️ [yellow]Request error: {e}. Retrying...[/yellow]")
                time.sleep(2)

        return {"error": "Max retries exceeded"}


    # ══════════════════════════════════════════════════════════════════════════
    # CLOUDFLARE
    # ══════════════════════════════════════════════════════════════════════════

    def _make_request_cloudflare(self, messages, temp) -> dict:
        # Bug 7: defense in depth — switch_backend() already refuses to activate
        # Cloudflare without an account ID, but guard the request path too in
        # case ACTIVE_BACKEND is ever set another way. Return the same
        # {"error": ...} dict shape every other backend in this file uses on
        # failure (the caller does `response_data.get("error")` and expects a
        # dict back, never an exception) instead of hitting the malformed
        # ".../accounts//ai/v1" URL.
        if not getattr(self, "cf_configured", bool(config.CLOUDFLARE_ACCOUNT_ID)):
            return {"error": "Cloudflare backend is not configured: CLOUDFLARE_ACCOUNT_ID is not set in .env."}

        max_retries = 3
        base_delay  = 3
        use_stream  = (config.THINKING_TYPE == "native")

        for attempt in range(max_retries):
            try:
                request_body = {
                    "model":       config.CLOUDFLARE_MODEL,
                    "messages":    messages,
                    "temperature": temp,
                    "max_tokens":  config.EMPEROR_MAX_TOKENS,
                    "stream":      use_stream,
                }

                if use_stream:
                    # Bug 22: without stream_options the server never sends usage
                    # metrics in the SSE stream, leaving telemetry at 0 tokens.
                    request_body["stream_options"] = {"include_usage": True}

                response = requests.post(
                    f"{self.cf_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.cf_api_key}",
                        "Content-Type":  "application/json",
                    },
                    json=request_body,
                    stream=use_stream,
                    timeout=120,
                )

                if response.status_code == 429:
                    backoff_wait(attempt, base_delay, reason="Cloudflare rate limit")
                    continue

                if response.status_code != 200:
                    return {"error": f"Cloudflare API error {response.status_code}: {response.text}"}

                if use_stream:
                    return self._parse_nim_streaming_response(response)

                data = response.json()
                message = data["choices"][0]["message"]
                
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)

                thinking = message.get("reasoning") or message.get("reasoning_content") or None

                if thinking:
                    console.print("\n[dim cyan]Thinking:[/dim cyan]")
                    console.print(thinking, highlight=False, markup=False)
                    console.print()

                return {
                    "content":  message.get("content", ""),
                    "thinking": thinking,
                    "usage": total_tokens,
                }

            except AttributeError as e:
                # Bug 25: client not initialised (missing API key) — no point retrying.
                return {"error": f"Cloudflare backend not initialised (missing API key?): {e}"}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"Cloudflare request failed: {str(e)}"}
                time.sleep(2)

        return {"error": "Max retries exceeded"}

    # ════════════════════════════════════════════════════════════════════════════
    # TOKEN ROUTER  (OpenAI-compatible)
    # ════════════════════════════════════════════════════════════════════════════

    def _make_request_tokenrouter(self, messages, temp) -> dict:
        """
        POST to Token Router's OpenAI-compatible /v1/chat/completions endpoint.

        Token Router is a routing proxy that exposes the same SSE streaming
        format as NIM, so we reuse _parse_nim_streaming_response directly
        (same pattern Cloudflare uses).
        """
        if not getattr(self, "tr_api_key", None):
            return {"error": "Token Router backend is not configured: TOKEN_ROUTER_API_KEY_FILE key is missing or empty."}

        max_retries = 3
        base_delay  = 3
        use_stream  = (config.THINKING_TYPE == "native")

        for attempt in range(max_retries):
            try:
                request_body = {
                    "model":       config.TOKEN_ROUTER_MODEL,
                    "messages":    messages,
                    "temperature": temp,
                    "max_tokens":  config.EMPEROR_MAX_TOKENS,
                    "stream":      use_stream,
                }

                if use_stream:
                    request_body["stream_options"] = {"include_usage": True}

                response = requests.post(
                    f"{config.TOKEN_ROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.tr_api_key}",
                        "Content-Type":  "application/json",
                    },
                    json=request_body,
                    stream=use_stream,
                    timeout=120,
                )

                if response.status_code == 429:
                    backoff_wait(attempt, base_delay, reason="Token Router rate limit")
                    continue

                if response.status_code != 200:
                    return {"error": f"Token Router API error {response.status_code}: {response.text}"}

                if use_stream:
                    return self._parse_nim_streaming_response(response)

                # ── Non-streaming ─────────────────────────────────────────────────────────────────────
                data    = response.json()
                message = data["choices"][0]["message"]

                usage        = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)

                thinking = message.get("reasoning") or message.get("reasoning_content") or None
                if thinking:
                    console.print("\n[dim cyan]Thinking:[/dim cyan]")
                    console.print(thinking, highlight=False, markup=False)
                    console.print()

                return {
                    "content":  message.get("content", ""),
                    "thinking": thinking,
                    "usage":    total_tokens,
                }

            except AttributeError as e:
                return {"error": f"Token Router backend not initialised (missing API key?): {e}"}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"Token Router request failed: {str(e)}"}
                time.sleep(2)

        return {"error": "Max retries exceeded"}

    # ══════════════════════════════════════════════════════════════════════════
    # LOCAL  (llama-server, OpenAI-compatible)
    # ══════════════════════════════════════════════════════════════════════════

    def _make_request_local(self, messages, temp) -> dict:
        """
        POST to a local llama-server instance (OpenAI-compatible /chat/completions).
        No retries/backoff — a local server either responds or it's not running;
        retrying a dead localhost connection just wastes the user's time.

        Timeout is intentionally very generous (not the 120-300s used by the
        cloud backends): local CPU/iGPU generation runs at single-digit tokens/sec,
        and EMPEROR_MAX_TOKENS allows responses long enough that a short timeout
        would abort a turn that was still legitimately generating.
        """
        headers = {"Content-Type": "application/json"}
        if getattr(self, "local_api_key", None):
            headers["Authorization"] = f"Bearer {self.local_api_key}"

        try:
            response = requests.post(
                f"{self.local_base_url}/chat/completions",
                headers=headers,
                json={
                    "model":       config.LOCAL_MODEL,
                    "messages":    messages,
                    "temperature": temp,
                    "max_tokens":  config.EMPEROR_MAX_TOKENS,
                    "stream":      False,
                },
                timeout=3600,  # generous — see docstring; local gen can be slow
            )
        except requests.exceptions.ConnectionError:
            return {"error": (
                "Could not reach local llama-server at "
                f"{self.local_base_url}. Is llama-server.exe running?"
            )}
        except requests.exceptions.Timeout:
            return {"error": (
                "Local server timed out after 1hr — generation may be stuck, "
                "or you're hitting a genuinely very long response. Check the "
                "llama-server console for activity before retrying."
            )}
        except Exception as e:
            return {"error": f"Local request failed: {str(e)}"}

        if response.status_code != 200:
            return {"error": f"Local server error {response.status_code}: {response.text[:300]}"}

        data    = response.json()
        message = data["choices"][0]["message"]
        
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        return {
            "content":  message.get("content", ""),
            "thinking": None,
            "usage": total_tokens,
        }

    def warm_local_system_prompt(self, system_prompt: str) -> None:
        """
        Fire-and-forget: push the (fixed, known-in-advance) system prompt into
        llama-server's KV cache before the user's first real message arrives.
        Only useful for turn 1 of a session — after that, llama-server keeps
        the growing conversation warm in its slot automatically between turns,
        so this should NOT be called on every turn, only once at session/backend-
        switch start (or /tool toggle, which changes the system prompt's shape).
        Any failure here is silent — it's an optimization, not a requirement,
        and must never block or error out the actual chat flow.

        Timeout is generous (not the short 120s originally used): the full
        tool-mode system prompt is ~1700+ tokens, which at this hardware's
        measured ~6 t/s prefill speed can take several minutes — a short
        timeout would abort the warm-up before it ever finished, wasting the
        prefill work already done rather than banking it.
        """
        try:
            _headers = {"Content-Type": "application/json"}
            # Bug 34: include auth token if the local server requires it,
            # matching the pattern already used in _make_request_local.
            _local_key = getattr(self, "local_api_key", None)
            if _local_key:
                _headers["Authorization"] = f"Bearer {_local_key}"
            requests.post(
                f"{self.local_base_url}/chat/completions",
                headers=_headers,
                json={
                    "model":      config.LOCAL_MODEL,
                    "messages":   [{"role": "system", "content": system_prompt}],
                    "max_tokens": 1,
                    "stream":     False,
                },
                timeout=900,  # generous — see docstring
            )
        except Exception:
            pass  # best-effort warm-up only — never surface this to the user

    # ══════════════════════════════════════════════════════════════════════════
    # NVIDIA NIM  (OpenAI-compatible)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _convert_to_native_tools(messages: list) -> list:
        """
        Send-time conversion: rewrite pseudo-tool user messages into proper
        OpenAI-compatible tool role messages before sending to NIM.

        The rest of the system (chat_history, display, other backends) continues
        to use the user/assistant two-role format unchanged. This function runs
        only inside _make_request_nim and operates on a shallow copy.

        Pattern detected (produced by _generate_with_tools in emperor_agent.py):

            [assistant]  content: "<reasoning>...</reasoning>\n\nLet me check.\n
                                   <view_lines file='x' .../>\n
                                   <bash command='ls'/>"]
            [user]       content: "[SYSTEM: Power execution started ...]\n\n
                                   [SYSTEM — view_lines result:]\n...result...\n\n
                                   [SYSTEM — bash[ls] result:]\n...result...\n\n
                                   [SYSTEM: All power results above. Continue.]"

        Converted to:

            [assistant]  content: (same prose + XML as before)
                         tool_calls: [
                           {id: "nim_s_a1b2", type: "function",
                            function: {name: "view_lines", arguments: '{"file": "x"}'}},
                           {id: "nim_s_c3d4", type: "function",
                            function: {name: "bash", arguments: '{"command": "ls"}'}},
                         ]
            [tool]       content: "...view_lines result..."
                         tool_call_id: "nim_s_a1b2"
            [tool]       content: "...bash result..."
                         tool_call_id: "nim_s_c3d4"

        Key probe findings that informed this design:
          - NIM requires tool_call_id on every tool message (hard 400 without it)
          - assistant content with pseudo-XML alongside tool_calls[] is accepted
          - <reasoning> blocks in assistant content alongside tool_calls[] is accepted
          - [SYSTEM --] prefix in tool result content is accepted
          - Orphan tool_call_id (no matching tool_calls[] on assistant) is accepted,
            but we synthesize proper tool_calls[] for maximum correctness.
        """
        import re
        import uuid
        import json as _json

        # Regex: match [SYSTEM — <label> result:]\n<content>
        # Captures everything between this header and the next result-block
        # header OR EOF. The lookahead is intentionally narrow: only
        # [SYSTEM — / [SYSTEM - (em-dash or hyphen) terminates a block, so
        # tool output that itself contains "[SYSTEM: ..." text is not falsely
        # split into a new block.
        _RESULT_BLOCK_RE = re.compile(
            r'\[SYSTEM\s*[\u2014-]\s*(.+?)\s+result:\]\n(.*?)'
            r'(?=\n\n\[SYSTEM\s*[\u2014-]|$)',
            re.DOTALL,
        )

        # Regex: parse pseudo-XML self-closing tags from assistant content.
        # Matches <tool_name attr="val" .../> and <tool_name>...</tool_name>
        _TAG_RE = re.compile(
            r'<(\w+)([^>]*?)(?:/>|>(.*?)</\1>)',
            re.DOTALL,
        )
        # Regex: parse attr="val" or attr='val' from a tag's attribute string
        _ATTR_RE = re.compile(r'(\w+)=["\']([^"\']*)["\']')

        def _parse_args(attr_str: str, inner: str) -> str:
            """Build a JSON arguments string from XML attributes + inner text."""
            args = dict(_ATTR_RE.findall(attr_str or ""))
            if inner and inner.strip():
                # Block-style tag: inner text is the primary argument.
                # Heuristic: use the key name 'content' unless a single attr
                # already exists (e.g. <view_lines file="x">...content...</view_lines>).
                if not args:
                    args["content"] = inner.strip()
                else:
                    # The inner body supplements existing attrs (rare case)
                    args["_body"] = inner.strip()
            return _json.dumps(args)

        result = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            # ── Detect a tool-result user message ─────────────────────────────
            is_tool_result_msg = (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and (
                    "[SYSTEM: Power execution started" in msg["content"]
                    or "[SYSTEM \u2014" in msg["content"]
                    or "[SYSTEM -" in msg["content"]
                )
            )

            if not is_tool_result_msg:
                result.append(msg)
                i += 1
                continue

            # ── Parse [SYSTEM — X result:] blocks from the user message ───────
            content = msg["content"]
            result_blocks = _RESULT_BLOCK_RE.findall(content)

            if not result_blocks:
                # No parseable result blocks — pass through unchanged.
                result.append(msg)
                i += 1
                continue

            # ── Back-patch the preceding assistant message with tool_calls[] ──
            # Find the most recent assistant message in what we've already built.
            asst_idx = None
            for j in range(len(result) - 1, -1, -1):
                if result[j].get("role") == "assistant":
                    asst_idx = j
                    break

            # Generate one tool_call_id per result block.
            ids = [f"nim_s_{uuid.uuid4().hex[:8]}" for _ in result_blocks]

            if asst_idx is not None:
                asst_content = result[asst_idx].get("content") or ""

                # Parse pseudo-XML tags from assistant content to build tool_calls[].
                # Filter out known non-tool wrapper tags (<reasoning>, <think>, <thinking>)
                # so positional matching with result_blocks is not thrown off.
                _NON_TOOL_TAGS = {"reasoning", "think", "thinking"}
                found_tags = [
                    t for t in _TAG_RE.findall(asst_content)
                    if t[0].lower() not in _NON_TOOL_TAGS
                ]

                tool_calls = []
                for k, (label, _result_content) in enumerate(result_blocks):
                    # label is e.g. "view_lines" or "bash[ls]" — extract tool name
                    fn_name = label.split("[")[0].strip()

                    # Try to find the matching XML tag (positional match by index)
                    arguments = "{}"
                    if k < len(found_tags):
                        tag_name, attr_str, inner = found_tags[k]
                        arguments = _parse_args(attr_str, inner)
                    else:
                        # No tag found at this position — synthesize minimal args
                        # by parsing the label e.g. "bash[ls -la]" → {command: "ls -la"}
                        bracket = label.find("[")
                        if bracket != -1:
                            arg_val = label[bracket + 1:].rstrip("]")
                            arguments = _json.dumps({"_arg": arg_val})

                    tool_calls.append({
                        "id":   ids[k],
                        "type": "function",
                        "function": {
                            "name":      fn_name,
                            "arguments": arguments,
                        },
                    })

                # Replace the assistant message with a copy that includes tool_calls[].
                # We keep the original content intact (pseudo-XML + reasoning blocks
                # are accepted by NIM alongside tool_calls[] — confirmed by probe).
                patched_asst = dict(result[asst_idx])
                patched_asst["tool_calls"] = tool_calls
                result[asst_idx] = patched_asst

            # ── Emit one tool message per result block ─────────────────────────
            for k, (label, result_content) in enumerate(result_blocks):
                result.append({
                    "role":         "tool",
                    "content":      result_content.strip(),
                    "tool_call_id": ids[k],
                })

            i += 1

        return result

    def _make_request_nim(self, messages, temp, model: str = None) -> dict:
        """
        POST to NIM's OpenAI-compatible endpoint.

        model defaults to config.NIM_TEXT_MODEL.
        Pass an explicit model string to override for a specific call.
        """
        # Convert pseudo-tool user messages → native tool role messages.
        # This is a send-time transformation only — chat_history and all other
        # backends remain on the two-role (user/assistant) format unchanged.
        messages = self._convert_to_native_tools(messages)

        max_retries = 10
        base_delay  = 3
        top_p       = config.EMPEROR_STATEFUL_TOP_P
        _model      = model or config.NIM_TEXT_MODEL
        # NIM streams reasoning via delta.reasoning / delta.reasoning_content.
        use_stream  = (config.THINKING_TYPE == "native")

        for attempt in range(max_retries):
            try:
                request_body = {
                    "model":       _model,
                    "messages":    messages,
                    "temperature": temp,
                    "top_p":       top_p,
                    "max_tokens":  config.EMPEROR_MAX_TOKENS,
                    "stream":      use_stream,
                }

                if use_stream:
                    # Bug 22: include_usage ensures the final SSE chunk carries
                    # token counts; without this the telemetry always shows 0.
                    request_body["stream_options"] = {"include_usage": True}

                # Model-specific request-body quirks (thinking-mode toggles etc.)
                # — see config.NIM_MODEL_QUIRKS for what each model needs and why.
                # Adding a new model's quirk is a config-only change; this loop
                # never needs to grow another elif.
                model_lower = _model.lower()
                for substr, extra_fields in config.NIM_MODEL_QUIRKS.items():
                    if substr in model_lower:
                        request_body.update(extra_fields)
                        break

                response = requests.post(
                    f"{self.nim_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.nim_api_key}",
                        "Content-Type":  "application/json",
                    },
                    json=request_body,
                    stream=use_stream,
                    timeout=900,
                )

                if response.status_code == 429:
                    backoff_wait(attempt, base_delay, reason="Rate limit (429)", max_attempts=max_retries)
                    continue

                if response.status_code >= 500:
                    backoff_wait(attempt, base_delay, reason=f"NIM server error {response.status_code}")
                    continue

                if response.status_code != 200:
                    return {"error": f"NIM API error {response.status_code}: {response.text}"}

                if use_stream:
                    return self._parse_nim_streaming_response(response)

                # ── Non-streaming ───────────────────────────────────────────
                data    = response.json()
                message = data["choices"][0]["message"]

                # NIM calls the field "reasoning"; official Moonshot API calls it
                # "reasoning_content". Try both for forward compatibility.
                thinking = message.get("reasoning") or message.get("reasoning_content") or None

                if thinking:
                    console.print("\n[dim cyan]Thinking:[/dim cyan]")
                    console.print(thinking, highlight=False, markup=False)
                    console.print()

                return {
                    "content":  message.get("content", ""),
                    "thinking": thinking,
                }

            except AttributeError as e:
                # Bug 25: nim_api_key or nim_base_url not set (missing key) — fail fast.
                return {"error": f"NIM backend not initialised (missing API key?): {e}"}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": f"NIM request failed after {max_retries} attempts: {str(e)}"}
                console.print(f"⚠️ [yellow]NIM request error: {e}. Retrying...[/yellow]")
                time.sleep(2)

        return {"error": "Max retries exceeded"}

    def _parse_nim_streaming_response(self, response) -> dict:
        """
        Consume NIM's SSE stream for native-thinking models.
        Prints reasoning_content live as it arrives, accumulates content silently.
        reasoning_content → thinking field; content → content field.
        """
        full_thinking   = []
        full_content    = []
        thinking_active = False
        total_tokens    = 0

        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                line = (
                    raw_line if isinstance(raw_line, str)
                    else raw_line.decode("utf-8", errors="ignore")
                )
                if line.strip() in ("data: [DONE]", "[DONE]"):
                    break
                if not line.startswith("data:"):
                    continue

                try:
                    chunk = json.loads(line[len("data:"):].strip())
                except Exception:
                    continue

                # Top-level usage field — present in the last SSE chunk from NIM
                _usage = chunk.get("usage") or {}
                if _usage.get("total_tokens"):
                    total_tokens = _usage["total_tokens"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Reasoning — stream live to terminal
                # NIM uses "reasoning"; Moonshot official uses "reasoning_content"
                reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if reasoning:
                    if not thinking_active:
                        console.print("\n[dim cyan]Thinking:[/dim cyan]")
                        thinking_active = True
                    console.print(reasoning, end="", highlight=False, markup=False)
                    full_thinking.append(reasoning)

                # Content — accumulate silently
                content = delta.get("content") or ""
                if content:
                    full_content.append(content)

        except Exception as e:
            console.print(f"\n⚠️ [yellow]NIM streaming parse error: {e}[/yellow]")

        if thinking_active:
            console.print()

        return {
            "content":  "".join(full_content),
            "thinking": "".join(full_thinking) or None,
            "usage":    total_tokens,
        }