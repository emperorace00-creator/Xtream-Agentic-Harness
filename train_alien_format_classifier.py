#!/usr/bin/env python3
"""
train_alien_format_classifier.py

Binary classifier for the "alien tool format" problem: the model emitted its
OWN native tool-call syntax (DeepSeek DSML, Qwen/Hermes <tool_call>, MiniMax
<minimax:tool_call>, an older DeepSeek-V3/R1 <|tool_call_begin|> block, etc.)
instead of this project's pseudo-XML "powers" format - as opposed to just
mentioning a tool name in prose, quoting a past result, or showing an example
in a code fence.

SCOPE (read this before wiring it in):
  This model answers ONE narrow question: "does this response contain a
  STRUCTURAL alien tool-call block?" It is meant to run only on the sliver of
  turns where your existing deterministic checks are ambiguous:
    - pseudo_calls == []                     (your XML parser found nothing)
    - self._current_tool_call_log == []      (no tool has already run this turn
                                                - rules out final-synthesis prose)
    - at least one known tool name appears   (cheap pre-filter, in this file
      somewhere in the text                    as `pref_known_tool_hits`)
  It does NOT handle "model said 'let me search' and then wrote nothing" -
  that's a phrase-presence check, not a format-detection problem, and belongs
  in plain code, not a classifier.

WHY THIS DESIGN (vs. Flash Lite / an LLM judge):
  This only needs to fire on the ambiguous slice of turns, so a ~50KB local
  model is enough. No API dependency in your retry path, no added latency,
  no external rate limit, and it improves for free as you log real
  production judgments (see RETRAINING ON REAL DATA below).

USAGE
  pip install scikit-learn scipy numpy joblib
  python train_alien_format_classifier.py --train
  python train_alien_format_classifier.py --predict "<tool_call>\n{\"name\": \"quick_search\", ...}"
  python train_alien_format_classifier.py --eval-examples   # prints the held-out test cases it got wrong

OUTPUT
  alien_format_clf.joblib   <- everything predict() needs, self-contained

RETRAINING ON REAL DATA (do this once you have production traffic)
  This bootstrap trains entirely on synthetic examples generated from
  documented tool-call formats (cited in comments below). That's a
  reasonable cold start, but it WILL have blind spots - most importantly,
  totally novel syntaxes from model families not covered here, and
  "code block explaining the format" negatives it wasn't shown a similar
  case for. Log every case this fires on (or nearly fires on) in
  production, along with the correct label once you know it, as JSONL:
      {"text": "...", "label": 1}
  one per line, into real_examples.jsonl, then:
      python train_alien_format_classifier.py --train --extra real_examples.jsonl
  Real examples are oversampled (see REAL_EXAMPLE_OVERSAMPLE below) so a
  few hundred real ones start to dominate the decision boundary over the
  synthetic bulk.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

RNG = random.Random(1234)

MODEL_PATH = Path(__file__).parent / "alien_format_clf.joblib"

# GUESS, NOT VERIFIED - point this at wherever your real chat history is
# actually stored (in emperor_agent.py that's `config.CHAT_HISTORY_FILE` +
# `self.global_history.write_turn(...)`, which this script has not seen the
# internals of). Mining is OFF by default (opt in with --mine) until you've
# either confirmed this path/format is right or edited it below.
CHAT_DIR   = Path(__file__).parent / "chat_histories"
REAL_EXAMPLE_OVERSAMPLE = 8   # each real logged example is repeated this many
                              # times relative to one synthetic example, so
                              # a modest amount of real data can outweigh the
                              # synthetic bulk once you start collecting it.

# ════════════════════════════════════════════════════════════════════════
# YOUR PROJECT'S TOOL SCHEMA
# Mirrors GROUP_TOOLS_UNION / STRUCTURED / SIMPLE_PARAM in
# core_tool_definitions.py + emperor_agent.py. Edit this if your tool list
# changes - it drives both the "your own correct format" negatives and the
# realistic args used inside the alien-format positives.
# ════════════════════════════════════════════════════════════════════════

TOOLS = {
    "quick_search":     {"kind": "simple", "param": "query"},
    "url_search":        {"kind": "attrs",  "attrs": ["url", "context"]},
    "view_lines":        {"kind": "struct", "fields": ["file", "start", "end", "context"]},
    "search_in_file":    {"kind": "struct", "fields": ["file", "pattern", "regex", "context_lines", "max_results"]},
    "workspace_search":  {"kind": "struct", "fields": ["query", "file_filter", "semantic"]},
    "search_history":    {"kind": "struct", "fields": ["query", "top_k"]},
    "str_replace":       {"kind": "struct", "fields": ["file", "old_str", "new_str", "count"]},
    "ingest_chat":       {"kind": "simple", "param": "filename"},
    "ingest_pdf":        {"kind": "simple", "param": "filename"},
    "ingest_text":       {"kind": "simple", "param": "filename"},
    "doc_search":        {"kind": "struct", "fields": ["query", "top_k"]},
    "search_semantic_scholar": {"kind": "struct", "fields": ["query", "limit", "year"]},
    "bash":              {"kind": "simple", "param": "command"},
    "show_image":        {"kind": "simple", "param": "file"},
}
TOOL_NAMES = list(TOOLS.keys())

QUERY_POOL = [
    "asyncio gather vs wait difference", "latest Indian Army TES notification",
    "current inflation rate india", "photosynthesis process steps",
    "react useEffect cleanup function", "how to reverse a linked list",
    "population of tokyo 2026", "best practices for postgres indexing",
]
FILE_POOL = ["api/start.py", "config.py", "src/utils/parser.py", "README.md", "tests/test_agent.py", "notes.txt"]
CMD_POOL = ["python start.py", "pytest tests/ -v", "ls -la", "git status", "npm run build"]
URL_POOL = ["https://docs.python.org/3/library/asyncio-task.html", "https://arxiv.org/abs/2504.07164"]


def _val_for(tool: str, field: str) -> str:
    if field in ("query",):
        return RNG.choice(QUERY_POOL)
    if field in ("file", "file_filter"):
        return RNG.choice(FILE_POOL)
    if field in ("command",):
        return RNG.choice(CMD_POOL)
    if field in ("url",):
        return RNG.choice(URL_POOL)
    if field in ("filename",):
        return RNG.choice(["paper.pdf", "gemini_chat.txt", "notes.txt"])
    if field in ("old_str",):
        return "def foo():\n    pass"
    if field in ("new_str",):
        return "def foo():\n    return 42"
    if field in ("start", "end", "top_k", "count", "max_results", "context_lines", "limit"):
        return str(RNG.randint(1, 200))
    if field in ("regex", "semantic"):
        return RNG.choice(["true", "false"])
    if field == "year":
        return RNG.choice(["2020-", "2021-2023", "2019-", ""])
    if field == "context":
        return "gather vs wait"
    if field == "pattern":
        return "def .*("
    return "value"


def random_args(tool: str) -> dict:
    spec = TOOLS[tool]
    if spec["kind"] == "simple":
        return {spec["param"]: _val_for(tool, spec["param"])}
    if spec["kind"] == "attrs":
        fields = RNG.sample(spec["attrs"], k=RNG.randint(1, len(spec["attrs"])))
        return {f: _val_for(tool, f) for f in fields}
    # struct
    required = spec["fields"][: RNG.randint(1, len(spec["fields"]))]
    return {f: _val_for(tool, f) for f in required}


# ════════════════════════════════════════════════════════════════════════
# ALIEN FORMAT RENDERERS  (positives)
# Each takes (tool_name, args) -> rendered native-format string.
# Sources, so the shapes are real and not guessed:
#   - DeepSeek V3.2/V4 DSML:   docs.vllm.ai/.../parser/deepseek_v32
#   - DeepSeek V3/R1 (older):  docs.vllm.ai/.../deepseekv3_tool_parser
#   - Qwen/Hermes <tool_call>: qwen.readthedocs.io/framework/function_call
#   - Qwen3-Coder nested tags: github.com/morph-labs/hermes-agent-fork
#   - Qwen legacy ✿FUNCTION✿: github.com/QwenLM/Qwen-Agent
#   - MiniMax M2 bracket:      huggingface.co/MiniMaxAI/MiniMax-M2/docs/tool_calling_guide.md
#   - MiniMax M2 XML:          huggingface.co/MiniMaxAI/MiniMax-M2/docs/tool_calling_guide.md
#   - Claude XML:              docs.anthropic.com/en/docs/tool-use/how-to-use-tools
#   - Gemma/FunctionGemma:     ai.google.dev/gemma/docs/function-calling
#   - Gemma 4 thinking:        <|tool_call> special token observed in prod
#   - Nemotron (Llama-based):  build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1
#   - Nemotron (native):       huggingface.co/nvidia/Nemotron-Mini-4B-Instruct
#   - Plain JSON blob:         OpenAI-style dump without any wrapper
#   - YAML-style:              key-value flat format observed in practice
#   - Pipe-delimited:          |tool| name |key| value |/tool| style
#   - ASCII-pipe DSML:         <|DSML|> with ASCII | not fullwidth ｜
#   - Kimi native - NOT included here on purpose: your code already parses
#     it correctly (_parse_kimi_native_tools), so it's not an "alien" case.
# ════════════════════════════════════════════════════════════════════════

def _render_deepseek_dsml(tool, args):
    # Real DeepSeek DSML uses U+FF5C FULLWIDTH VERTICAL LINE (｜), not ASCII |.
    # The outer wrapper alternates between the two documented tag names so the
    # classifier sees both variants during training.
    outer = RNG.choice(["tool_calls", "function_calls"])
    params = "\n".join(
        f'<｜DSML｜parameter name="{k}" string="true">{v}</｜DSML｜parameter>' for k, v in args.items()
    )
    return (
        f'<｜DSML｜{outer}>\n<｜DSML｜invoke name="{tool}">\n{params}\n'
        f'</｜DSML｜invoke>\n</｜DSML｜{outer}>'
    )


def _render_deepseek_v3_old(tool, args):
    return (
        f'<|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>{tool}\n'
        f'```json\n{json.dumps(args)}\n```<|tool▁call▁end|><|tool▁calls▁end|>'
    )


def _render_hermes(tool, args):
    return f'<tool_call>\n{json.dumps({"name": tool, "arguments": args})}\n</tool_call>'


def _render_qwen3_coder(tool, args):
    params = "".join(f'<parameter={k}>{v}</parameter>' for k, v in args.items())
    return f'<tool_call><function={tool}>{params}</function></tool_call>'


def _render_minimax(tool, args):
    params = "\n".join(f'<parameter name="{k}">{v}</parameter>' for k, v in args.items())
    return f'<minimax:tool_call>\n<invoke name="{tool}">\n{params}\n</invoke>\n</minimax:tool_call>'


def _render_raw_json_leak(tool, args):
    # A model occasionally just dumps the OpenAI-style function_call object
    # as plain text instead of any wrapper at all.
    return json.dumps({"type": "function", "function": {"name": tool, "arguments": args}})


def _render_pythonic(tool, args):
    # Newer "pythonic" tool-call convention (e.g. Olmo3-style), wrapped in
    # <function_calls> tags.
    arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
    return f'<function_calls>\n{tool}({arg_str})\n</function_calls>'


def _render_glm(tool, args):
    # GLM-4.5/4.6: <tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value>...
    # Note the outer tag NAME collides with Hermes's <tool_call> - the
    # internal structure (arg_key/arg_value pairs, no JSON) is what
    # distinguishes it, which is a genuinely useful case for the classifier
    # to see (same wrapper token, different internals).
    # Source: github.com/zai-org/GLM-4.5/blob/main/resources/glm_4.6_tir_guide.md
    params = "".join(f'<arg_key>{k}</arg_key><arg_value>{v}</arg_value>' for k, v in args.items())
    return f'<tool_call>{tool}{params}</tool_call>'


def _render_claude_xml(tool, args):
    # Claude 3.5/3.7/4.x native XML: <function_calls><invoke name="..."><parameter ...>
    # Source: docs.anthropic.com/en/docs/tool-use/how-to-use-tools
    params = "\n    ".join(f'<parameter name="{k}">{v}</parameter>' for k, v in args.items())
    return f'<function_calls>\n  <invoke name="{tool}">\n    {params}\n  </invoke>\n</function_calls>'


def _render_minimax_bracket(tool, args):
    # MiniMax M2/M2.5/M3 bracket style: [TOOL_CALL]{...}[/TOOL_CALL]
    # Source: huggingface.co/MiniMaxAI/MiniMax-M2/docs/tool_calling_guide.md
    payload = json.dumps({"tool": tool, "args": args}, indent=2)
    return f'[TOOL_CALL]\n{payload}\n[/TOOL_CALL]'


def _render_gemma(tool, args):
    # FunctionGemma / Gemma 3: <start_function_call>call:func{...}<end_function_call>
    # Source: ai.google.dev/gemma/docs/function-calling
    arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
    return f'<start_function_call>\ncall:{tool}{{{arg_str}}}\n<end_function_call>'


def _render_gemma4_thinking(tool, args):
    # Gemma 4 thinking models: <|tool_call>call:func{args}<tool_call|>
    arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
    return f'<|tool_call>call:{tool}{{{arg_str}}}<tool_call|>'


def _render_nemotron_llama(tool, args):
    # Llama-based Nemotron (Ultra/Super): <|python_tag|>func(args)<|eom_id|>
    # Source: build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1
    arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
    if RNG.choice([True, False]):
        return f'<|python_tag|>{tool}({arg_str})<|eom_id|>'
    else:
        return f'<|python_tag|>{json.dumps({"name": tool, "parameters": args})}<|eom_id|>'


def _render_nemotron_native(tool, args):
    # Native NVIDIA Nemotron-4/Mini: <extra_id_1>...<toolcall>{...}</toolcall>
    # Source: huggingface.co/nvidia/Nemotron-Mini-4B-Instruct
    payload = json.dumps({"name": tool, "arguments": args})
    return f'<extra_id_1>Assistant\n<toolcall>\n{payload}\n</toolcall>'


def _render_qwen_legacy(tool, args):
    # Legacy Qwen-Agent: ✿FUNCTION✿: name\n✿ARGS✿: {...}\n✿RESULT✿:
    # Source: github.com/QwenLM/Qwen-Agent
    return f'\u273fFUNCTION\u273f: {tool}\n\u273fARGS\u273f: {json.dumps(args)}\n\u273fRESULT\u273f:'


def _render_plain_json(tool, args):
    # Plain JSON blob with 'tool', 'name', or 'function_name' key - no wrapper.
    style = RNG.choice(["tool_key", "name_key", "function_name_key"])
    indent = RNG.choice([None, 2])
    if style == "tool_key":
        d = {"tool": tool}
        d.update(args)
        return json.dumps(d, indent=indent)
    elif style == "name_key":
        return json.dumps({"name": tool, "arguments": args}, indent=indent)
    else:
        return json.dumps({"function_name": tool, "parameters": args}, indent=indent)


def _render_yaml_style(tool, args):
    # YAML-like key-value flat format observed in practice
    lines = [f"tool: {tool}"]
    for k, v in args.items():
        safe_v = str(v)
        lines.append(f'{k}: "{safe_v}"' if " " in safe_v else f"{k}: {safe_v}")
    return "\n".join(lines)


def _render_pipe_delimited(tool, args):
    # Pipe-delimited: |tool| name |key| value |/tool|
    inner = " ".join(f"|{k}| {v}" for k, v in args.items())
    return f"|tool| {tool} {inner} |/tool|"


def _render_dsml_ascii(tool, args):
    # ASCII-pipe variant of DeepSeek DSML (<|DSML|> with | not ｜)
    # Observed when models respond without fullwidth unicode
    params = ", ".join(f'{k}="{v}"' for k, v in args.items())
    return f'<|DSML|>{{invoke tool_name={tool}, {params}}}<|DSML|>'


def _render_yaml_multiline(tool, args):
    # YAML-style with dashes/blocks, more structured than _render_yaml_style
    lines = ["---", f"tool: {tool}", "arguments:"]
    for k, v in args.items():
        lines.append(f"  {k}: \"{v}\"")
    lines.append("---")
    return "\n".join(lines)


def _render_markdown_bold(tool, args):
    # Markdown-style bold labels: **Tool:** name **Query:** ...
    parts = [f"**Tool:** {tool}"]
    for k, v in args.items():
        parts.append(f"**{k.capitalize()}:** {v}")
    return "  \n".join(parts)


ALIEN_RENDERERS = [
    _render_deepseek_dsml,
    _render_deepseek_v3_old,
    _render_hermes,
    _render_qwen3_coder,
    _render_minimax,
    _render_raw_json_leak,
    _render_pythonic,
    _render_glm,
    # New formats added from cross-model research:
    _render_claude_xml,
    _render_minimax_bracket,
    _render_gemma,
    _render_gemma4_thinking,
    _render_nemotron_llama,
    _render_nemotron_native,
    _render_qwen_legacy,
    _render_plain_json,
    _render_yaml_style,
    _render_pipe_delimited,
    _render_dsml_ascii,
    _render_yaml_multiline,
    _render_markdown_bold,
]


def _truncate_mid_stream(s: str) -> str:
    """
    Simulate a real, commonly-reported failure mode: the model starts
    emitting an alien tool call and generation stops (hits EOS / max
    tokens / gets cut) before the closing tag - e.g. the llama.cpp issue
    where GLM/MiniMax/Qwen3-Coder models sometimes emit only a partial
    tool-call block. Cutting at a random point (never past the halfway
    mark, so the fragment is still recognizably alien) trains the
    classifier not to require a complete, well-closed block to be
    confident.
    """
    cut = RNG.randint(max(1, len(s) // 4), max(2, len(s) // 2))
    return s[:cut]

# ════════════════════════════════════════════════════════════════════════
# YOUR PROJECT'S OWN (correct) FORMAT - used for negatives so the model
# learns "this shape is fine", mirroring _parse_pseudo_tools in
# emperor_agent.py.
# ════════════════════════════════════════════════════════════════════════

def _render_own_format(tool, args):
    spec = TOOLS[tool]
    if spec["kind"] == "simple":
        return f'<{tool}>{list(args.values())[0]}</{tool}>'
    if spec["kind"] == "attrs":
        attrs = " ".join(f'{k}="{v}"' for k, v in args.items())
        return f'<{tool} {attrs}/>'
    inner = "".join(f'<{k}>{v}</{k}>' for k, v in args.items())
    return f'<{tool}>{inner}</{tool}>'


# ════════════════════════════════════════════════════════════════════════
# SURROUNDING PROSE POOLS - real responses aren't bare tags, they have
# narrative text around them. Sampling this in gives the vectorizer
# something realistic to generalize over instead of memorizing 7 fixed
# templates.
# ════════════════════════════════════════════════════════════════════════

LEAD_IN_TOOLCALL = [
    "Let me check that for you.", "I'll look into this now.",
    "Sure, give me a moment.", "Checking the file first.",
    "I need to verify this before answering.", "",
]

FINAL_SYNTHESIS_PROSE = [
    "I did multiple {t} calls, and I found this gold info: the answer is that "
    "the correct approach is to use dependency injection here.",
    "Based on the {t} results above, the tests passed and no further changes are needed.",
    "I already called {t} earlier in this turn, so here's the final summary.",
    "After running {t}, the output confirms the config is valid.",
    "To recap what I found using {t}: the root cause was a missing import.",
    "So {t} turned up three relevant hits — the second one answers your question directly.",
    "Now that {t} has run, here's what changed and why it should fix the bug.",
    "Given what {t} returned, I'd recommend going with option B.",
    "That's everything from {t} — nothing else needs checking here.",
    "Thanks to {t}, I can confirm the file was updated correctly.",
    "{t} came back empty, so I'm answering from general knowledge instead.",
    "I ran {t} twice with different queries and both agree on the same conclusion.",
]

EXPLAIN_FORMAT_PROSE = [
    "For reference, DeepSeek's native tool syntax looks like this:\n```\n{alien}\n```\n"
    "whereas this project expects the XML powers format instead.",
    "Here's an example of what {family} emits when it wants to call a tool:\n```\n{alien}\n```\n"
    "You don't need to write that yourself — just use the tag format described above.",
    "Just so you know what to watch for, a malformed response might look like:\n```\n{alien}\n```",
    "The old parser used to choke on strings such as `{alien_inline}` before we added the fix.",
    "You asked what {family}'s tool-call format looks like — here you go:\n```\n{alien}\n```",
    "If you're curious, the raw output for that model family is:\n```\n{alien}\n```\n"
    "That's what the nudge-and-retry logic is supposed to catch.",
    "Quoting the earlier bug report for context:\n```\n{alien}\n```\n"
    "which is exactly the malformed case we discussed.",
    "In case it helps debugging, this is what leaked into the transcript:\n\n{alien}\n\n"
    "— that's the artifact we need to filter out, not a real request.",
]

PROSE_MENTION_NO_TAG = [
    "I could use {t} to check that, but I don't think it's necessary here.",
    "The {t} power is useful for exactly this kind of question.",
    "You could also try {t} manually if you want to double check.",
    "Let me use the tool to help you with that.",
    "I'll search for that now.",
    "Do you want me to run {t} on this, or would you rather I answer directly?",
    "Normally I'd reach for {t} here, but since you've already given me the details, I don't need to.",
    "One option is calling {t} — let me know if you'd like me to.",
    "I was about to call {t} but realized I already know the answer.",
    "{t} would help confirm this, though it's optional given what you've told me.",
]

UNRELATED_PROSE = [
    "The capital of France is Paris, and it has a population of roughly 2.1 million.",
    "Here's a summary of the meeting notes you shared earlier.",
    "Sure — recursion is when a function calls itself to solve a smaller "
    "instance of the same problem.",
    "That error usually means the virtual environment isn't activated.",
    "Great question! The short answer is it depends on your use case.",
    "Thanks for clarifying — here's the revised plan based on what you said.",
    "That's a reasonable approach, though there's a simpler way to write it.",
    "I don't have enough context to answer that confidently — could you share the file?",
    "Here's the corrected version of your function with the off-by-one fixed.",
    "In short: yes, but only if the dependency is pinned to that version.",
    "Happy to help — what's the current behavior you're seeing versus what you expect?",
]

SYSTEM_RESULT_ECHO = [
    "[SYSTEM — {t}[{q}] result:]\nHere are the top 3 results discussing the topic in detail.",
    "[SYSTEM: All power results above. Continue.]",
    "[SYSTEM — {t} result:]\n[exit 0]\nAll 12 tests passed.",
]

# Hard negatives: JSON/YAML that is NOT a tool call (code, config, examples)
JSON_YAML_PROSE = [
    'Here\'s the config format:\n```json\n{{"tool": "hammer", "size": 12}}\n```\nNote: this is not a tool call.',
    'The response schema looks like:\n```\n{{"name": "{t}", "status": "ok"}}\n```',
    'In YAML, the config would be:\n```yaml\ntool: linter\nquery: check all\n```',
    'The JSON payload has fields like "tool" and "name" that map to the internal ID.',
    'Example API response:\n{{"function_name": "handler", "result": 42}}',
    'Here\'s what the test fixture expects:\n```\ntool: {t}\nquery: test input\n```\nThis is just test data.',
    'The OpenAI format returns:\n```json\n{{"type": "function", "function": {{"name": "{t}", "arguments": {{}}}}}}\n```',
    'Your config.yaml should have:\n```\ntool: webpack\nquery: build\n```',
]


def _wrap_code_fence(s: str) -> str:
    return f"```\n{s}\n```"


# ════════════════════════════════════════════════════════════════════════
# EXAMPLE GENERATION
# ════════════════════════════════════════════════════════════════════════

def gen_positive() -> str:
    """An alien-format tool call, optionally mid-turn after correct calls."""
    tool = RNG.choice(TOOL_NAMES)
    args = random_args(tool)
    renderer = RNG.choice(ALIEN_RENDERERS)
    alien = renderer(tool, args)
    if RNG.random() < 0.15:
        alien = _truncate_mid_stream(alien)
    lead = RNG.choice(LEAD_IN_TOOLCALL)

    pieces = []
    if lead:
        pieces.append(lead)

    # ~30% of the time: 1-2 correctly-formatted calls happened first, then
    # the model switches to alien format mid-turn - the hardest positive.
    if RNG.random() < 0.3:
        n_correct = RNG.randint(1, 2)
        for _ in range(n_correct):
            ctool = RNG.choice(TOOL_NAMES)
            pieces.append(_render_own_format(ctool, random_args(ctool)))
        pieces.append("Now let me also check this:")

    pieces.append(alien)
    return "\n".join(pieces)


def gen_negative() -> str:
    """One of several hard-negative families, roughly evenly weighted."""
    kind = RNG.choice([
        "own_format", "own_format", "own_format",   # weighted up: this is
                                                       # the single most
                                                       # important negative
        "final_synthesis", "final_synthesis",
        "explain_format", "explain_format",
        "prose_mention_no_tag",
        "unrelated",
        "system_echo",
        "json_yaml_prose", "json_yaml_prose",   # JSON/YAML that is NOT a
                                                  # tool call - teaches the
                                                  # classifier that mere JSON
                                                  # structure isn't enough
    ])

    if kind == "own_format":
        tool = RNG.choice(TOOL_NAMES)
        args = random_args(tool)
        lead = RNG.choice(LEAD_IN_TOOLCALL)
        return (lead + "\n" if lead else "") + _render_own_format(tool, args)

    if kind == "final_synthesis":
        tool = RNG.choice(TOOL_NAMES)
        return RNG.choice(FINAL_SYNTHESIS_PROSE).format(t=tool)

    if kind == "explain_format":
        tool = RNG.choice(TOOL_NAMES)
        args = random_args(tool)
        renderer = RNG.choice(ALIEN_RENDERERS)
        alien = renderer(tool, args)
        family = renderer.__name__.replace("_render_", "").replace("_", " ")
        template = RNG.choice(EXPLAIN_FORMAT_PROSE)
        return template.format(alien=alien, family=family, alien_inline=alien.replace("\n", " "))

    if kind == "prose_mention_no_tag":
        tool = RNG.choice(TOOL_NAMES)
        return RNG.choice(PROSE_MENTION_NO_TAG).format(t=tool)

    if kind == "system_echo":
        tool = RNG.choice(TOOL_NAMES)
        return RNG.choice(SYSTEM_RESULT_ECHO).format(t=tool, q=RNG.choice(QUERY_POOL))

    if kind == "json_yaml_prose":
        tool = RNG.choice(TOOL_NAMES)
        return RNG.choice(JSON_YAML_PROSE).format(t=tool)

    # unrelated
    return RNG.choice(UNRELATED_PROSE)


def build_synthetic_dataset(n_per_class: int = 1500):
    texts, labels = [], []
    for _ in range(n_per_class):
        texts.append(gen_positive())
        labels.append(1)
    for _ in range(n_per_class):
        texts.append(gen_negative())
        labels.append(0)
    return texts, labels


def load_real_examples(path: Path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def mine_from_chat_histories() -> tuple[list, list]:
    """
    Scan chat_histories/*.jsonl for assistant messages that contain known tool
    names and classify them as alien-format positives (label 1) or clean
    negatives (label 0). Returns (texts, labels) ready for oversampling.

    Only auto-labels the cases where a fixed regex is actually reliable
    ground truth:
      - has_correct_xml  -> 0   (matches your real parser's tag shape)
      - has_sys_result    -> 0   (matches your own result-echo scaffolding)
      - has_alien_marker (and NOT Kimi's real format) -> 1

    Anything else - a known tool name present, but none of the above match -
    is NOT auto-labeled. That "I don't recognize this shape" bucket is
    exactly where a genuinely novel alien format would land, and trusting
    the same regex to call it "clean" would silently teach the classifier
    to ignore the one case mining is supposed to help it generalize to.
    Those go to needs_review.jsonl instead, for you to label by hand before
    they ever enter training.

    Also explicitly excludes Kimi's native <|tool_call_begin|> format from
    the positive bucket - your code already parses and executes that
    correctly (_parse_kimi_native_tools), so labeling it "alien / needs a
    nudge" would teach the classifier to flag a working code path.

    Runs only when --mine is passed. Safe to call even if chat_histories/
    does not exist (returns empty lists).
    """
    if not CHAT_DIR.exists():
        return [], []

    jsonl_files = sorted(
        p for p in CHAT_DIR.glob("*.jsonl")
        if not p.name.endswith(".idx.json")
    )
    if not jsonl_files:
        return [], []

    texts, labels = [], []
    review_rows = []
    n_pos = n_neg = n_review = n_kimi_skipped = 0

    own_xml_re = re.compile(
        r'<(' + '|'.join(re.escape(t) for t in TOOL_NAMES) + r')(\s[^>]*)?(>|/>)'
    )
    # Kimi's real format, mirroring _parse_kimi_native_tools's own regex -
    # a legitimately supported call, not an alien one.
    kimi_native_re = re.compile(
        r'<\|tool_call_begin\|>\s*functions\.\w+(?::\d+)?\s*<\|tool_call_argument_begin\|>'
    )

    for path in jsonl_files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            visible = _strip_thinking(content)

            if not re.search(
                r'\b(' + '|'.join(re.escape(t) for t in TOOL_NAMES) + r')\b',
                visible
            ):
                continue

            is_kimi = bool(kimi_native_re.search(visible))
            has_correct_xml  = bool(own_xml_re.search(visible))
            has_alien_marker = bool(_ALIEN_MARKER_RE.search(visible))
            has_sys_result   = bool(_SYSTEM_RESULT_RE.search(visible))

            if is_kimi:
                # Legitimate, already-handled format - exclude entirely,
                # don't feed it in as either label.
                n_kimi_skipped += 1
                continue
            elif has_alien_marker:
                texts.append(visible)
                labels.append(1)
                n_pos += 1
            elif has_correct_xml or has_sys_result:
                texts.append(visible)
                labels.append(0)
                n_neg += 1
            else:
                # Ambiguous - don't guess. Surface for human labeling instead.
                review_rows.append(visible)
                n_review += 1

    if review_rows:
        review_path = Path(__file__).parent / "needs_review.jsonl"
        with open(review_path, "a") as f:
            for t in review_rows:
                f.write(json.dumps({"text": t, "label": None}) + "\n")
        print(f"  {n_review} ambiguous example(s) written to {review_path.name} "
              f"— label them by hand, then pass via --extra")

    print(f"  Mined {n_pos} confident positive(s) + {n_neg} confident negative(s) "
          f"from {len(jsonl_files)} chat file(s) (excluded {n_kimi_skipped} legitimate "
          f"Kimi call(s), oversampling x{REAL_EXAMPLE_OVERSAMPLE})")
    return texts, labels


# ════════════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES
# Char n-grams alone pick up the special-token morphology reasonably well
# (`<|`, `:tool_call>`, `<invoke`, quote styles...), but these few explicit
# booleans directly encode the highest-value structural signals so the
# model doesn't have to re-derive them from scratch on ~3000 examples.
# ════════════════════════════════════════════════════════════════════════

_KNOWN_TOOL_RE = re.compile(r'\b(' + '|'.join(re.escape(t) for t in TOOL_NAMES) + r')\b')
_ALIEN_MARKER_RE = re.compile(
    # Structural markers that are unambiguous signs of a native tool-call block.
    # ｜ = U+FF5C fullwidth vertical line (actual DeepSeek DSML production output)
    r'('
    # ── DeepSeek ──────────────────────────────────────────────────────────
    r'<\|tool[▁_]'               # old V3/R1: <|tool▁calls▁begin|
    r'|tool_call_begin'           # old V3/R1 inner marker
    r'|<\|DSML\|'                 # DSML ASCII-pipe variant
    r'|<｜DSML｜'                  # DSML fullwidth ｜
    # ── Qwen / Hermes ─────────────────────────────────────────────────────
    r'|<tool_call>'               # Hermes, Qwen2.5, Qwen3, GLM
    r'|✿FUNCTION✿'             # legacy Qwen-Agent ✿FUNCTION✿
    # ── MiniMax ───────────────────────────────────────────────────────────
    r'|<minimax:tool_call>'       # MiniMax XML style
    r'|\[TOOL_CALL\]'             # MiniMax bracket style
    # ── Claude ────────────────────────────────────────────────────────────
    r'|<invoke\s+name='           # Claude / Anthropic XML
    r'|function_calls>'           # Claude / Pythonic wrapper
    # ── Gemma / FunctionGemma ─────────────────────────────────────────────
    r'|<start_function_call>'     # Gemma 3 / FunctionGemma
    r'|<\|tool_call>'             # Gemma 4 thinking token
    r'|call:[a-z_]+\{'            # Gemma call:func{...} syntax
    # ── Nemotron ──────────────────────────────────────────────────────────
    r'|<\|python_tag\|>'          # Llama-based Nemotron
    r'|<toolcall>'                # Native Nemotron-4/Mini
    r'|<extra_id_'                # Native Nemotron system markers
    # ── Misc / generic ────────────────────────────────────────────────────
    r'|<function='                # Qwen3-Coder nested
    r'|\|tool\|'                  # pipe-delimited format
    r'|"tool"\s*:'                # plain JSON blob with "tool" key
    r'|"function_name"\s*:'       # plain JSON blob with "function_name" key
    r')'
)
_OWN_TAG_RE = re.compile(
    r'</?(' + '|'.join(re.escape(t) for t in TOOL_NAMES) + r')(\s|>|/>)'
)
_SYSTEM_MARKER_RE = re.compile(r'\[SYSTEM[:\s—-]')
_SYSTEM_RESULT_RE = re.compile(r'\[SYSTEM[:\s—\-].*?result', re.IGNORECASE)
_CODE_FENCE_RE    = re.compile(r'```')
_THINK_BLOCK_RE   = re.compile(r'<think>.*?</think>',         re.DOTALL)
_REASONING_BLK_RE = re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove private reasoning blocks that are not part of visible model output."""
    text = _THINK_BLOCK_RE.sub('', text)
    text = _REASONING_BLK_RE.sub('', text)
    return text.strip()


def engineered_features(texts):
    rows = []
    for t in texts:
        n_known = len(_KNOWN_TOOL_RE.findall(t))
        n_alien_marker = len(_ALIEN_MARKER_RE.findall(t))
        has_own_tag = 1.0 if _OWN_TAG_RE.search(t) else 0.0
        has_system_marker = 1.0 if _SYSTEM_MARKER_RE.search(t) else 0.0
        n_fences = len(_CODE_FENCE_RE.findall(t))
        in_fence = 1.0 if n_fences >= 2 else 0.0
        rows.append([
            min(n_known, 5) / 5.0,
            min(n_alien_marker, 5) / 5.0,
            has_own_tag,
            has_system_marker,
            in_fence,
            len(t) / 1000.0,
        ])
    return csr_matrix(np.array(rows, dtype=np.float64))


class AlienFormatDetector:
    """Self-contained: vectorizer + engineered features + logistic regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), max_features=20000, sublinear_tf=True
        )
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)

    def _features(self, texts, fit=False):
        X_tfidf = self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        X_eng = engineered_features(texts)
        return hstack([X_tfidf, X_eng]).tocsr()

    def fit(self, texts, labels):
        X = self._features(texts, fit=True)
        self.clf.fit(X, labels)

    def predict_proba(self, texts):
        X = self._features(texts, fit=False)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, text: str, threshold: float = 0.65):
        # Default 0.65 (not 0.5): a false-positive nudge (interrupting a good
        # response) is worse than a false-negative (alien call silently missed).
        # Callers can lower the threshold if they prefer higher recall.
        p = float(self.predict_proba([text])[0])
        return (p >= threshold), p

    def save(self, path=MODEL_PATH):
        # Ensure the class is pickled as train_alien_format_classifier.AlienFormatDetector
        # and NOT as __main__.AlienFormatDetector.
        #
        # Problem: when the script runs directly (__name__ == '__main__'), the class
        # lives in __main__ and pickle records it there.  When emperor_agent loads it,
        # __main__ is emperor_agent - no AlienFormatDetector → AttributeError.
        #
        # Python 3.14 fix: register the current __main__ module under the canonical
        # module name in sys.modules, AND patch __module__ on the class.  Pickle then
        # finds the class at 'train_alien_format_classifier.AlienFormatDetector' and
        # verifies it successfully.
        import sys as _sys
        main_mod = _sys.modules.get("__main__")
        already_registered = "train_alien_format_classifier" in _sys.modules
        if not already_registered and main_mod is not None:
            _sys.modules["train_alien_format_classifier"] = main_mod
        orig_module = self.__class__.__module__
        self.__class__.__module__ = "train_alien_format_classifier"
        try:
            joblib.dump(self, path)
        finally:
            self.__class__.__module__ = orig_module
            if not already_registered:
                _sys.modules.pop("train_alien_format_classifier", None)


    @staticmethod
    def load(path=MODEL_PATH) -> "AlienFormatDetector":
        return joblib.load(path)


# ════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ════════════════════════════════════════════════════════════════════════

def train(n_per_class: int, extra_path: str | None, mine: bool = False):
    texts, labels = build_synthetic_dataset(n_per_class)

    # ── Auto-mine real examples from chat_histories/ (opt-in, --mine) ──────────
    if mine:
        real_texts, real_labels = mine_from_chat_histories()
        if real_texts:
            texts  += real_texts  * REAL_EXAMPLE_OVERSAMPLE
            labels += real_labels * REAL_EXAMPLE_OVERSAMPLE

    # ── Optional extra JSONL supplied via --extra ──────────────────────────────
    if extra_path:
        ex_texts, ex_labels = load_real_examples(Path(extra_path))
        print(f"  Loaded {len(ex_texts)} example(s) from {extra_path}, "
              f"oversampling x{REAL_EXAMPLE_OVERSAMPLE}")
        texts  += ex_texts  * REAL_EXAMPLE_OVERSAMPLE
        labels += ex_labels * REAL_EXAMPLE_OVERSAMPLE

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    det = AlienFormatDetector()
    det.fit(X_train, y_train)

    probs = det.predict_proba(X_test)
    preds = (probs >= 0.5).astype(int)

    print("\n" + classification_report(y_test, preds, target_names=["clean", "alien_format"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, preds))

    det.save()
    print(f"\nSaved model to {MODEL_PATH.resolve()}")

    # Show a few misclassifications for a sanity check
    wrong = [(X_test[i], y_test[i], preds[i], probs[i]) for i in range(len(X_test)) if y_test[i] != preds[i]]
    if wrong:
        print(f"\n{len(wrong)} misclassified example(s) on the held-out set (showing up to 8):")
        for text, y, pred, p in wrong[:8]:
            snippet = text[:100].encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"  true={y} pred={pred} p={p:.2f} | {snippet!r}")

    return det


def eval_manual_examples(det: AlienFormatDetector):
    """A handful of hand-written sanity checks distinct from synthetic training data."""
    cases = [
        ("<tool_call>\n{\"name\": \"quick_search\", \"arguments\": {\"query\": \"weather\"}}\n</tool_call>", 1),
        ("<quick_search>weather in tokyo</quick_search>", 0),
        ("I did multiple quick_search calls, and I found this gold info: the sky is blue "
         "due to Rayleigh scattering.", 0),
        ("Let me use the tool to help you with that.", 0),
        ("For reference, MiniMax emits calls like:\n```\n<minimax:tool_call><invoke name=\"bash\">"
         "<parameter name=\"command\">ls</parameter></invoke></minimax:tool_call>\n```\n"
         "but you should use the XML powers format instead.", 0),
        # U+FF5C fullwidth vertical line - matches what DeepSeek actually emits.
        ("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" "
         "string=\"true\">pytest tests/</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>", 1),
        ("<minimax:tool_call>\n<invoke name=\"str_replace\">\n<parameter name=\"file\">config.py</parameter>"
         "\n</invoke>\n</minimax:tool_call>", 1),
        ("[SYSTEM — quick_search[weather] result:]\nHere are the top 3 results.", 0),
    ]
    print("\nManual sanity checks:")
    n_correct = 0
    for text, expected in cases:
        pred, p = det.predict(text)
        ok = "OK" if int(pred) == expected else "MISMATCH"
        n_correct += int(pred) == expected
        snippet = text[:70].encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  [{ok}] expected={expected} got={int(pred)} (p={p:.2f}) | {snippet!r}")
    print(f"{n_correct}/{len(cases)} correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",    action="store_true")
    ap.add_argument("--n-per-class", type=int, default=2500)
    ap.add_argument("--extra",    type=str, default=None, help="path to extra real_examples.jsonl")
    ap.add_argument("--mine",     action="store_true",
                    help="opt in to auto-mining chat_histories/ (verify CHAT_DIR path first — off by default)")
    ap.add_argument("--predict",  type=str, default=None, help="text to classify")
    ap.add_argument("--eval-examples", action="store_true", help="run hand-written sanity checks")
    args = ap.parse_args()

    if args.train:
        det = train(args.n_per_class, args.extra, mine=args.mine)
        eval_manual_examples(det)
        return

    if args.predict is not None:
        if not MODEL_PATH.exists():
            print("No trained model found — run with --train first.", file=sys.stderr)
            sys.exit(1)
        det = AlienFormatDetector.load()
        pred, p = det.predict(args.predict)
        print(json.dumps({"alien_format_detected": bool(pred), "probability": round(p, 4)}))
        return

    if args.eval_examples:
        if not MODEL_PATH.exists():
            print("No trained model found — run with --train first.", file=sys.stderr)
            sys.exit(1)
        eval_manual_examples(AlienFormatDetector.load())
        return

    ap.print_help()


if __name__ == "__main__":
    main()