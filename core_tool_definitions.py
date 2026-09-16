# core_tool_definitions.py
#
# Single source of truth for ALL pseudo-tool prompt blocks.
# No JSON schemas. No tools= API param. Ever.
#
# Adding a new tool:
#   1. Add its tag description to the relevant *_PROMPT constant below.
#   2. Add one entry to _build_dispatch() in tool_handlers.py.
#   3. Write one _handle_* method in tool_handlers.py.
#
# Tool mode is now per-group: 'web', 'files', 'pdf', 'bash'.
# Toggle groups interactively via /tool (checkboxlist dialog).
# When a group is active, its corresponding *_PROMPT constant is
# injected into the system prompt automatically.

from datetime import date


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT SPEC  — always injected, teaches the model the tag system
# ══════════════════════════════════════════════════════════════════════════════

PSEUDO_TOOL_FORMAT = """
### POWERS FORMAT

Use these powers by writing XML tags in your response, matching the exact
shape shown in each power's own example below. Results come back, then you
continue. Emit as many power tags as you need in one response.

RULES:
- Each tag on its own line. Never inline mid-sentence.
- Write file content and code raw inside tags — no escaping needed.
- Only use tag names defined below, checked fresh each turn — not from
  memory. Your own earlier power calls stay visible in this conversation's
  history even after a power is turned off, so seeing one you used before
  isn't proof it's still available now. (The user can turn availability of specific powers on or off between turns)
- Don't mess up the syntax of powers, pay attention to the syntax given below.
- If you want to use a power, the XML tag must appear in your actual response — not just in your thinking/reasoning trace. The runtime reads only your final response, not your thoughts.
"""


# ══════════════════════════════════════════════════════════════════════════════
# WEB  — quick_search + url_search
# ══════════════════════════════════════════════════════════════════════════════

WEB_TOOLS_PROMPT = """
### WEB POWERS

**quick_search** — search the web for facts, recent events, or to verify or refine your answer.

1. First synthesize the answer in your internal reasoning using your internal knowledge.
2. Identify the gaps, uncertainties, or potential out-of-date points in your draft.
3. Use those gaps to construct your `quick_search` queries.
4. After the search power returns results, you synthesize and write your final, verified response to the user.

Write multiple queries to get comprehensive results.

<quick_search>asyncio gather vs asyncio wait difference</quick_search>

Optional `from_date` attribute (YYYY-MM-DD) restricts results to content
published on or after that date. Use it when recency matters and
older sources would actively mislead — e.g. "current" notifications,
policies, specs, or version numbers — since low-authority sites (like SEO aggregators) often republish stale info without updating it, and
that stale content can directly contradict a fresher source in the same
result set with no way to tell them apart otherwise. Don't set it for
evergreen/historical topics where older sources are still valid.

STRATEGY for time-sensitive queries: start narrow (a recent from_date) and
widen only if needed.
  1. First call: set from_date to a recent cutoff appropriate to the topic
     (e.g. ~6-12 months back for "current" policies/specs; a few months for
     fast-moving topics).
  2. Judge the actual results, not the result count — Read the content: do the dates, version numbers,
     or specifics mentioned actually look current, or do they look like
     older material that happens to still exist online? If what came back
     looks stale, thin on substance, or off-topic, that's your signal to
     widen — issue a follow-up call with an earlier from_date (or omit it).
  3. If a later, more permissive search directly contradicts an earlier,
     more recent-dated one, trust the more recent-dated result.

<quick_search from_date="2026-01-01">latest Indian Army TES notification training duration</quick_search>

Don't get overwhelmed by web search results — they might be misleading, sometimes conflicting.
---

**url_search** — read content from a specific URL. Use when the user pastes a link,
or when you have a direct URL from a search result. context is optional.

<url_search url="https://docs.python.org/3/library/asyncio-task.html" context="gather vs wait"/>

---

"""


def web_date_context() -> str:
    """
    Returns a REFERENCE DATE block for the system prompt, injected only when
    the 'web' group is active (see emperor_agent._base_system). Computed
    fresh on every call — NOT a module-level constant — so a long-running
    session still has the correct date even if it spans midnight or runs for
    days. Without this, the model has no reliable anchor for "recent,"
    "current," or "last N months," and can't correctly compute a from_date
    value for quick_search — it would be guessing at today's date from
    training data, which is exactly the kind of staleness this whole
    from_date feature exists to avoid.
    """
    today = date.today()
    return (
        "### REFERENCE DATE\n"
        f"Today's date is {today.isoformat()} ({today.strftime('%B %d, %Y')}). "
        "Use this as your anchor when reasoning about what's recent/current, "
        "and when computing a from_date value for quick_search — not a "
        "remembered or assumed date.\n\n---"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY  — search_history (always bundled with files group)
# ══════════════════════════════════════════════════════════════════════════════

HISTORY_TOOLS_PROMPT = """
**search_history** — search everything you and the user have ever discussed.
ONLY use this power when the user explicitly asks you to look something up from
a past conversation (e.g. "do you remember when we discussed X?", "find that
derivation from last time", "what did we say about Y before?").
Do NOT use it proactively or on every turn.

Returns the exact matched turns (your question + the agent's answer) so you
can answer immediately from the result. If a turn is long, a preview is shown
with a view_lines hint — use view_lines with the absolute path from the result
to read the full text, or search_in_file to search within it.

<search_history>
<query>kinetic theory gas pressure derivation</query>
<top_k>5</top_k>
</search_history>

---

**ingest_chat** — import a raw chat transcript (pasted from Gemini/Claude web) into your permanent history
so search_history can find it later. The user places the .txt (or .md) in /uploads;
you call this power on the filename. 
And you call this tool only if user asks explicitly to ingest external chats.

<ingest_chat>gemini_chat.txt</ingest_chat>

---

"""

# Backward-compat alias (referenced by old code expecting CORE_PROMPT)
CORE_PROMPT = WEB_TOOLS_PROMPT + HISTORY_TOOLS_PROMPT


# ══════════════════════════════════════════════════════════════════════════════
# FILE OPS  — view_lines, search_in_file, workspace_search, str_replace
# ══════════════════════════════════════════════════════════════════════════════

FILE_TOOLS_PROMPT = """
### YOUR ENVIRONMENT

**Read-only** — /uploads
  All files the user drops in the project's uploads/ folder appear here automatically.
  You can read any of them directly with view_lines, bash, or any file power.
  No ingestion step needed for .py, .txt, .md, .csv, .json, etc.
  PDFs are the only exception — they need ingest_pdf first.
 ⚠️ PATH NOTE: Use the same absolute paths in every power — `/uploads/filename`, `/workspace/scratch/filename`, `/outputs/filename`. For scratch files you can also use bare relative names like `start.py`.

**Your sandbox** — /workspace/scratch
  Create files, run code, save your work here. This volume is PERSISTENT across sessions.

**Share files with user** — /outputs
  The user CANNOT access /workspace/scratch directly.
  To share a finished file, copy it to /outputs:
    <bash>cp /workspace/scratch/result.py /outputs/result.py</bash>
  Files in /outputs appear in the user's project folder immediately.

---

**DOCUMENT READING RULES**
- "first N pages", "pages X to Y", "beginning of the doc" → use view_lines DIRECTLY
- Only use doc_search when the user names a specific concept/topic to find

---

**str_replace** — surgical edit for a small, exact snippet. old_str must match
exactly including indentation. Use view_lines first if unsure of the exact text.

Default (count omitted or count=1): old_str must appear exactly once.
Use `<count>all</count>` to replace every occurrence — ideal for renaming a variable.

<str_replace>
<file>notes/physics.md</file>
<old_str>    return {"status": "ok"}</old_str>
<new_str>    return {"status": "ok", "version": "1.0"}</new_str>
</str_replace>

Rename a variable everywhere in the file:

<str_replace>
<file>api/server.py</file>
<old_str>self._db_conn</old_str>
<new_str>self._connection</new_str>
<count>all</count>
</str_replace>

To delete a block, leave <new_str> empty:

<str_replace>
<file>notes/physics.md</file>
<old_str>this whole wrong paragraph goes here</old_str>
<new_str></new_str>
</str_replace>

---

**view_lines** — read a segment of a file. `start` defaults to 1; omit `end` to read to the end of the file.
Works on any path: /workspace, /uploads, /outputs, and absolute host paths (e.g. from search_history results).

<view_lines>
<file>/uploads/reference.py</file>
<start>10</start>
<end>30</end>
</view_lines>

---

**search_in_file** — search within a single known file.
`context_lines` = lines before/after each match (default 3). `max_results` caps total matches returned (default 10).
Works on any path: /workspace, /uploads, /outputs, and absolute host paths (e.g. from search_history results).

<search_in_file>
<file>/uploads/notes.md</file>
<pattern>osmosis</pattern>
<context_lines>3</context_lines>
</search_in_file>

For regex patterns, set `regex=true`:

<search_in_file>
<file>/uploads/server.py</file>
<pattern>def \w+_handler</pattern>
<regex>true</regex>
<max_results>5</max_results>
</search_in_file>

---

**SEARCH QUICK REFERENCE**
- Topic in an ingested PDF or text file → `doc_search`
- Something in your saved notes/summaries → `workspace_search`
- Positional ("page 3", "first section") → `view_lines` directly
- Past conversation → `search_history`
- Content in any known file → `search_in_file`

---

**workspace_search** — search across your codebase and scratch files.
Default is exact match (grep). Set semantic=true for concept-level code search — finds
relevant functions and classes even when your exact words don't appear in the code.

<workspace_search>
<query>_parse_pseudo_tools</query>
</workspace_search>

<workspace_search>
<query>where do we handle rate limit errors</query>
<semantic>true</semantic>
</workspace_search>

---

"""


# ══════════════════════════════════════════════════════════════════════════════
# PDF  — ingest_pdf + doc_search + view_lines
# ══════════════════════════════════════════════════════════════════════════════

PDF_TOOLS_PROMPT = """
**ingest_pdf** — convert a PDF from /uploads into searchable text.
Runs OCR on every page, saves a .txt to /workspace/scratch, and builds a
semantic index so doc_search can find passages by concept.

Only PDFs need this step. Every other file type (.py, .txt, .md, .csv, images)
is directly readable from /uploads — no ingestion needed.

<ingest_pdf>research_paper.pdf</ingest_pdf>

After ingestion, use doc_search to query the content:

<doc_search>
<query>types of assessment tools in physical education</query>
<top_k>5</top_k>
</doc_search>

---

**ingest_text** — chunk + embed any .txt or .md file from /uploads so doc_search
can query it semantically, without the OCR step PDFs need.

<ingest_text>notes.txt</ingest_text>

After ingestion, query it the same way as an ingested PDF:

<doc_search>
<query>key takeaways from the meeting notes</query>
<top_k>5</top_k>
</doc_search>

---

**doc_search** — search ingested documents (PDFs and text files) by concept or topic.
Write queries as descriptive phrases, not single keywords.
For best results emit 2-3 calls with varied phrasings in one response.

<doc_search>
<query>types of assessment tools in physical education</query>
<top_k>5</top_k>
</doc_search>

ocr'ed pdf contents may have some inconsistency/typos introduced during the text extraction, so try your best to understand the text.
---

**view_lines** — read a line range directly from the OCR'd .txt in scratch. `start` defaults to 1; omit `end` to read to the end of the file.
Use this to read surrounding context after doc_search returns line numbers,
or to read the document sequentially (e.g. "first 5 pages").

<view_lines>
<file>/workspace/scratch/research_paper.txt</file>
<start>142</start>
<end>175</end>
</view_lines>

---
"""


# ══════════════════════════════════════════════════════════════════════════════
# BASH  — bash sandbox tool
# ══════════════════════════════════════════════════════════════════════════════

BASH_TOOLS_PROMPT = """
**bash** — run any shell command inside the container. cwd=/workspace/scratch.
Full bash is available: python, pip, mkdir, cp, cat, grep, find, curl, git, etc.
Paths: `/uploads` (read-only), `/workspace/scratch` (your sandbox), `/outputs` (share with user).

<bash>python solution.py</bash>

<bash>cat /uploads/data.csv | head -20</bash>

<bash>cp /workspace/scratch/report.pdf /outputs/report.pdf</bash>

---

**show_image** — preview an image file you already created, inline in the
terminal, and copy it to /outputs so the user can open it. Give the filename
only (relative to your sandbox) or an absolute /workspace/scratch, /uploads,
or /outputs path — not code, not a description. bash creates the file first;
show_image just displays a file that already exists.

<show_image>chart.png</show_image>

Don't hand-draw ASCII art for diagrams or charts — generate a real image
with bash, then show_image it. example: 

  <bash>
  cat > plot.py << 'EOF'
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme()
  fig, ax = plt.subplots()
  ax.quiver(0, 0, 3, 4, angles='xy', scale_units='xy', scale=1, color='crimson', label='F1')
  ax.set_xlim(-1, 8); ax.set_ylim(-1, 6)
  ax.legend(); ax.grid(True)
  plt.savefig('plot.png')
  EOF
  python plot.py
  </bash>
  <show_image>plot.png</show_image>

If show_image returns an error (bad path, not an image, file doesn't exist),
fix the mistake and retry.

---
"""


# Backward-compat alias (old imports expect FILE_OPS_PROMPT)
# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH  — search_semantic_scholar
# ══════════════════════════════════════════════════════════════════════════════

RESEARCH_TOOLS_PROMPT = """
**search_semantic_scholar** — search the Semantic Scholar academic database across all fields
(Physics, Chemistry, Math, CS, Biology, Medicine, etc.).
Returns titles, publication years, AI-generated TL;DR summaries (or abstract snippets), and
Open Access PDF links (including arXiv) where available.
Use this when quick_search keeps returning shallow, exam-prep, or non-peer-reviewed content on a
scientific claim, and you need actual literature with citable sources.

Do NOT guess specific molecules, compounds, or author names if unknown — search for the
underlying physical concept or phenomenon instead.

The <year> field is optional and supports the following formats:
  2024         — exact year
  2022-        — from 2022 onward (most common: "recent papers on X")
  2020-2024    — explicit range

Today's date is provided in the system context so you can construct accurate year filters.

<search_semantic_scholar>
<query>triplet ground state diatomic molecules nitrogen oxygen</query>
<limit>5</limit>
<year>2015-</year>
</search_semantic_scholar>
"""

FILE_OPS_PROMPT = FILE_TOOLS_PROMPT + PDF_TOOLS_PROMPT + BASH_TOOLS_PROMPT


# ── Group → prompt block mapping (used by emperor_agent._base_system) ─────────
# Order matters: it determines the order sections appear in the system prompt.
GROUP_PROMPTS: dict = {
    "web":      WEB_TOOLS_PROMPT,
    "files":    FILE_TOOLS_PROMPT + HISTORY_TOOLS_PROMPT,  # history is included with files only
    "pdf":      PDF_TOOLS_PROMPT,
    "bash":     BASH_TOOLS_PROMPT,
    "research": RESEARCH_TOOLS_PROMPT,
}

# Tools belonging to each group — drives _get_active_known_tools()
GROUP_TOOLS: dict = {
    "web":      {"quick_search", "url_search"},
    "files":    {"view_lines", "search_in_file", "workspace_search", "search_history", "str_replace", "ingest_chat"},
    "pdf":      {"ingest_pdf", "ingest_text", "doc_search", "view_lines"},
    "bash":     {"bash", "show_image"},
    "research": {"search_semantic_scholar"},
}

# item 22 / SP-20: union of every tool declared across GROUP_TOOLS. This must
# always equal _build_dispatch().keys() in tool_handlers.py — if a tool is
# added to one but not the other, it either can't be dispatched (parser has
# no handler) or can't ever be invoked (no XML tag advertised for it). There
# was previously no automated check for this drift; EmperorAgent.__init__()
# now asserts the two sets match at startup — see emperor_agent.py.
GROUP_TOOLS_UNION: frozenset = frozenset().union(*GROUP_TOOLS.values())