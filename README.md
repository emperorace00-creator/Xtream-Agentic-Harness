# Xtream - Sandboxed AI Assistant

## TL;DR

- **XML tool calling** - tools are invoked via XML tags so its easy for any model to call tools
- **Academic &  web search** - Linkup for live web search; Semantic Scholar for academic papers with TL;DR summaries and Open Access PDF links
- **PDF & text RAG** - ingest PDFs or `.txt`/`.md` files, chunk + embed them, then search semantically via `<doc_search>`
- **Semantic code search** - `workspace_search` with `semantic=true` embeds functions/classes (tree-sitter chunking) so model can query by concept.
- **State rollback** - every turn is zip-archived; `/restore N` reverts the workspace and conversation to any of the last 20 turns
- **Sandboxed code execution** - `<bash>` commands run inside an isolated Docker container; the CLI stays on the host
- **Cross-session history** - every conversation is appended to a global JSONL archive, searchable via `<search_history>` using embeddings across all past sessions
- **Auto tool-format correction** - a local ML classifier detects when a model hallucinates the wrong tool format and nudges it to retry correctly
- **Image OCR & vision tiling** - dense images are tiled to capture fine detail

---

Xtream is an agentic assistant that talks to multiple LLM backends (NVIDIA, Gemini, or your own local llama.cpp server) through a simple XML tool-calling
protocol - so tool use works with any model. Code execution runs inside a Docker sandbox,
PDFs are ingested through a RAG pipeline (chunking → embeddings → cosine
similarity → reranking), workspace code search runs on Codestral Embed (with
BM25 as a silent fallback), and every turn is
archived so you can undo, rerun, or resume mid-task after an interrupt.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           start.py (Entry Point)                            │
│        Creates: ImageOCRAgent, EmperorAgent, TurnStateManager               │
│        Commands: /tool  /google /nim  /cf  /local  /reset  /restore         │
│                 /edit  /rerun  /delete  /history  /turns                    │
└────────┬──────────────────────────┬──────────────────────────┬──────────────┘
         │                          │                          │
┌────────▼────────┐       ┌─────────▼─────────┐      ┌─────────▼────────┐
│  ImageOCRAgent  │  ──>  │   EmperorAgent    │      │ TurnStateManager │
│ (used for PDFs) │       │                   │      │ (Zip Archives,   │
│                 │       └─────────┬─────────┘      │  Turn Ledger)    │
└─────────────────┘                 │                └──────────────────┘
                                    |
          ┌─────────────────────────┼─────────────────────────┬──────────────┐
          │                         │                         │              │
┌─────────▼─────────┐ ──> ┌─────────▼─────────┐     ┌─────────▼────────┐  ┌───▼──────────────┐
│ WorkspaceTracker  │ owns│   FileOpsAgent    │     │  DocSearchAgent  │  │  Docker Sandbox  │
│ (BM25 fallback &  │     │  (str_replace,    │     │  (chunk+embed,   │  │  (docker exec,   │
│  Reconcile)       │     │  view_lines,      │     │   cosine sim,    │  │  via bash tool)  │
│                   │     │  search_in_file,  │     │   doc_search)    │  │                  │
│                   │     │  workspace_search)│     │                  │  │                  │
└───────────────────┘     └─────────┬─────────┘     └──────────────────┘  └──────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  CodeSearchAgent  │
                          │  (tree-sitter     │
                          │   chunk + Codestral│
                          │   Embed semantic) │
                          └───────────────────┘

Extra features: ToolCallSummarizer and
GlobalHistoryWriter - see Operations & UI and State Management below.
```

## Features

### Core
- **Multi-Backend Support** - Supports Cloudflare Workers AI (default), NVIDIA NIM,
  Google Gemini, and a local llama.cpp server. Switch backends during a session with
  `/cf`, `/nim`, `/google`, or `/local`.
- **XML Tool System** - Uses XML tag format (`<bash>`, `<str_replace>`, etc.) for tool
  execution, making it compatible with any text-generating model.
- **NIM Send-Time XML-to-JSON Tool Mapping** - Translates inline XML tags into standard OpenAI/NIM `tool_calls` JSON schemas at send-time. This aligns with the native tool-calling format the models were trained on (preserving a coherent reasoning trajectory)
- **Gemini Thought Signature Serialization** - Captures the encrypted `thought_signature` from Google Gemini’s streaming thought chunks. When the subsequent tool result is sent back, the agent injects this signature as a native thought part, allowing the Gemini API to restore its internal reasoning state
- **Tool Calling Format Auto-Correction** - Detects when models hallucinate absurd tool-calling structure instead of the required XML tags, using a local logistic regression classifier. Automatically intercepts and nudges the model to retry in the correct format without breaking the turn.
- **Academic Literature Search** - Queries the Semantic Scholar database via the `research` tool group. Returns title, year, AI-generated TL;DR, and Open Access PDF link per paper. Results are reranked by relevance and support year filtering (`2022-`, `2020-2024`). Enabled via `/tool` → research.
- **Sandboxed Execution** - Code execution runs inside an isolated Docker container.
  The CLI runs natively on the host. `/tool` opens a tool group selector (`web`, `files`, `pdf`, `bash`).
  Docker is required only if the `bash` group is enabled.
- **File Storage** - Files are organized into three directories: `uploads/` for user-provided files, `scratch/` for the persistent working directory, and `outputs/` for generated shared artifacts.

### Intelligence & Search
- **Web Search** - Uses Linkup for web search. Truncates long pages by extracting main
  content and stripping navigation elements.
- **Search Reranking** - Uses an cross-encoder model to rerank both web search results and global history search candidates before returning
  them to the model.
- **Image OCR & Vision Tiling** - Processes images and PDFs with parallel OCR(gemini+NIM). Dense documents are automatically sliced into tiles and evaluated to capture fine micro-details.
- **PDF Ingestion** - Processes PDFs through the OCR pipeline, chunks the text, generates embeddings, and performs cosine similarity search via
  `doc_search`. Plaintext files (`.txt`, `.md`) skip the OCR step entirely -
  `ingest_text` chunks and embeds them directly so they're searchable the
  same way.
- **Workspace Search** - Two distinct modes, not one feature with a toggle:
  by default (`semantic=false`, the default) `workspace_search` is a plain
  substring grep across **both** `scratch/` and `uploads/`. Setting
  `semantic=true` switches to Codestral Embed concept search: files are
  chunked by function/class (tree-sitter, regex fallback), embedded, and
  ranked by cosine similarity — so "where do we handle rate limiting?"
  finds the right function even when those words never appear. BM25 is
  kept as a silent fallback if the Mistral key is missing or the embed
  call fails. Note this is true embedding-based semantic search, sharing
  the same idea as `doc_search` / `search_history` but using a code-native
  embedder (Codestral Embed via `api.mistral.ai`) rather than Nemotron.

### State Management
- **State Rollback** - Archives the scratch directory and workspace registry into a zip
  file at every successful turn (rolling window of the last 20 turns). `/restore N`
  reverts the environment to any archived state.
- **History Editing** - `/edit N` modifies any previous message. Editing a **user** or
  **agent** message replaces the stored text. Run `/rerun N` afterward if you want to regenrate response.
- **Turn Deletion** - `/delete N` removes turn N (user+assistant pair) from conversation
  memory and renumbers subsequent turns without changing the live workspace.
- **Execution Interruption** - Pressing `Ctrl+C` mid-turn pauses tool execution, rolls
  back any workspace changes made so far, and prompts first for an optional pasted
  thinking trace, then for guidance. Typing guidance resumes the turn from exactly
  where it was interrupted, with any pasted thinking trace injected alongside it so
  the model sees its own reasoning as well as your correction; pressing Enter on
  guidance cancels and saves a partial summary.
- **Global History Search** - Every turn is appended to a per-project JSONL file in a
  configurable global directory. The `<search_history>` tool indexes those turns with embeddings (incremental, sidecar-cached) and reranks the top candidates before
  returning them. This archive is permanent and cross-session - **`/reset` does not
  clear it.** `/reset` only wipes the current chat session, scratch files, and
  per-turn backups; the long-term history that `search_history` searches survives
  on purpose. `ingest_chat` lets you fold in transcripts pasted from other chat
  UIs (Gemini, ChatGPT web) - drop the raw text in `uploads/` and it's imported
  into this same archive, searchable via `search_history` from then on.
- **Workspace Synchronisation** - After every bash command, the agent calls
  `reconcile_workspace()` to detect filesystem changes and keep the BM25 index in sync
  with what is actually on disk.

### Operations & UI
- **String Replacement** - Exact-match, uniqueness-validated text replacement. Supports
  `count=N` to replace the first N occurrences, or `count=all` to rename a symbol
  everywhere in a file.
- **Tool Summarisation** - Compresses every tool call's outcome deterministically
  (no LLM needed) into a compact audit log, which is injected into the next turn's
  context so the model always knows what it did last.
- **Math Rendering** - Converts LaTeX math expressions to Unicode for clean terminal output.
- **Terminal UI** - Syntax-highlighted code blocks and Rich Markdown rendering via the
  Rich library.
- **Image Preview & `<show_image>`** - Renders images inline in the terminal via `chafa`. The `<show_image>` tool allows the model to display generated plots, diagrams, or images directly in the console.
- **Auto Image Enhancement & Preprocessing** - Automatically enhances image quality (contrast, orientation, brightness) on a copy before sending to OCR or vision models, without modifying the original file.

## Setup

### Prerequisites
- Python 3.11+
- Container runtime (e.g., Docker) - required only for tool mode (`/tool`)

### 1. Clone & Install

```bash
git clone https://github.com/emperorace00-creator/Xtream-Agentic-Harness.git
cd Xtream-Agentic-Harness
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the environment template:

```bash
cp .env.example .env
```

Edit `.env` - each variable points to a text file containing the raw key string:

```bash
GOOGLE_API_KEY_FILE=/path/to/google_api_key.txt
NVIDIA_API_KEY_FILE=/path/to/nvidia_api_key.txt
LINKUP_API_KEY_FILE=/path/to/linkup_api_key.txt
CLOUDFLARE_API_KEY_FILE=/path/to/cloudflare_api_key.txt
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
SEMANTIC_SCHOLAR_API_KEY_FILE=/path/to/semantic_scholar_api_key.txt
MISTRAL_API_KEY_FILE=/path/to/mistral_api_key.txt  # Codestral Embed, semantic code search only

# Optional: shared history directory across projects.
# If unset, falls back to database/chat_histories/ inside this project folder.
GLOBAL_HISTORIES_DIR=/path/to/global/histories
```

**Required Services:**
 ________________________________________________________________________________________________
| Service               | Function                                | URL                          |
| :-------------------- | :---------------------------------------| :----------------------------|
| NVIDIA NIM            | Default backend (Embeddings,OCR)        | build.nvidia.com             |
| Workers AI            | Cloudflare backend                      | dash.cloudflare.com          |
| Google AI             | Gemini backend                          | aistudio.google.com          |
| Linkup                | Web search                              | linkup.so                    |
| Mistral               | Codestral Embed (semantic code search)  | console.mistral.ai           |
| Local (llama.cpp)     | Self-hosted backend                     | github.com/ggml-org/llama.cpp|

Only NVIDIA is required for the full feature set (embeddings, reranking, OCR). Linkup is
needed for web search. Mistral is needed only for `workspace_search semantic=true`
(Codestral Embed); without it, that mode silently falls back to BM25. Each cloud LLM
backend is optional - the agent starts fine without any of them configured. Note that a
missing key isn't always surfaced up front: `WebAgent` and `ImageOCRAgent` print a
warning at startup if their key is missing, but a missing Google/NIM/Cloudflare key
currently fails silently until you actually try to use that backend, at which point
the API call errors out. Configure the key for whichever backend you plan to use as
your default before you start chatting with it.

### 3. Build the Sandbox (tool mode only)

```bash
docker build -t emperor-base:latest .
```

### 4. Run

```bash
python start.py
```

The sandbox container starts automatically when the `bash` group is enabled in `/tool`.

## Usage

### Chat
Enter messages directly into the terminal prompt.

### Commands
 _____________________________________________________________________________________
| Command               | Action                                                      |
| :-------------------- | :-----------------------------------------------------------|
| `/tool`               | Select tool groups (`web`, `files`, `pdf`, `bash`).         |
| `/google`             | Switch to Google Gemini backend                             |
| `/nim`                | Switch to NVIDIA NIM backend                                |
| `/cf`                 | Switch to Cloudflare backend (default)                      |
| `/local`              | Switch to a local llama.cpp server (see Local Backend below)|
| `/tools`              | Display current tool mode and active backend                |
| `/history`            | Print the current conversation history                      |
| `/turns`              | Display the turn timeline and revision markers              |
| `/review [focus]`     | Self-review the last response (optional focus text)         |
| `/rerun N` / `last`   | Regenerate the AI response for a specific turn              |
| `/edit N` / `last`    | Edit a past message                                         |
|                       | (see History Editing below for user vs. agent behavior)     |    
| `/restore N` / `last` | Revert workspace and memory to end of turn N                |
| `/delete N`           | Delete turn N (user+assistant pair) from conversation memory|
| `/reset`              | Clear the current session, scratch files, and backups       |
|                       | (Global History permanent chat storage still survives)      |

> **Tool groups**: `web` `files` `pdf` `bash` `research` - toggle any combination via `/tool`.

### Tool Mode

When `/tool` is active, the model can invoke tools using XML tags:

```xml
<!-- Run shell commands in the sandbox -->
<bash>python solution.py</bash>

<!-- Search the web (optional from_date="YYYY-MM-DD" attribute to restrict results) -->
<quick_search from_date="2024-01-01">Python asyncio gather vs wait</quick_search>

<!-- Read a URL -->
<url_search url="https://docs.python.org/3/" context="asyncio"/>

<!-- Replace string in a file -->
<str_replace>
<file>main.py</file>
<old_str>return None</old_str>
<new_str>return result</new_str>
</str_replace>

<!-- Rename a variable everywhere in a file -->
<str_replace>
<file>api/server.py</file>
<old_str>self._db_conn</old_str>
<new_str>self._connection</new_str>
<count>all</count>
</str_replace>

<!-- Read a section of a file -->
<view_lines>
<file>solution.py</file>
<start>1</start>
<end>50</end>
</view_lines>

<!-- Ingest a PDF for semantic search -->
<ingest_pdf>textbook.pdf</ingest_pdf>

<!-- Ingest a plaintext file for semantic search -->
<ingest_text>notes.txt</ingest_text>

<!-- Search ingested documents (PDFs and text files) using embeddings -->
<doc_search>
<query>Newton's second law derivation</query>
<top_k>5</top_k>
</doc_search>

<!-- Keyword search across scratch and uploads (exact grep, the default) -->
<workspace_search>
<query>API_KEY</query>
</workspace_search>

<!-- Semantic code search: concept-level, tree-sitter chunks + Codestral Embed -->
<workspace_search>
<query>where do we handle rate limit errors</query>
<semantic>true</semantic>
</workspace_search>

<!-- Search within a specific file -->
<search_in_file>
<file>utils.py</file>
<pattern>def parse_</pattern>
<context_lines>3</context_lines>
<regex>true</regex>
<max_results>10</max_results>
</search_in_file>

<!-- Search global conversation history -->
<search_history>
<query>kinetic theory gas pressure derivation</query>
<top_k>5</top_k>
</search_history>

<!-- Import a raw chat transcript pasted from another AI's web UI -->
<ingest_chat>gemini_chat.txt</ingest_chat>

<!-- Search the Semantic Scholar academic database -->
<search_semantic_scholar>
<query>self-supervised learning vision transformers</query>
<limit>5</limit>
<year>2021-</year>
</search_semantic_scholar>
```

### Local Backend

Run any model on your own machine via [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server` and connect to it with `/local`. Emperor is purely a client here - it
never launches or manages the server process. You start `llama-server.exe` yourself in
its own terminal, with whatever model, quant, and flags you want, and `/local` just
points at whatever's listening on the configured port. This keeps the workflow flexible
if you swap models/quants often - no code or config changes needed to try a different one.

**1. Start your local server** (example, adjust model/flags to your hardware):
```bash
llama-server.exe -m your-model.gguf --mmproj mmproj.gguf -c 6000 -fa on --port 1234 -np 1
```

**2. Set the endpoint** in `.env` if it's not the default:
```bash
LOCAL_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_MODEL=your-model-name
LOCAL_API_KEY_FILE=   # leave blank unless you set --api-key on llama-server
```

**3. Switch to it** in-session:
```
/local
```

**Cache warm-up:** switching to `/local` (and toggling `/tool` while already on `/local`)
fires a background request that pushes the current system prompt into llama-server's KV
cache ahead of time, so your first real message only has to prefill the new content, not
the whole system prompt from scratch. This is a one-time-per-shape optimization - llama-server
keeps the growing conversation cached between turns on its own afterward (visible as
`graphs reused` in the server log), nothing further needed from Emperor's side.

**Reasoning / thinking control:** on models with a Gemma-4-style thinking mode, this is
controlled entirely by the flag you launch `llama-server` with, not from inside Emperor:
- Thinking **on** (default for most reasoning-capable models) - no flag needed.
- Thinking **off** - add `--reasoning off` to your launch command.

To switch modes, close the running `llama-server.exe` and relaunch with the flag added or
removed, then run `/local` again in Emperor to reconnect and re-warm the cache.

## Project Structure

```text
emperor/
├── start.py                  # Entry point: Docker setup, CLI loop, command dispatch
├── config.py                 # Configuration, paths, model names, env vars
├── emperor_agent.py          # Core agent: generation loop, XML tool parser, history
├── llm_backends.py           # API integrations for NIM, Gemini, Cloudflare (mixin)
├── tool_handlers.py          # Tool dispatch and handler methods (mixin)
├── core_tool_definitions.py  # System prompt blocks for all XML tools
├── renderer.py               # Terminal rendering: LaTeX → Unicode, syntax highlight
├── web_agent.py              # Linkup search, URL fetch, content extraction, reranker
├── file_ops_agent.py         # str_replace, view_lines, search_in_file, workspace search
├── workspace_tracker.py      # File registry, BM25 index, reconcile
├── code_search_agent.py      # Tree-sitter chunking, Codestral Embed, semantic code search
├── doc_search_agent.py       # PDF chunking, NVIDIA embeddings, cosine similarity
├── image_ocr_agent.py        # 3-tier NIM OCR fallback chain (parallel threads)
├── turn_state_manager.py     # Per-turn zip archives, turn ledger, /restore pipeline
├── tool_summarizer.py        # Deterministic tool-call log compressor
├── utils.py                  # Token counter, GlobalHistoryWriter, image preprocessor
├── Dockerfile                # Sandbox container image
├── entrypoint.sh             # Container initialisation script
├── requirements.txt          # Python package dependencies
├── .env.example              # Environment variable template
└── .gitignore                # Excludes secrets and runtime data
```

## Design Decisions

- **XML Tool Calling as Primary Format**: Tool calls are emitted as XML tags inside
  normal text generation, so any model that can produce text can drive the tool loop.

- **Codestral Embed for Semantic Code Search**: `workspace_search semantic=true`
  chunks files at function/class boundaries (tree-sitter, regex fallback) and
  embeds them with Mistral Codestral Embed. Index updates are incremental and
  lazy — `bash`/`str_replace` only flip a stale flag; re-embed happens on the
  next semantic query, and only for files whose MD5 changed. BM25 remains a
  silent fallback if the Mistral key is missing or the API is unreachable.
  Grep (`semantic=false`) is unchanged. Embeddings for PDFs and global history
  still use NVIDIA Nemotron via the existing `_embed()` path.

- **Zip Archives for State Management**: Per-turn state is stored as a zip of the
  scratch directory plus a registry snapshot, rather than a git history. This avoids
  merge conflicts and keeps the rollback path simple: unzip, reload, truncate history.
  Only the last 20 archives are kept; ledger entries (turn metadata) are kept forever.

- **Host CLI, Containerised Bash**: `start.py` runs natively on the host so readline,
  arrow keys, and Rich formatting work without TTY emulation. Only `<bash>` tool
  commands are proxied through `docker exec`, giving full shell access inside an
  isolated container without affecting the terminal experience.

- **Deterministic Tool Summarisation**: Summarising with another LLM call would add
  latency and cost to every single turn. Since tool calls are structured data, a plain string-inspection pass (see Tool Summarisation above) is both
  cheaper and more accurate to the ground truth.

## License

MIT - see [LICENSE](LICENSE) for details.
