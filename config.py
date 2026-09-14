"""Project-wide configuration — paths, model settings, API endpoints."""
# config.py
#
# All paths are host-local. Only the bash tool proxies through Docker.

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (next to this file) so contributors only
# need to copy .env.example → .env and fill in their paths. System env vars
# take precedence over .env values — standard python-dotenv behaviour.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# -- WORKSPACE ROOT (auto-detected from this file's location) -----------------
_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = _THIS_DIR

# -- LOCAL WORKSPACE FOLDERS --------------------------------------------------
SCRATCH_DIR    = os.path.join(WORKSPACE_ROOT, "scratch")
UPLOADS_FOLDER = os.path.join(WORKSPACE_ROOT, "uploads")
OUTPUTS_DIR    = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(SCRATCH_DIR,    exist_ok=True)
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_DIR,    exist_ok=True)

# -- DATABASE -----------------------------------------------------------------
DATABASE_DIR            = os.path.join(WORKSPACE_ROOT, "database")
os.makedirs(DATABASE_DIR, exist_ok=True)

CHAT_HISTORY_FILE       = os.path.join(DATABASE_DIR, "chat_history.json")
WORKSPACE_REGISTRY_FILE = os.path.join(DATABASE_DIR, "workspace_registry.json")
WEB_CACHE_FILE          = os.path.join(DATABASE_DIR, "web_cache.json")

# -- PER-TURN BACKUPS ---------------------------------------------------------
# Per-turn scratch/ zips and workspace registry snapshots live here.
# Only the last MAX_BACKUP_TURNS archives are kept on disk (rolling window).
# Ledger entries (turn_ledger.json) are kept forever — they are tiny.
BACKUPS_DIR       = os.path.join(DATABASE_DIR, "backups")
MAX_BACKUP_TURNS  = 20
os.makedirs(BACKUPS_DIR, exist_ok=True)

# -- GLOBAL CHAT HISTORIES ----------------------------------------------------
# Cross-project shared history folder. Set GLOBAL_HISTORIES_DIR in .env.
# Falls back to a local database/chat_histories/ folder if not configured.
GLOBAL_HISTORIES_DIR = os.environ.get(
    "GLOBAL_HISTORIES_DIR",
    os.path.join(DATABASE_DIR, "chat_histories"),
)
os.makedirs(GLOBAL_HISTORIES_DIR, exist_ok=True)
CHAT_HISTORIES_DIR = GLOBAL_HISTORIES_DIR  # alias used by tool handlers

# -- DOCKER SANDBOX -----------------------------------------------------------
# Container is only used by the bash tool to run arbitrary commands safely.
# start.py runs on the host so arrow keys / readline work natively.
# Container name matches start.py naming convention.
_project_name  = Path(WORKSPACE_ROOT).name
import re as _re
_safe_name = _re.sub(r"[^a-z0-9\-]", "", _project_name.lower().replace(" ", "-").replace("_", "-"))
_safe_name = _re.sub(r"-{2,}", "-", _safe_name).strip("-")  # collapse runs of dashes
CONTAINER_NAME = "emperor-" + (_safe_name or "project")

# Working directory INSIDE the container for bash commands.
# The local SCRATCH_DIR is bind-mounted here by start.py (_setup_sandbox).
CONTAINER_SCRATCH = "/workspace/scratch"

# -- MODELS -------------------------------------------------------------------
CLOUDFLARE_ACCOUNT_ID    = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_MODEL         = "@cf/moonshotai/kimi-k2.7-code"
CLOUDFLARE_THINKING_TYPE = "native"
CLOUDFLARE_VISION        = True

# When True, each image is enhanced as a master then sliced into an overview
# + 4 named tiles (top_left/top_right/bottom_left/bottom_right). Requires
# opencv-python. Dramatically helps models read fine detail in high-res docs,
# physics notes, dense screenshots, etc.
# Set True per-session when you need it; False for casual/photo use.
IMAGE_TILING = True

# Grid dimensions (rows, cols) for the tiling pipeline.
# 3×2 is ideal for portrait pages (physics notes, A4 scans): tiles are
# 55 % wide × 40 % tall with 10 % overlap → 6 tiles + 1 overview = 7 images.
# Try (3, 3) for very dense pages (10 images total).
TILING_GRID = (2, 2)

# When False, skips the cv2 CLAHE + bleed-suppression + unsharp pipeline and
# falls back to the original basic PIL enhancement (brightness / contrast / inversion).
# Useful for clean, already high-contrast input images.
IMAGE_ENHANCE = False

GOOGLE_MODEL         = "gemini-3.5-flash-lite"
GOOGLE_THINKING_TYPE = "native"
GOOGLE_VISION        = True

NIM_TEXT_MODEL    = "deepseek-ai/deepseek-v4-pro-0813"
NIM_THINKING_TYPE = "native"
NIM_VISION        = False   # set True when using a multimodal NIM model

# -- TOKEN ROUTER backend -----------------------------------------------------
# OpenAI-compatible proxy — same /v1/chat/completions interface, so we reuse
# the NIM streaming parser (SSE format is identical).
TOKEN_ROUTER_BASE_URL     = "https://api.tokenrouter.com/v1"
TOKEN_ROUTER_MODEL        = "deepseek-ai/deepseek-v4-pro-0813"
TOKEN_ROUTER_THINKING_TYPE = "native"   # update to "native" if your chosen model supports it
TOKEN_ROUTER_VISION       = False

# Runtime state — reflects the currently active backend.
# Initialised from NIM (the default). switch_backend() updates these.
ACTIVE_BACKEND  = "nim"
THINKING_TYPE   = NIM_THINKING_TYPE
SUPPORTS_VISION = NIM_VISION

# -- NIM MODEL QUIRKS ----------------------------------------------------------
# Some models on NIM need extra request-body fields to enable their thinking
# mode, and NVIDIA doesn't standardize this across model families. Keyed by a
# lowercase substring matched against the model name — llm_backends.py checks
# these in order and applies the first match. Add a new model's quirk here;
# no code change is needed in llm_backends.py.
NIM_MODEL_QUIRKS = {
    # DeepSeek: enables max-thinking mode. NVIDIA recommends this for
    # deepseek-v4-pro to unlock the full extended reasoning budget.
    "deepseek": {"chat_template_kwargs": {"thinking": True,"reasoning_effort":"max"}},

    # MiniMax: thinking is off/default unless explicitly requested via
    # chat_template_kwargs.thinking_mode. NVIDIA's own example sends
    # thinking_mode="enabled" — without this key MiniMax M3 was not being
    # asked to think at all.
    # NOTE: this only fixes the *request* side. M3 still streams its
    # reasoning as inline <mm:think>...</mm:think> tags inside delta.content
    # rather than delta.reasoning/reasoning_content — that parsing fix is
    # separate, in llm_backends._parse_nim_streaming_response.
    "minimax": {"chat_template_kwargs": {"thinking_mode": "enabled"}},

    # Moonshot / Kimi (k3): reasoning_effort is a TOP-LEVEL request field for
    # this model family — NOT nested inside chat_template_kwargs like
    # DeepSeek/MiniMax above. Confirmed from NVIDIA's own kimi-k3 example
    # payload. NOTE: older Kimi versions (k2.5/k2.6) use a different toggle
    # (chat_template_kwargs.thinking: true/false) and don't support
    # reasoning_effort at all — if NIM_TEXT_MODEL ever moves off k3 back to
    # k2.x, this entry will need to change too.
    "kimi": {"reasoning_effort": "max"},
}

# -- LOCAL (llama.cpp / llama-server) BACKEND ---------------------------------
# Point at your own llama-server instance. -c 6000 on the server side; nothing
# here needs to match that number, it's just the client-side endpoint config.
LOCAL_BASE_URL      = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_MODEL         = os.environ.get("LOCAL_MODEL", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL")
LOCAL_THINKING_TYPE = "none"
LOCAL_VISION        = True   # mmproj is loaded server-side
LOCAL_API_KEY_FILE  = os.environ.get("LOCAL_API_KEY_FILE", "")  # usually empty for local

# -- OCR MODEL CHAIN ----------------------------------------------------------
# Each provider gets an ORDERED list of models to try. A model that gets
# rate-limited/overloaded just moves to the next one in its own provider's
# list — no code change needed to add or remove a model, just edit the list.
#   - Keep exactly one entry to pin a single model (a "singleton list").
#   - Add more entries (e.g. when a new/better image model releases) and the
#     dispatcher will automatically fall through to them in order.
# The dispatcher (image_ocr_agent.py) also picks between NIM vs Google based
# on live circuit-breaker/capacity state — these lists only control which
# model(s) get tried WITHIN a given provider once that provider is chosen.
OCR_NIM_MODELS = [
    
]

OCR_GOOGLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemma-4-31b-it",
  
]

# Max tokens the OCR model may output per page.
OCR_MAX_TOKENS           = 50000

# Hard cap on the OCR thread pool inside PDFIngestAgent.
# The AIMD gates limit actual API concurrency independently per provider
# (NIM: MAX_CONCURRENCY, Google: GOOGLE_MAX_CONCURRENCY — see image_ocr_agent.py);
# capping threads here prevents OS resource waste on large PDFs.
PDF_OCR_MAX_WORKERS      = 16

# -- API ENDPOINTS ------------------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# -- API KEY FILES ------------------------------------------------------------
# Paths are loaded from .env — see .env.example for the variable names.
# Each file should contain just the raw key string, nothing else.
GOOGLE_API_KEY_FILE           = os.environ.get("GOOGLE_API_KEY_FILE", "")
# Pool of Google API keys for round-robin rotation on 429.
# File should contain one key per line (blank lines are ignored).
GOOGLE_API_KEY_FILE_POOL      = os.environ.get("GOOGLE_API_KEY_FILE_POOL", "")
NVIDIA_API_KEY_FILE           = os.environ.get("NVIDIA_API_KEY_FILE", "")
LINKUP_API_KEY_FILE           = os.environ.get("LINKUP_API_KEY_FILE", "")
CLOUDFLARE_API_KEY_FILE       = os.environ.get("CLOUDFLARE_API_KEY_FILE", "")
TOKEN_ROUTER_API_KEY_FILE     = os.environ.get("TOKEN_ROUTER_API_KEY_FILE", "")
SEMANTIC_SCHOLAR_API_KEY_FILE = os.environ.get("SEMANTIC_SCHOLAR_API_KEY_FILE", "")
# Mistral Codestral Embed — separate provider from NVIDIA, used only for
# semantic code search (workspace_search semantic=true). See _embed_code()
# in utils.py — the API shape (endpoint, payload keys) differs from the
# NVIDIA _embed() used everywhere else.
MISTRAL_API_KEY_FILE          = os.environ.get("MISTRAL_API_KEY_FILE", "")

# -- LIMITS & TUNING ----------------------------------------------------------
EMPEROR_MAX_TOKENS  = 150000

EMPEROR_STATEFUL_TEMP  = 1.025

EMPEROR_STATEFUL_TOP_P  = 1.0

EMPEROR_REVIEW_TEMP     = 1.025  # slightly higher entropy for self-critique turns

RERANKER_MODEL              = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RERANKER_URL                = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"
# NVIDIA reranker hard limit is 512 tokens per passage.
# 1800 chars ≈ 450 tokens @ 4 chars/token — comfortably under the limit.
RERANKER_MAX_PASSAGE_CHARS  = 8000

TOKEN_SAFETY_MARGIN    = 1.05

# -- EMBEDDING & CHUNKING -----------------------------------------------------
EMBED_MODEL           = "nvidia/nemotron-3-embed-1b"
EMBED_CHUNK_SIZE      = 1600   # chars (~400 tokens at 4 chars/token)
EMBED_CHUNK_OVERLAP   = 200    # chars overlap between consecutive chunks
EMBED_MIN_CHUNK_CHARS = 100    # discard chunks shorter than this
EMBED_BATCH_SIZE      = 32     # passages per embedding API call

# -- CODE SEARCH (SEMANTIC) ---------------------------------------------------
# Powers workspace_search(semantic=true). Separate model/provider from
# EMBED_MODEL above (Nemotron/NVIDIA, used for doc_search + search_history) —
# see the "Critical Discovery" section of implementation_plan.md for why
# Codestral Embed needs its own request shape and its own embed function.
CODE_EMBED_MODEL         = "codestral-embed"          # Mistral model name
CODE_EMBED_BASE_URL      = "https://api.mistral.ai/v1"
CODE_EMBED_DIMENSION     = 1024                       # Matryoshka: 256/512/1024/1536/3072
CODE_EMBED_BATCH_SIZE     = 64                        # item cap per request
# Codestral Embed's documented context is 8192 tokens/input (Mistral's own
# Codestral Embed page + OpenRouter's model listing both confirm this; the
# API truncates a single over-limit input server-side rather than erroring —
# see the "built-in 8192-token truncation" note on _embed_code() in utils.py).
# Mistral doesn't publish a separate aggregate-per-request limit for multiple
# batched inputs, so CODE_EMBED_TOKEN_CAP stays well under 8192 as a
# conservative stand-in for that unknown number, and CODE_EMBED_MAX_CHUNK_TOKENS
# guards the true per-item ceiling directly. Both are estimated via tiktoken's
# cl100k_base (see _estimate_code_tokens in utils.py) — a proxy for Codestral's
# own (non-public) tokenizer, not ground truth — hence the headroom on each.
CODE_EMBED_TOKEN_CAP      = 6000     # est. tokens summed across one batched request
CODE_EMBED_MAX_CHUNK_TOKENS = 7500   # est. tokens for any single chunk — above this,
                                      # skip it outright instead of risking silent
                                      # server-side truncation of the tail
CODE_INDEX_DIR           = os.path.join(DATABASE_DIR, "code_index")
os.makedirs(CODE_INDEX_DIR, exist_ok=True)
CODE_CHUNK_MAX_LINES     = 120    # cap per chunk (prevents giant classes becoming one chunk)
CODE_CHUNK_MIN_LINES     = 3      # discard trivially short chunks
CODE_INDEX_EXTENSIONS    = {".py", ".js", ".ts", ".go", ".c", ".cpp", ".h", ".java",
                            ".rs", ".rb", ".php", ".sh", ".sql", ".md", ".txt",
                            ".yaml", ".yml", ".toml", ".json", ".css", ".html"}
CODE_INDEX_EXCLUDE       = {".env", ".gitignore"}     # never index these, regardless of extension
CODE_INDEX_EXCLUDE_DIRS  = {".git", "__pycache__", "node_modules", ".venv", "venv"}

# -- PDF OCR BATCHING ---------------------------------------------------------
PDF_OCR_BATCH_SIZE  = 4        # pages per parallel OCR batch
PDF_OCR_BATCH_SLEEP = 3.0      # seconds between batches (keeps RPM under limit)

# -- TURN BACKUP --------------------------------------------------------------
# If scratch/ exceeds this size (MB) the per-turn zip backup is skipped to
# avoid blocking the REPL. A warning is printed instead.
BACKUP_MAX_SCRATCH_MB = 150

# -- GENERATION LIMITS --------------------------------------------------------
MAX_HISTORY_TURNS       = 150
MAX_TOOL_CALLS_PER_TURN = 100

# -- WEB CACHE ----------------------------------------------------------------
WEB_CACHE_TTL = 24 * 60 * 60 * 30   # 30 days in seconds

# -- OCR RETRY CHAIN ----------------------------------------------------------
OCR_RETRIES_PER_TIER = 2

# -- PATH TRANSLATION ---------------------------------------------------------
# The model always uses container paths in tool arguments:
#   /workspace/scratch/...  ->  SCRATCH_DIR
#   /uploads/...            ->  UPLOADS_FOLDER
#   /outputs/...            ->  OUTPUTS_DIR
#
# All host-side file operations must translate before touching the filesystem.

def get_active_model() -> str:
    """Return the model string for the currently active backend."""
    if ACTIVE_BACKEND == "nim":
        return NIM_TEXT_MODEL
    if ACTIVE_BACKEND == "google":
        return GOOGLE_MODEL
    if ACTIVE_BACKEND == "cloudflare":
        return CLOUDFLARE_MODEL
    if ACTIVE_BACKEND == "local":
        return LOCAL_MODEL
    if ACTIVE_BACKEND == "tokenrouter":
        return TOKEN_ROUTER_MODEL
    return NIM_TEXT_MODEL  # fallback


def container_to_host_path(path: str) -> str:
    """
    Translate a container-side absolute path to its host-side equivalent.
    Returns the path unchanged if it does not match any known container prefix.
    """
    _map = [
        (CONTAINER_SCRATCH, SCRATCH_DIR),    # /workspace/scratch -> host scratch
        ("/uploads",        UPLOADS_FOLDER), # /uploads           -> host uploads
        ("/outputs",        OUTPUTS_DIR),    # /outputs           -> host outputs
    ]
    for c_prefix, h_prefix in _map:
        if path == c_prefix or path.startswith(c_prefix + "/"):
            rel = path[len(c_prefix):].lstrip("/")
            return os.path.join(h_prefix, rel) if rel else h_prefix
    return path


def host_to_container_path(path: str) -> str:
    """
    Translate a host-side absolute path to its container-side equivalent.
    Reverse of container_to_host_path(). Used by code_search_agent.py so the
    index stores container-style paths (/workspace/scratch/...) — the same
    form the model already uses in tool calls — instead of host paths.
    Returns the path unchanged if it does not match any known host prefix.
    """
    _map = [
        (SCRATCH_DIR,    CONTAINER_SCRATCH),
        (UPLOADS_FOLDER, "/uploads"),
        (OUTPUTS_DIR,    "/outputs"),
    ]
    normed = os.path.normpath(path)
    for h_prefix, c_prefix in _map:
        h_normed = os.path.normpath(h_prefix)
        if normed == h_normed or normed.startswith(h_normed + os.sep):
            rel = os.path.relpath(normed, h_normed).replace("\\", "/")
            return f"{c_prefix}/{rel}" if rel != "." else c_prefix
    return path