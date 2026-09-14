# utils.py - Common utilities used across agents
import re
import os
import sys
import json
import time
import functools
import importlib.util
import tiktoken
import requests
import numpy as np
from urllib.parse import urlparse
from rich.console import Console
import config

# Temporary fix for catppuccin + matplotlib bug (#129)
#
# QA-2 fix, part 1: the patch + import used to be a bare sequence — if the
# catppuccin import raised, find_spec stayed monkey-patched for the rest of
# the process, silently blocking matplotlib detection everywhere else.
# try/finally guarantees the restore runs no matter what happens inside.
#
# QA-2 fix, part 2: decouple startup from catppuccin's availability entirely.
# rich.console.Console accepts theme=None and just falls back to its own
# default colours, so a missing package or a changed API surface shouldn't
# be able to crash the whole application at import time.
_orig_find_spec = importlib.util.find_spec
try:
    try:
        importlib.util.find_spec = lambda name, *args, **kwargs: (
            None if name == "matplotlib" else _orig_find_spec(name, *args, **kwargs)
        )
        # pyrefly: ignore [missing-import]
        from catppuccin.extras.rich_ctp import mocha
    finally:
        importlib.util.find_spec = _orig_find_spec
except ImportError:
    mocha = None

console = Console(theme=mocha, force_terminal=True, color_system="truecolor")

def read_api_key(filepath: str, silence_warning: bool = False) -> str:
    """Read an API key from a file safely, returning empty string on failure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        if not silence_warning:
            name = os.path.basename(filepath)
            console.print(f"⚠️  [yellow]API key file not found: {name}[/yellow]")
        return ""
    except Exception as e:
        console.print(f"🚨 [bold red]Error reading {filepath}: {e}[/bold red]")
        return ""

def robust_rmtree(path: str):
    """Safely remove a directory tree, handling Windows read-only locks."""
    import shutil
    import stat

    def _on_rm_error(func, fpath, exc_info):
        """Clear read-only flag and retry — body is identical for onerror/onexc."""
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    for _ in range(3):
        if not os.path.exists(path):
            return
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_on_rm_error)
        else:
            shutil.rmtree(path, onerror=_on_rm_error)
        if not os.path.exists(path):
            return
        time.sleep(0.5)
    # Final fallback, just ignore errors
    shutil.rmtree(path, ignore_errors=True)


def _force_remove(path: str) -> None:
    """
    Remove a file, clearing the read-only attribute first on Windows if needed.
    Used by /restore and /reset scratch cleanup to handle read-only files that
    os.remove() would otherwise fail on with PermissionError.
    """
    import stat as _stat
    try:
        os.remove(path)
    except PermissionError:
        try:
            os.chmod(path, _stat.S_IWRITE)
            os.remove(path)
        except Exception:
            pass  # best-effort — log was already printed by caller
    except Exception:
        pass

# Shared language extension map (used by file_ops_agent, workspace_tracker, etc.)
_LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.html': 'html',
    '.css': 'css',
    '.json': 'json',
    '.md': 'markdown',
    '.sh': 'bash',
    '.sql': 'sql',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.txt': 'text',
}

# Shared MIME type map for image extensions.
# Single source of truth — imported by start.py and image_ocr_agent.py
# so there is only one place to update when adding a new format.
MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".avif": "image/avif",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}

def mime_type_from_url(url: str, default: str = "image/jpeg") -> str:
    """
    Infer an image MIME type from a URL's path extension, ignoring any query
    string or fragment (e.g. 'photo.png?w=800' -> '.png', not '.png?w=800').
    Falls back to `default` when the extension is missing or unrecognised —
    e.g. dynamic image URLs with no file extension at all.
    """
    try:
        ext = os.path.splitext(urlparse(url).path)[1].lower()
    except Exception:
        ext = ""
    return MIME_MAP.get(ext, default)

def detect_language(filepath: str) -> str:
    """Detect programming language from file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return _LANG_MAP.get(ext, 'text')

# Single source of truth for "this is a source/text file we understand" —
# derived directly from _LANG_MAP so the two can never drift apart.
# tool_handlers.UPLOAD_EXTENSIONS and workspace_tracker._RECONCILE_EXTENSIONS
# both build on top of this base set instead of hand-maintaining their own
# copies of the same core list.
CODE_EXTENSIONS = frozenset(_LANG_MAP.keys())

def extract_thinking_tags(text: str) -> tuple:
    """
    Extract content from <think> or <reasoning> tags (case-insensitive).

    Some backends leak extended thinking as inline tags in content instead
    of using a structured reasoning field — MiniMax does this with <think>,
    and Gemini inconsistently does the same with <reasoning> (on top of its
    normal structured `thought` part). Both are treated identically here.

    Returns:
        Tuple of (thinking_content or None, text_without_thinking)
    """
    if not isinstance(text, str):
        return None, str(text or "")

    tag_pattern = re.compile(r"<(think|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
    think_match = tag_pattern.search(text)

    if think_match:
        thinking = think_match.group(2).strip()
        clean = tag_pattern.sub("", text).strip()
        return thinking, clean

    return None, text

def extract_keywords(text: str) -> list:
    """Extract meaningful keywords, excluding common stop words."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'is', 'are', 'was', 'write', 'code', 'file', 'find',
        'create', 'make', 'generate', 'using', 'helper', 'script', 'simple',
        'just', 'recent', 'latest', 'last', 'now', 'created', 'made', 'wrote'
    }
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return list({w for w in words if w not in stop_words})

# ══════════════════════════════════════════════════════════════════════════════
# TOKEN COUNTER
# ══════════════════════════════════════════════════════════════════════════════

class TokenCounter:
    """
    Tracks and accumulates token usage across API calls during a session,
    estimating limits and preserving max context thresholds.
    """
    def __init__(self, model_name="gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.current_session_output = 0

    def count_tokens(self, text) -> int:
        """Count tokens in text or multi-modal content list."""
        if not text:
            return 0

        if isinstance(text, list):
            total = 0
            for item in text:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        total += len(self.encoding.encode(item.get('text', '')))
                    elif item.get('type') == 'image_url':
                        total += 85  # Approximate image token cost
                elif isinstance(item, str):
                    total += len(self.encoding.encode(item))
            return int(total * config.TOKEN_SAFETY_MARGIN)

        if isinstance(text, str):
            return int(len(self.encoding.encode(text)) * config.TOKEN_SAFETY_MARGIN)

        return 0

    def count_messages(self, messages) -> dict:
        """Count tokens in a message list."""
        total = 0
        for message in messages:
            total += 4  # Per-message overhead
            if 'role' in message:
                total += len(self.encoding.encode(message['role']))
            if 'content' in message:
                total += self.count_tokens(message['content'])

        return {"total": total, "messages": len(messages)}

    def track_output(self, response_text: str):
        """
        Track only output tokens from a single API response.
        Called on every loop iteration in _generate_with_tools so all
        intermediate tool-calling responses are counted, not just the final answer.
        """
        output_count = self.count_tokens(response_text)
        self.current_session_output += output_count

    def reset_session(self):
        """Reset session token counters."""
        self.current_session_output = 0
        console.print("[dim]Session tokens reset[/dim]")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def auto_enhance(image_path: str) -> list[str]:
    """
    Auto-detect and fix common image quality issues before the model sees the image.

    Two problems handled:
      1. Dark background  — avg pixel brightness < 80 → invert + mild contrast boost
      2. Low contrast     — pixel range < 100 AND image is not a natural photo → contrast boost

    Modifies the file in-place. Returns list of operation strings applied, or [] if untouched.
    Non-fatal — original image is unchanged if an exception occurs before save().
    """
    if not image_path or not os.path.isfile(image_path):
        return []

    # Skip URLs and data URIs — nothing to do on them here
    if image_path.startswith("http") or image_path.startswith("data:"):
        return []

    try:
        import numpy as np
        from PIL import Image, ImageOps, ImageEnhance
    except ImportError:
        # PIL/numpy not installed — silently skip, OCR still works without preprocessing
        return []

    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        applied: list[str] = []

        # ── Check 1: Dark background (brightness + saturation guard) ──────────────────────────────────
        # Only invert if the image is dark AND low-saturation (i.e. dark-mode text, chalkboard,
        # monochrome document). A dark natural photo (night scene, black furniture) will have
        # higher saturation and should NOT be inverted.
        avg_brightness = arr.mean()

        if avg_brightness < 80:
            # Compute per-pixel saturation: (max_channel - min_channel) / max_channel
            arr_norm = arr / 255.0
            max_c = arr_norm.max(axis=2)
            min_c = arr_norm.min(axis=2)
            sat_map = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)
            avg_saturation = sat_map.mean()  # 0.0 – 1.0

            # Threshold: saturation < 0.12 ≈ 30/255 in OpenCV HSV — safe for b&w docs
            if avg_saturation < 0.12:
                img = ImageOps.invert(img.convert("RGB"))
                img = ImageEnhance.Contrast(img).enhance(1.3)
                applied.append(f"inverted(brightness={avg_brightness:.0f}, sat={avg_saturation:.3f})")
                applied.append("contrast(1.3)")
                # Recompute array after inversion for the contrast check below
                arr = np.array(img, dtype=np.float32)

        # ── Check 2: Low contrast (non-photo only) ──────────────────────────────────────────
        # Per-channel std dev > 55 → likely a natural photo → skip to avoid artefacts
        per_channel_std = arr.reshape(-1, 3).std(axis=0).mean()
        is_natural_photo = per_channel_std > 55

        if not is_natural_photo:
            grey = arr.mean(axis=2)          # luminance proxy
            pixel_range = grey.max() - grey.min()

            if pixel_range < 100:
                img = ImageEnhance.Contrast(img).enhance(1.5)
                applied.append(f"contrast(1.5, range={pixel_range:.0f})")

        # ── Save if anything changed ──────────────────────────────────────────────────────────────────
        if applied:
            img.save(image_path)
            console.print(
                f"   [dim cyan]auto_enhance: {os.path.basename(image_path)} "
                f"→ {', '.join(applied)}[/dim cyan]"
            )

        return applied

    except Exception as e:
        # Non-fatal — original image is untouched if we crash before save()
        console.print(f"   [dim yellow]auto_enhance skipped ({e})[/dim yellow]")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL HISTORY
# ══════════════════════════════════════════════════════════════════════════════

class GlobalHistoryWriter:
    """
    Appends each user+assistant turn to a permanent per-project .jsonl file
    inside config.GLOBAL_HISTORIES_DIR. Non-fatal — never raises.
    """

    def __init__(self):
        try:
            os.makedirs(config.GLOBAL_HISTORIES_DIR, exist_ok=True)
        except Exception as e:
            console.print(f"[dim yellow]global_history: could not create dir: {e}[/dim yellow]")

        project_name  = os.path.basename(os.path.normpath(config.WORKSPACE_ROOT))
        self.filepath = os.path.join(config.GLOBAL_HISTORIES_DIR, f"{project_name}.jsonl")

    def write_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Append one user+assistant turn to this project's global .jsonl file.
        Non-fatal — logs warning on failure, never raises.

        Both lines are written as a single f.write() call so a process kill
        between the two writes can never leave an orphaned user-only line that
        would corrupt downstream search_history pairing.
        """
        try:
            line_u = json.dumps({"role": "user",      "content": user_query})
            line_a = json.dumps({"role": "assistant",  "content": assistant_response})
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line_u + "\n" + line_a + "\n")
            return True
        except Exception as e:
            console.print(f"[dim yellow]global_history write failed: {e}[/dim yellow]")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# ATOMIC JSON PERSISTENCE
#
# Single implementation of "read JSON, tolerate any error" and "write JSON
# safely via tmp-file + os.replace" — previously hand-rolled independently
# in web_agent (cache), workspace_tracker (registry), and turn_state_manager
# (ledger), with only the registry/ledger versions actually being atomic.
# ══════════════════════════════════════════════════════════════════════════════

def load_json(filepath: str, default=None):
    """
    Load a JSON file, returning `default` on any error (missing file,
    corrupt JSON, permissions, etc). Never raises.
    """
    if default is None:
        default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"⚠️  [yellow]utils.load_json: failed to load {os.path.basename(filepath)}: {e}[/yellow]")
        return default

def save_json_atomic(filepath: str, data) -> bool:
    """
    Write `data` as JSON to `filepath` atomically (write to a .tmp file,
    then os.replace it into place) so a crash mid-write never leaves a
    truncated/corrupt file behind. Returns True on success, False on error.
    """
    try:
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
        return True
    except Exception as e:
        console.print(f"⚠️  [yellow]utils.save_json_atomic: failed to save {os.path.basename(filepath)}: {e}[/yellow]")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA RERANKING
#
# Cross-encoder reranking via NVIDIA NIM API.
# ══════════════════════════════════════════════════════════════════════════════

def rerank_passages(query: str, passages: list, api_key: str = None, label: str = "items") -> list:
    """
    Return the indices of `passages` reordered by relevance to `query`,
    most relevant first, using the NVIDIA cross-encoder reranker.

    Falls back to the original order (list(range(len(passages)))) if no API
    key is available, there's nothing to rerank, or the API call fails —
    never raises.

    Args:
        query:    the search query text
        passages: list of passage strings to rerank
        api_key:  NVIDIA API key. If omitted, read from config.NVIDIA_API_KEY_FILE.
        label:    noun used in the console status messages (e.g. "snippets", "history turn(s)")
    """
    if api_key is None:
        api_key = read_api_key(config.NVIDIA_API_KEY_FILE, silence_warning=True)

    original_order = list(range(len(passages)))

    if not api_key or len(passages) <= 1:
        return original_order

    try:
        console.print(f"[dim]Reranking {len(passages)} {label}...[/dim]")

        # Truncate passages to the reranker's token limit before submission.
        # The NVIDIA reranker silently truncates passages beyond 512 tokens,
        # which degrades ranking quality non-obviously. Pre-truncating here
        # gives the model consistent, predictable inputs.
        _max_chars = getattr(config, "RERANKER_MAX_PASSAGE_CHARS", 1800)
        safe_passages = [p[:_max_chars] for p in passages]

        resp = requests.post(
            config.RERANKER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept":        "application/json",
            },
            json={
                "model":    config.RERANKER_MODEL,
                "query":    {"text": query},
                "passages": [{"text": p} for p in safe_passages],
            },
            timeout=30,
        )
        resp.raise_for_status()

        rankings = resp.json().get("rankings", [])
        if not rankings:
            console.print("⚠️  [yellow]Reranker returned empty rankings — using original order[/yellow]")
            return original_order

        order = [r["index"] for r in sorted(rankings, key=lambda x: x["logit"], reverse=True)]
        console.print(f"[dim]Reranked — top result was originally #{order[0] + 1}[/dim]")
        return order

    except Exception as e:
        console.print(f"⚠️  [yellow]Reranker failed (using original order): {e}[/yellow]")
        return original_order


# ══════════════════════════════════════════════════════════════════════════════
# LINE-RANGE CLIPPING
#
# Shared "expand [start, end] by ±context lines, clipped to the file bounds" math.
# ══════════════════════════════════════════════════════════════════════════════

def compute_view_window(total_lines: int, start: int, end: int, context: int) -> tuple:
    """
    Compute the 0-based [start_idx, end_idx) window to display for a
    view_lines-style request.

    Args:
        total_lines: total number of lines in the file
        start:       1-indexed requested start line (clamped to >= 1)
        end:         1-indexed requested end line, or None/0 for "to end of file"
        context:     extra lines of context to include before/after

    Returns:
        (start_idx, end_idx) — 0-based, end_idx exclusive, both clipped to
        [0, total_lines].
    """
    start_idx = max(0, max(1, start) - 1 - context)
    end_idx   = min(total_lines, (end or total_lines) + context)
    return start_idx, end_idx


# ══════════════════════════════════════════════════════════════════════════════
# EXPONENTIAL BACKOFF
#
# Shared exponential backoff used by LLM backend retry loops.
# ══════════════════════════════════════════════════════════════════════════════

def backoff_wait(attempt: int, base_delay: int = 3, reason: str = "Rate limit", max_attempts: int = None):
    """
    Sleep for an exponentially increasing delay (base_delay * 2**attempt),
    printing a status line first. `attempt` is 0-indexed.
    """
    wait = min(base_delay * (2 ** attempt), 60)  # cap at 60s to prevent runaway sleeps
    suffix = f" ({attempt + 1}/{max_attempts})" if max_attempts else ""
    console.print(f"⏳ [yellow]{reason}. Retrying in {wait}s...{suffix}[/yellow]")
    time.sleep(wait)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED API KEY READER
#
# Single per-process cache for key files — avoids each module independently
# re-reading the same file and maintaining its own cache dict/global.
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=None)
def read_api_key_cached(filepath: str) -> str:
    """Read an API key from a file, caching the result for the process lifetime.
    Silences the missing-file warning — callers check the empty-string result.
    """
    return read_api_key(filepath, silence_warning=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SECURITY & EXCLUSION CONSTANTS
#
# Centralised here so a single audit covers all security-sensitive patterns
# and all "ephemeral / ignore" directory sets across the project.
# ══════════════════════════════════════════════════════════════════════════════

# Bash hard blocklist — patterns that would cause catastrophic damage inside
# the Docker sandbox. Defined here (not config.py) because these are security
# primitives, not operator-tunable settings.
BASH_BLOCKLIST = [
    # rm with recursive+force flags targeting root or home (catches -rf, -fr, -r -f, -f -r, etc.)
    r"\brm\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+[/~]",  # -rf or -rxf / ~
    r"\brm\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+[/~]",  # -fr or -fxr / ~
    r"\brm\s+-[rR]\s+-[fF]\s+[/~]",   # rm -r -f / or rm -r -f ~
    r"\brm\s+-[fF]\s+-[rR]\s+[/~]",   # rm -f -r / or rm -f -r ~
    r":\(\)\{\s*:\|:&\s*\};:",         # fork bomb
    r">\s*/dev/sd[a-z]",               # raw disk write
    r"\bmkfs\.",                       # format a filesystem
    r"\bdd\s+if=/dev/zero\s+of=/dev/"  # wipe block device
]

# Zip / walk exclusion sets — shared between turn_state_manager (backup zips)
# and workspace_tracker (reconcile walks) so both lists can't drift apart.
EXCL_PREFIXES = (
    "enhanced_",   # enhanced_* images  (auto-generated)
    "overview_",   # overview_* PDF tile images (auto-generated by PDF OCR)
    "tile_",       # tile_* PDF tile images (auto-generated by PDF OCR)
    "master_",     # master_* PDF master images (auto-generated by PDF OCR)
)
EXCL_SUFFIXES = ("_backup",)            # scratch_backup dirs (Ctrl+C rollback)
EXCL_DIRS     = {
    "__pycache__", "node_modules", "venv", "env", ".venv",
    ".git", "target", "build", "dist",
}


# ══════════════════════════════════════════════════════════════════════════════
# NVIDIA EMBEDDING API
#
# Moved here from doc_search_agent.py so tool_handlers.py can import them
# without creating an upward dependency into a domain module.
# ══════════════════════════════════════════════════════════════════════════════

def _embed(texts: list, input_type: str = "passage") -> list:
    """
    Embed a list of strings using the configured NVIDIA embed model.

    Args:
        texts:      List of strings to embed.
        input_type: "passage" at ingest time, "query" at search time.
                    The bi-encoder uses different projections for each.

    Returns:
        List of embedding vectors (list of float), same order as input.

    Raises:
        RuntimeError on non-200 response (caller handles retries).
    """
    api_key = read_api_key_cached(config.NVIDIA_API_KEY_FILE)

    payload = {
        "model":      config.EMBED_MODEL,
        "input":      texts,
        "input_type": input_type,
        "truncate":   "END",       # silently truncate beyond 512 tokens
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{config.NVIDIA_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=120,
            )

            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                console.print(f"[yellow]⏳ Embeddings rate-limited. Waiting {wait}s...[/yellow]")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                console.print(f"[yellow]⚠️  Embeddings server error {resp.status_code}. Retrying...[/yellow]")
                time.sleep(3)
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:200]}")

            data  = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]

        except RuntimeError:
            raise
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Embedding request failed: {e}")
            time.sleep(2)

    raise RuntimeError("Embedding: max retries exceeded")


def _embed_batched(texts: list, input_type: str = "passage") -> list:
    """
    Embed texts in batches of config.EMBED_BATCH_SIZE to respect API limits.
    Returns a flat list of embeddings in original order.
    """
    all_embeddings = []
    batch_size     = config.EMBED_BATCH_SIZE
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        console.print(
            f"   [dim]Embedding batch {i // batch_size + 1}/"
            f"{(len(texts) + batch_size - 1) // batch_size} "
            f"({len(batch)} chunks)...[/dim]"
        )
        try:
            embeddings = _embed(batch, input_type=input_type)
            all_embeddings.extend(embeddings)
        except Exception as e:
            console.print(f"   [red]Batch {i // batch_size + 1} failed: {e}[/red]")
            all_embeddings.extend([None] * len(batch))
        # Small sleep between batches to be rate-limit friendly
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return all_embeddings


# ══════════════════════════════════════════════════════════════════════════════
# MISTRAL CODESTRAL EMBED API
#
# Separate from _embed()/_embed_batched() above (NVIDIA Nemotron, used for
# doc_search + search_history). Codestral Embed is served from Mistral's own
# endpoint with a different request/response shape:
#   - "input" (singular) in the raw REST body — the Python SDK uses "inputs"
#     but we use requests.post directly
#   - no input_type — same model/projection for query and passage text
#   - "output_dimension" selects a Matryoshka-truncated embedding size
#   - built-in 8192-token truncation server-side, no "truncate" flag needed
# Used exclusively by code_search_agent.py for workspace_search(semantic=true).
# ══════════════════════════════════════════════════════════════════════════════

def _embed_code(texts: list) -> list:
    """
    Embed code snippets via Mistral Codestral Embed.

    Args:
        texts: List of strings to embed (raw code or enriched embed_text).

    Returns:
        List of embedding vectors (list of float), same order as input.

    Raises:
        RuntimeError on non-200 response (caller handles retries/fallback).
    """
    api_key = read_api_key_cached(config.MISTRAL_API_KEY_FILE)
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY_FILE is not configured — semantic code search "
            "needs a Mistral API key. Set MISTRAL_API_KEY_FILE in .env."
        )

    payload = {
        "model":            config.CODE_EMBED_MODEL,
        "input":            texts,                      # REST API uses 'input' (singular); SDK uses 'inputs'
        "output_dimension": config.CODE_EMBED_DIMENSION,
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{config.CODE_EMBED_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=120,
            )

            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                console.print(f"[yellow]⏳ Codestral Embed rate-limited. Waiting {wait}s...[/yellow]")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                console.print(f"[yellow]⚠️  Codestral Embed server error {resp.status_code}. Retrying...[/yellow]")
                time.sleep(3)
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"Codestral Embed API error {resp.status_code}: {resp.text[:200]}")

            data  = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]

        except RuntimeError:
            raise
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Codestral Embed request failed: {e}")
            time.sleep(2)

    raise RuntimeError("Codestral Embed: max retries exceeded")


_CL100K_ENCODING = None


def _estimate_code_tokens(text: str) -> int:
    """
    Rough token count for batch-sizing purposes only — cl100k_base as a
    stand-in for Codestral's own (non-public) tokenizer. Good enough to keep
    requests under the server-side limit with headroom; not used for billing
    or anything that needs to be exact.
    """
    global _CL100K_ENCODING
    if _CL100K_ENCODING is None:
        _CL100K_ENCODING = tiktoken.get_encoding("cl100k_base")
    return len(_CL100K_ENCODING.encode(text or ""))


def _embed_code_range(texts: list, start: int, end: int, out: list) -> None:
    """
    Embed texts[start:end] into out[start:end] in place. On failure, bisect
    the range and retry each half — so one oversized/malformed chunk in a
    64-item batch only costs that one chunk (a None placeholder), not the
    other ~63. A range of size 1 that still fails is the base case.
    """
    batch = texts[start:end]
    try:
        embeddings = _embed_code(batch)
        out[start:end] = embeddings
    except Exception as e:
        if end - start <= 1:
            console.print(f"   [red]Code embed failed for 1 chunk: {e}[/red]")
            out[start] = None
            return
        mid = start + (end - start) // 2
        _embed_code_range(texts, start, mid, out)
        _embed_code_range(texts, mid, end, out)


def _embed_code_batched(texts: list) -> tuple:
    """
    Embed code chunk texts in batches capped by item count
    (config.CODE_EMBED_BATCH_SIZE), by estimated aggregate token count
    (config.CODE_EMBED_TOKEN_CAP), and by estimated per-chunk token count
    (config.CODE_EMBED_MAX_CHUNK_TOKENS) — see config.py for why each of
    these is sized the way it is. tiktoken's cl100k_base is used as an
    estimate throughout, not ground truth, since Codestral's own tokenizer
    isn't public.

    Any single chunk whose estimate exceeds CODE_EMBED_MAX_CHUNK_TOKENS is
    skipped before it's ever sent: Codestral Embed truncates an over-limit
    input server-side rather than rejecting it (see _embed_code's
    docstring), so sending it "succeeds" but silently embeds only the first
    ~8192 tokens — a quality regression with no error to catch. Skipping it
    up front turns that into a visible None instead.

    Returns (embeddings, oversized_indices):
      - embeddings:        flat list of embeddings (or None) in original order.
        On a batch failure (e.g. a 429, or a request that still slips past
        the aggregate token cap), the batch is bisected and retried rather
        than discarded outright — only a chunk that still fails on its own
        yields a None placeholder, so one bad chunk doesn't take the rest of
        its batch down with it.
      - oversized_indices: set of original indices skipped by the pre-filter
        above. These are a *permanent*, deterministic skip (the chunk will
        estimate over the ceiling again on every future attempt), unlike a
        None from a transient batch failure, which is worth retrying. Callers
        use this to tell "this chunk can never be embedded, stop retrying it"
        apart from "this chunk failed this time, try again next build."
    """
    all_embeddings: list = [None] * len(texts)
    if not texts:
        return all_embeddings, set()

    item_cap  = config.CODE_EMBED_BATCH_SIZE
    token_cap = config.CODE_EMBED_TOKEN_CAP
    max_chunk = config.CODE_EMBED_MAX_CHUNK_TOKENS

    token_counts = [_estimate_code_tokens(t) for t in texts]

    # ── Pre-filter: chunks too large for the model, full stop ──────────
    indexable = []
    oversized: set = set()
    for i, est in enumerate(token_counts):
        if est > max_chunk:
            console.print(
                f"   [yellow]⚠️ Code chunk #{i} is ~{est} estimated tokens — "
                f"over the {max_chunk}-token safety ceiling for Codestral "
                f"Embed's 8192-token context. Skipping it rather than risking "
                f"silent server-side truncation of the tail.[/yellow]"
            )
            oversized.add(i)
            continue
        indexable.append(i)

    if not indexable:
        return all_embeddings, oversized

    # Build batch boundaries (over the indexable subset) respecting both
    # the item cap and the conservative aggregate token cap.
    batches = []   # list of lists of original indices
    start = 0
    n = len(indexable)
    while start < n:
        end       = start
        tok_count = 0
        while end < n and (end - start) < item_cap:
            t = token_counts[indexable[end]]
            if end > start and tok_count + t > token_cap:
                break
            tok_count += t
            end += 1
        if end == start:
            end = start + 1   # single item already at/over the aggregate cap alone
        batches.append(indexable[start:end])
        start = end

    for bi, idxs in enumerate(batches):
        console.print(
            f"   [dim]Embedding code batch {bi + 1}/{len(batches)} "
            f"({len(idxs)} chunks)...[/dim]"
        )
        batch_texts = [texts[i] for i in idxs]
        out         = [None] * len(idxs)
        _embed_code_range(batch_texts, 0, len(batch_texts), out)
        for local_i, global_i in enumerate(idxs):
            all_embeddings[global_i] = out[local_i]
        if bi + 1 < len(batches):
            time.sleep(0.5)

    return all_embeddings, oversized


def _cosine_similarity(query_vec: list, passage_vecs: list) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix
    of passage vectors. Returns a 1-D array of scores.
    """
    q     = np.array(query_vec,    dtype=np.float32)
    P     = np.array(passage_vecs, dtype=np.float32)
    q    /= (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-10
    P    /= norms
    return P @ q
