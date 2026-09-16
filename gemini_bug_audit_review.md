# Bug Audit Review — Gemini's Findings vs. Actual Code

Verdict key: **✅ Confirmed** — real bug, fix below. **⚠️ Confirmed, overstated impact** — real mechanism, but the consequence is smaller than described. **❌ False positive** — not an issue.

Almost everything Gemini flagged checks out. A few have exaggerated blast radii, and one (#39) is actually worse than described in one part and fine in another. Details below.

---

## 🚨 Category 1: Critical Crashes, Security Leaks & Tool Failures

### 1. Path Traversal via Ingest Tools — ✅ Confirmed
`os.path.join(config.UPLOADS_FOLDER, filename)`: in Python, if `filename` is absolute (`/etc/passwd`, or on Windows `C:\...`), `os.path.join` **discards the first argument** and returns the second unchanged. Since the model fully controls `filename`, this is a real sandbox escape in `_handle_ingest_pdf`, `_handle_ingest_text`, and `_handle_ingest_chat`.

**Fix:** Before joining, reject filenames that are absolute or contain `..`/path separators, e.g.:
```python
filename = os.path.basename(filename)  # strip any directory component
if os.path.isabs(filename) or ".." in filename:
    return "[SYSTEM: ERROR] invalid filename"
```
Better yet, reuse the pattern already in `tool_handlers._is_allowed_read_path`: resolve `os.path.realpath()` on the joined path and verify it still starts with `UPLOADS_FOLDER`/`SCRATCH_DIR`.

**Fix Analysis:** Use the `os.path.basename()` approach, but the "better yet" realpath approach is actually cleaner and already matches the project's established guard pattern. The two-step approach (basename + realpath check) is the right call because `os.path.basename` strips directory components but doesn't prevent creative tricks on Windows (e.g. filenames with trailing slashes). Specifically: apply `os.path.basename(filename)` first to strip path separators, then join with the config folder, then do `os.path.realpath()` and prefix-check — that's the same pattern `_is_allowed_read_path` already uses. Note: `_handle_ingest_pdf` uses a fuzzy `os.walk` fallback if the direct path doesn't exist — this walk needs to be restricted to only return files whose `os.path.realpath` is still under `UPLOADS_FOLDER` too, otherwise the fallback itself is exploitable.

### 2. `/edit` External Editor Broken on POSIX — ✅ Confirmed
`subprocess.call([*editor_cmd, tmp_path], shell=True)` — with `shell=True` and a **list** of args on POSIX, the shell only executes `args[0]` and treats every subsequent list item as a positional parameter (`$0`, `$1`...) to the shell itself, not as arguments to the program. So `nano` launches with no file argument → blank buffer → unmodified original silently read back on exit.

**Fix:** Drop `shell=True` — it's not needed here since you already have a proper argv list:
```python
result = subprocess.call([*editor_cmd, tmp_path])
```

**Fix Analysis:** The proposed fix is correct and sufficient. The `edit_in_external_editor` function at line 207 of `start.py` already constructs `editor_cmd` via `shlex.split(editor_raw)` which gives a proper list, so dropping `shell=True` will make POSIX work correctly. On Windows, `shell=True` with a list does still work (Windows ignores the extra items differently), but `shell=False` is both correct and safer. No other changes needed.

### 3. `<show_image>` Crashes on `/uploads` Paths — ✅ Confirmed
The system prompt (`BASH_TOOLS_PROMPT`) explicitly tells the model `show_image` accepts `/uploads` paths. But `FileOpsAgent._resolve_path` only allows paths under `workspace_root` (scratch) or `OUTPUTS_DIR`. Any `<show_image>/uploads/x.png</show_image>` call raises `PermissionError`, a straight contradiction between prompt and implementation.

**Fix:** Add `UPLOADS_FOLDER` as a third allowed (read-only) root in `_resolve_path`:
```python
uploads_abs = os.path.realpath(config.UPLOADS_FOLDER)
in_uploads = resolved == uploads_abs or resolved.startswith(uploads_abs + os.sep)
if not (in_workspace or in_outputs or in_uploads):
    raise PermissionError(...)
```

**Fix Analysis:** The proposed fix is correct. However, note a naming collision risk: `_resolve_path` is used by both `str_replace` (write operation) and `show_image` (read-only display). Adding `uploads` as an allowed root for `_resolve_path` would also allow `str_replace` to write into `/uploads`, which is undesirable. The better approach is to NOT modify `_resolve_path` generically, and instead handle the `/uploads` case specifically inside `_handle_show_image` itself — translate the path via `container_to_host_path`, check that it's under `UPLOADS_FOLDER` using the same realpath-prefix pattern, and skip the `_resolve_path` call for that case. This keeps write operations restricted while fixing show_image.

### 4. `<show_image>` Crashes with `SameFileError` — ✅ Confirmed
`_handle_show_image` unconditionally does `shutil.copy2(full_path, out_path)`. If `full_path` is already inside `/outputs` (a legitimate path per the prompt), `out_path` resolves to the same file, and `shutil.copy2` raises `SameFileError`.

**Fix:** Skip the copy when src/dst are the same file:
```python
if os.path.exists(out_path) and os.path.samefile(full_path, out_path):
    pass  # already in outputs
else:
    shutil.copy2(full_path, out_path)
```

**Fix Analysis:** The proposed fix is correct. `os.path.samefile` raises `FileNotFoundError` if either path doesn't exist, hence the `os.path.exists(out_path)` guard is needed before calling it (the source `full_path` is already verified to exist at this point). The fix as written handles that correctly. Worth noting: if Bug #3 is fixed by doing the `/uploads` check separately in `_handle_show_image` (rather than modifying `_resolve_path`), the samefile check here also needs to cover the uploads path, which it already does since it only compares `full_path` and `out_path` by inode regardless of which directory they're in.

### 5. `MarkupError`-class Crash in `/turns` on Bash Commands — ✅ Confirmed
`list_turns()` escapes `prompt_preview`, `user_uploads`, `model_modified`, and `outputs_created` via `_esc_markup(...)` — but **not** `model_bash`:
```python
bash = entry.get("model_bash", [])   # <- not escaped, unlike everything else here
...
parts.append(f"[dim]$ {bash[0][:50]}...[/dim]")
```
A command like `pytest -k [test_1]` starts a bracketed run with a lowercase letter, which Rich's markup parser treats as a style tag; an unrecognized style name raises an exception at render time (Rich's own `StyleSyntaxError`/`MissingStyle` family — close enough to "MarkupError" that the crash is real even if the exact class name differs from Gemini's claim). The surrounding code clearly shows this exact risk was anticipated for every *other* field — `bash` was just missed.

**Fix:**
```python
bash = [_esc_markup(b) for b in entry.get("model_bash", [])]
```

**Fix Analysis:** The proposed fix is correct and minimal. Looking at the actual `list_turns()` code in `turn_state_manager.py` (line 807), `bash` is fetched unescaped while every other field is escaped on the same lines. The one-liner list comprehension fix is exactly right. The subsequent usage at lines 842–844 accesses `bash[0]` and `len(bash)`, which continue to work correctly with the list comprehension.

### 6. Uncaught `re.error` in External File Searches — ✅ Confirmed (impact slightly smaller than stated)
`_handle_search_in_file`'s "external path" branch (used for absolute paths, e.g. from `search_history` results) does `re.search(pattern, line)` with no `try/except re.error`, unlike `FileOpsAgent.search_in_file` which explicitly falls back to literal matching on bad regex. An invalid pattern (e.g. `def (`) raises uncaught.

Correction to Gemini's impact claim: this doesn't crash "the entire agent" — `_call_tool` isn't wrapped in try/except, so the exception propagates out of `_generate_with_tools` up to the REPL's outer `except Exception` in `start.py`, which prints an error and continues. So the *process* survives, but the *whole turn* is silently aborted with no graceful tool-error response — still a real bug, just not fatal to the app.

**Fix:** mirror `file_ops_agent.py`'s guard:
```python
try:
    hit = re.search(pattern, line) if use_regex else pattern.lower() in line.lower()
except re.error:
    hit = pattern in line
```

**Fix Analysis:** The proposed fix is correct. One clarification: the external-path branch in `_handle_search_in_file` needs to mirror the full logic from `FileOpsAgent.search_in_file`, which checks `use_regex` first (and only calls `re.search` when `use_regex=True`). The fix above already captures that structure. The fallback to `pattern in line` (case-sensitive literal) on `re.error` matches `file_ops_agent.py`'s existing pattern exactly.

---

## 🔴 Category 2: Deadlocks, Freezes & Resource Blowouts

### 7. Thread Deadlock in PDF Ingestion — ✅ Confirmed (background thread, not app-fatal)
`_render_pages`'s `finally` block does a raw, untimed `render_q.put(None)`. Everywhere else in that function, puts go through a timeout-and-check-cancel loop — but the sentinel put in `finally` doesn't. If the consumer already exited and the queue (`maxsize=8`) is full, this blocks forever. Since `render_thread` is `daemon=True` and the caller only does `render_thread.join(timeout=5)`, the *process* doesn't hang — but the background thread leaks indefinitely for the rest of the session.

**Fix:** apply the same timeout/cancel-check pattern to the sentinel put:
```python
finally:
    doc.close()
    while True:
        try:
            render_q.put(None, timeout=1.0)
            break
        except _q.Full:
            if cancel_event.is_set():
                break
```

**Fix Analysis:** The proposed fix is correct in structure. One detail: `_q.Full` is the alias for `queue.Full` used in `image_ocr_agent.py` — confirm the same alias/import name is used in the `_render_pages` function's file scope. The fix correctly mirrors the exact pattern used by the rest of `_render_pages`'s puts. The `doc.close()` line is correctly placed before the sentinel loop since the queue being full doesn't affect the PDF handle, and the loop will eventually break when the consumer drains the queue.

### 8. Circuit Breaker Permanently Locks on Client Errors — ✅ Confirmed
`allow()` returns `False` for every caller once `_state == "half_open"`, and the only way out is `record_success()` or `record_failure()`. `_try_provider_models` explicitly *skips* `record_failure()` for client-side errors (400/413/422) — correctly, since those aren't the provider's fault. But if the one probe request that got through during half-open state hits a client-side error, **neither** method is ever called, so the breaker is stuck in `half_open` forever, silently disabling that OCR provider for the rest of the session.

**Fix:** add a third outcome that releases the half-open slot without touching failure counters:
```python
def record_inconclusive(self):
    with self._lock:
        if self._state == "half_open":
            self._state = "open"          # let the next call re-probe after cooldown
            self._half_open_inflight = False
```
Call `breaker.record_inconclusive()` (instead of doing nothing) when `_is_client_error` is True.

**Fix Analysis:** The proposed fix is correct and the right design choice. Looking at `_try_provider_models` (line 592–598 in `image_ocr_agent.py`): when `_is_client_error` is True, `record_failure()` is correctly skipped, but if the breaker was in `half_open` state, `_half_open_inflight` remains `True` forever (since only `record_success()` and `record_failure()` reset it), and `allow()` returns `False` for all subsequent calls (line 321: `return False` when state is `half_open`). The proposed `record_inconclusive()` with `_state = "open"` is correct — it resets the inflight flag and goes back to `open` so the cooldown timer can elapse and another probe can be attempted. One clarification needed: `record_inconclusive` should NOT increase `_cooldown` (unlike the failed-probe path in `record_failure`), since client-side errors aren't provider failures. The proposed fix correctly omits the `_cooldown` doubling.

### 9. Client-Side Image Errors Trigger Long Backoff Sleeps — ✅ Confirmed
`_call_with_rounds` can't distinguish "every provider failed transiently" from "this image will fail identically every time" (400/413) — it just sleeps `30s`, `60s` between rounds regardless. "90-second UI freeze" is a bit dramatic (it's one background OCR thread, and the pipeline waits on it because results are collected in page order), but the wasted wait for a permanently-bad image is real.

**Fix:** propagate whether *all* failures across a round were client-side (you already detect this per-provider in `_try_provider_models`/`_try_provider_models_group`); if so, have `_call_with_rounds` skip the sleep-and-retry and fail immediately instead of exhausting `MAX_ROUNDS`.

**Fix Analysis:** The proposed fix is directionally correct but the implementation detail needs refinement. Currently `_try_provider_models` returns `None` on failure regardless of whether the cause was client-side or transient. To propagate this distinction, the return type needs to change: instead of `None`, return a sentinel like `(None, "CLIENT_ERROR")` when all failures were client-side. Then `_call_with_fallback` propagates this, and `_call_with_rounds` checks the return to break the retry loop immediately. This is a bigger-than-it-looks refactor because `_call_with_rounds`/`_call_with_fallback`/`_try_provider_models` all currently share a clean `None=failure, tuple=success` contract. An alternative: have `_try_provider_models` set an instance variable (e.g. `self._last_failure_was_client_error`) that `_call_with_rounds` checks before deciding to sleep, avoiding the return-type change. The instance-variable approach is simpler and less risky given the existing code structure.

### 10. Unbounded Work Queue Defeats PDF Render Lookahead — ✅ Confirmed
`RENDER_LOOKAHEAD=8` caps `render_q`'s size, but the consumer drains it and hands items straight to `ThreadPoolExecutor.submit()`, whose internal work queue is unbounded and accepts submissions instantly regardless of worker availability. So the lookahead limit only throttles the producer relative to the *consumer's pull rate*, not relative to actual OCR completion — for a large PDF, all pages can get rendered to disk well before OCR catches up.

**Fix:** gate submission on in-flight future count, not just queue draining, e.g. a bounded semaphore sized to `PDF_OCR_MAX_WORKERS + small buffer`, acquired before `submit()` and released in a future callback.

**Fix Analysis:** The proposed fix is correct in approach. The semaphore pattern is the cleanest solution here. Concretely: create a `threading.Semaphore(PDF_OCR_MAX_WORKERS + 2)` before the consumer loop; acquire it before each `executor.submit()` call, and add a `.add_done_callback(lambda _: sem.release())` to each returned future. This limits in-flight executor tasks to `PDF_OCR_MAX_WORKERS + 2` regardless of how fast the consumer drains the queue. The `+ small buffer` (e.g. +2) prevents the semaphore from being the new bottleneck when workers finish slightly out of order. Note: the semaphore acquire happens in the consumer thread (which drains `render_q`), so it will block there — this is intentional and correct, since it's exactly what provides backpressure to the producer via the queue's `maxsize`.

### 11. Chunk Duplication on Large Paragraphs — ✅ Confirmed (real, traced through)
In `_chunk_text`, if a paragraph exceeds `EMBED_CHUNK_SIZE` alone, the overflow check (`... and window`) doesn't fire while `window` is still empty, so the giant paragraph gets added anyway. On the *next* paragraph, it flushes — and the overlap-seeding loop, walking backward from the newest item, only drops the giant paragraph from the retained window if the *most recent* item(s) alone accumulate ≥ `EMBED_CHUNK_OVERLAP` (200 chars) before the loop reaches it. If several small paragraphs follow (common in OCR'd bullet lists / equations), the loop can reach all the way back to the giant paragraph without hitting that threshold, and since it never explicitly excludes an oversized element, the giant paragraph stays in the window — permanently, since nothing ever trims from the front again as long as it's followed by short paragraphs. Every subsequent flush re-emits the (growing) window, i.e. genuine O(N²) chunk growth and re-embedding of the same giant text repeatedly.

**Fix:** never let the overlap-seeding logic re-include a paragraph that alone is ≥ `EMBED_CHUNK_SIZE`:
```python
for i in range(len(window) - 1, -1, -1):
    if len(window[i][0]) >= config.EMBED_CHUNK_SIZE:
        overlap_start_idx = i + 1   # exclude the oversized paragraph from carry-forward
        break
    overlap_len += len(window[i][0]) + 2
    overlap_start_idx = i
    if overlap_len >= config.EMBED_CHUNK_OVERLAP:
        break
```

**Fix Analysis:** The proposed fix resolves the duplication but has a subtle edge case: if `i + 1 == len(window)` (the oversized paragraph is the very last element of the window), then `window[i+1:]` is an empty list, meaning the next window starts empty — which is correct behavior. However there's another edge case: what if the oversized paragraph is at index 0 (i.e. the only element)? Then `overlap_start_idx = 1` and `window[1:]` is empty, which is again correct. The fix is safe. One additional improvement: the `_chunk_text` function should also handle the case where a single paragraph exceeds `EMBED_CHUNK_SIZE` and the window was empty when it was appended (line 85: `and window` prevents the overflow flush when window is empty, so the paragraph gets added anyway). That first-flush path doesn't cause duplication on its own, but the paragraph will appear in the next chunk's overlap. The proposed fix in the overlap loop correctly stops it from being permanently carried forward, so no separate change is needed for the initial append case.

### 12. `_scratch_size_mb` Permanently Disables Backups — ✅ Confirmed
`_scratch_size_mb()` walks the whole tree with no `EXCL_DIRS` pruning, while `_zip_scratch()` (the actual backup) prunes `node_modules`/`venv`/`.git`/etc. So a project with `node_modules` present (extremely common) gets its scratch size reported as huge even though the real archive would exclude all of it — permanently tripping the `BACKUP_MAX_SCRATCH_MB` (150MB) guard and disabling turn backups for good.

**Fix:** prune the same way `_zip_scratch` does:
```python
for root, dirs, files in os.walk(scratch_path):
    dirs[:] = [d for d in dirs if not any(d.startswith(p) for p in EXCL_PREFIXES)
               and not any(d.endswith(s) for s in EXCL_SUFFIXES) and d not in EXCL_DIRS]
    for fn in files:
        if any(fn.startswith(p) for p in EXCL_PREFIXES):
            continue
        ...
```

**Fix Analysis:** The proposed fix is correct. Looking at `_scratch_size_mb` in `turn_state_manager.py` (line 160–172), it does a bare `os.walk` with no pruning at all. The fix should mirror `_zip_scratch`'s exact exclusion logic, which uses `EXCL_PREFIXES`, `EXCL_SUFFIXES`, and `EXCL_DIRS` from `utils.py`. There's already a `_count_scratch_files` function (line 46–63) that does this correctly with `EXCL_DIRS` and `EXCL_SUFFIXES` pruning — `_scratch_size_mb` should adopt the same pattern for directories, and also skip files matching `EXCL_PREFIXES`. The fix as proposed is complete and correct.

### 13. Grep Search Floods Context with Dependencies — ✅ Confirmed
`FileOpsAgent.search_workspace`'s grep-mode `os.walk` doesn't prune `EXCL_DIRS` at all (unlike `workspace_tracker`'s reconcile, which does). A `.git` or `node_modules` folder in scratch gets fully crawled and grepped. The 50-result cap limits *output* size but not the wasted I/O or the risk of vendored/minified matches drowning out real ones.

**Fix:** `for root, dirs, files in os.walk(s_dir): dirs[:] = [d for d in dirs if d not in EXCL_DIRS]`.

**Fix Analysis:** The proposed fix is correct but incomplete — it only prunes on `EXCL_DIRS` (exact name match), while `workspace_tracker`'s reconcile also excludes directories matching `EXCL_PREFIXES` and `EXCL_SUFFIXES` (e.g. directories starting with `.` or ending with `_cache`). For full consistency, the fix should use the same three-part filter:
```python
dirs[:] = [d for d in dirs
           if d not in EXCL_DIRS
           and not any(d.startswith(p) for p in EXCL_PREFIXES)
           and not any(d.endswith(s) for s in EXCL_SUFFIXES)]
```
This is how `_count_scratch_files` and `_zip_scratch` already handle it in `turn_state_manager.py`.

### 14. Vision/OCR Pipelines Leak Preprocessed Images — ✅ Confirmed (one detail wrong)
`_copy_and_enhance`, `_build_master_enhanced`, and `_slice_master_into_tiles` all write files into `scratch/` for normal chat-turn image attachments, and unlike the PDF-ingestion path (which has an explicit `_cleanup()`), nothing ever deletes them. They accumulate indefinitely.

Correction: Gemini's claim that they "get indexed as project files" is **not accurate** — `workspace_tracker`'s tracked extensions (`CODE_EXTENSIONS`) and `code_search_agent`'s `CODE_INDEX_EXTENSIONS` don't include image extensions, so these leaked files are invisible to both the workspace tree and semantic code index. They're just untracked disk clutter, not corrupt search results.

**Fix:** After a turn using vision images completes, delete the `enhanced_*`, `master_*`, `overview_*`, `tile_*` files it created (exact-path tracking, same pattern `PDFIngestAgent._cleanup`/`_prepare_page_images` already use) — or at minimum sweep them in `/reset`.

**Fix Analysis:** The proposed fix is correct in approach but needs an implementation plan. The cleanest fix is to track the exact temp file paths created during a turn (already done in the PDF path via a `_temp_files` list passed to `_cleanup`) and call cleanup in a `finally` block after `process_images()` returns. For the non-PDF vision path (image attachments sent via the normal turn flow), the `ImageOCRAgent.process_images()` call happens in `renderer.process_images_via_ocr()` — a cleanup hook there or in `_run_turn`'s finally block would work. At minimum, add a glob sweep for `enhanced_*`, `master_*`, `overview_*`, `tile_*` patterns in `config.SCRATCH_DIR` to the `/reset` handler. The exact-path tracking approach is preferred since it avoids accidentally deleting intentionally named files that happen to start with those prefixes.

### 15. Unbounded RAM on Remote Image URLs — ✅ Confirmed
`_r.read()` in `_call_google_ocr`, `_call_google_ocr_multi`, and `_convert_messages_to_gemini` reads the entire HTTP response with no size cap, unlike the local-file path which explicitly checks a 20MB limit.

**Fix:** check `Content-Length` before reading, or read in a capped loop:
```python
raw = _r.read(20 * 1024 * 1024 + 1)
if len(raw) > 20 * 1024 * 1024:
    raise ValueError("Remote image exceeds 20MB limit")
```

**Fix Analysis:** The proposed fix is correct. The `+1` trick (read one byte past the limit, then check length) is a standard pattern. However it still buffers the full 20MB+1 bytes in RAM before checking — for true memory safety, a chunked read approach is better:
```python
chunks = []
total = 0
for chunk in iter(lambda: _r.read(65536), b''):
    total += len(chunk)
    if total > 20 * 1024 * 1024:
        raise ValueError("Remote image exceeds 20MB limit")
    chunks.append(chunk)
raw = b''.join(chunks)
```
The `Content-Length` pre-check is a useful early-exit but can't be relied upon alone (servers can omit or lie about it). The chunked approach is safer. For the simpler context here (OCR pipeline), the `+1` read is acceptable since 20MB is the known limit throughout the codebase.

### 16. Default `urllib` User-Agent Triggers 403s — ✅ Confirmed
`urllib.request.urlopen(image_url, timeout=15)` sends `Python-urllib/x.y`, which Cloudflare/Imgur/Wikipedia and most CDNs block by default.

**Fix:**
```python
req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=15) as _r: ...
```

**Fix Analysis:** The proposed fix is correct and sufficient. The `Mozilla/5.0` user-agent is the standard workaround for CDN bot-detection. No other changes required — this is a direct drop-in at each `urllib.request.urlopen(image_url, ...)` call site. There are three such call sites mentioned (in `_call_google_ocr`, `_call_google_ocr_multi`, and `_convert_messages_to_gemini`); all three need the same fix.

---

## 🟠 Category 3: Data Loss, State Corruption & Silent Failures

### 17 & 18. `/rerun` Ledger Corruption + WAL Silently Drops the Target Turn — ✅ Confirmed (same root cause)
`cmd_rerun` calls `tsm.delete_turn_archives(target)` — permanently deleting turn N's ledger entry and archive — **before** attempting regeneration. `tail_history`/`tail_ledger` only ever capture turns *after* `target`. If the rerun is cancelled or the process crashes mid-flight, `_reappend_tail` restitches the tail at `target - 1`: what was `target+1` now becomes the de-facto new turn `target`, but its ledger entry still carries its *old* turn number and filenames — a numbering/ledger desync — and turn N's own pre-rerun content and archive are simply gone, whether the app is cancelled gracefully or killed outright (WAL recovery has no way to bring it back since it was never captured).

**Fix:** don't delete turn N's archive/ledger entry until *after* `_run_turn` succeeds. Move `tsm.delete_turn_archives(target)` to the success branch (right before/with the final `_reappend_tail(tail_history, tail_ledger, target, current)` call). For full crash-safety, also persist turn N's own pre-rerun messages/ledger entry into the WAL alongside the tail, so a crash mid-regeneration can restore the *original* turn N instead of leaving a gap.

**Fix Analysis:** The proposed fix is correct. Looking at `cmd_rerun` in `start.py` (lines 941–991): `tsm.delete_turn_archives(target)` is called at line 954, before `_run_turn` at line 975. The fix is to move line 954 to after the `if result is None:` block, i.e., into the success path just before line 991's `_reappend_tail`. The WAL (`_save_rerun_tail`) at line 946 already saves the tail turns (T(target+1)..T(current)), but it doesn't save T(target)'s own content — adding the pre-rerun T(target) messages and ledger entry to the WAL is the right call for crash-safety. The cancellation path (lines 983–988) correctly re-appends the tail at `target-1` and returns without committing, so moving the delete to the success path is safe.

### 19. `/edit` Erases Image-Attachment Marker — ✅ Confirmed, impact overstated
`cmd_edit` strips the `[SYSTEM: Images attached — ...]` marker under a comment saying "`_run_turn` will regenerate it" — but `/edit` never calls `_run_turn`. The marker is gone from that message's stored text for good.

Correction: this does **not** erase the actual images or break `/rerun` — `/rerun` determines `had_images` from the ledger's `images_snap` field, not from this text marker (the marker is only used there as a display label, falling back to `"unknown"` if absent). So the real loss is one line of *display* context for `/history`/future-turn recall, not the underlying image data.

**Fix:** rebuild the marker from the actual loaded snapshot instead of just deleting it:
```python
if edit_images:
    names = ", ".join(img["filename"] for img in edit_images)
    new_content = f"{new_content}\n[SYSTEM: Images attached — {names}]"
```

**Fix Analysis:** The proposed fix is correct. Looking at `cmd_edit` in `start.py` (lines 1155–1162): the current code strips the marker via `re.sub` at line 1158 and never re-adds it. The fix should be applied after line 1158 — if `edit_images` is not None (i.e. the turn had images), re-append the marker using filenames from the snapshot. Note the em-dash character: the existing marker uses `—` (U+2014), and the `re.findall` at line 1145 already uses `—` to match it, so the rebuilt marker should also use `—`. The fix as shown uses `—` (standard hyphen) which would make future regex matching inconsistent — use `f"{new_content}\n[SYSTEM: Images attached — {names}]"` with U+2014.

### 20. Incomplete Batch Embedding Locks a Broken Index — ✅ Confirmed
`chunk_and_embed` drops any chunk whose embedding came back `None` but still writes `source_hash` for the *full* file. A later re-ingest attempt is blocked by the MD5 skip-guard ("already ingested, unchanged") even though chunks are silently missing. Notably, `search_history_agent.py`'s own code comments describe this *exact* failure mode as "Bug 16" and explicitly avoid it (never persists the sidecar if any embedding is `None`) — `doc_search_agent.py` just never got the same fix.

**Fix:** mirror that pattern — don't write `source_hash`/report success if any embedding is `None`:
```python
if any(e is None for e in embeddings):
    return {"success": False, "error": "Some chunks failed to embed — retry ingest_pdf."}
```

**Fix Analysis:** The proposed fix is correct. However, it should be placed precisely — in `doc_search_agent.py`'s `chunk_and_embed` (or `ingest` method), the `source_hash` write must be skipped only when the embedding API returned `None` for any chunk, not just when the API call raised an exception (which is already caught separately). Additionally, the fix should distinguish between partial failures (some chunks failed) and total failures (all chunks failed) — partial failures could still save a degraded index rather than failing completely. The `search_history_agent.py` pattern (referenced in the bug) treats any `None` as a full abort, which is the safer choice here too.

### 21. Total OCR Failure Stored as Successful Ingestion — ✅ Confirmed
When every OCR round is exhausted, `_call_with_rounds`/`_call_with_rounds_group` return the literal string `"[OCR ERROR — all tiers exhausted across all retry rounds]"` as the markdown — which is **non-empty**, so `if not full_text.strip():` never catches an all-pages-failed scenario; the error text gets written to the `.txt`, chunked, embedded, and reported as `success: True`.

**Fix:** check for the error marker explicitly per page before concatenating, and fail (or at least warn per-page) instead of silently treating error text as content:
```python
if markdown and not markdown.startswith("[OCR ERROR"):
    parts.append(...)
...
if not parts:  # every page's markdown was empty or an error marker
    return {"success": False, "error": "OCR failed on every page."}
```

**Fix Analysis:** The proposed fix is correct. Additionally, partial-failure cases (some pages succeeded, some returned OCR error markers) should also be surfaced — currently, if even one page succeeds, the error marker text for failed pages gets embedded as actual content (chunks containing `[OCR ERROR — all tiers exhausted...]` strings). The improved fix should track error pages separately and warn the user about partial OCR failure even on partial success:
```python
error_pages = []
for page_num, markdown in enumerate(page_markdowns, 1):
    if markdown and not markdown.startswith("[OCR ERROR"):
        parts.append(markdown)
    else:
        error_pages.append(page_num)
```
Then report `error_pages` in the success response so the user knows which pages failed.

### 22. `ingest_chat` Silently Overwrites Prior Imports — ✅ Confirmed
Output filename is `f"{stem}_imported.jsonl"` — derived purely from basename. Two different uploads both named `chat.txt` (different sessions, different content) collide on the same output file; the hash-match skip guard only prevents overwrite when content is *identical* — a differing hash falls through to an unconditional overwrite with no warning, contradicting the stated "permanent, cross-session" design goal.

**Fix:** include the content hash (or a timestamp) in the output filename: `f"{stem}_{file_hash[:8]}_imported.jsonl"`.

**Fix Analysis:** The proposed fix is correct. The content-hash approach is better than a timestamp because it's deterministic — importing the same file twice still produces the same output filename, preserving idempotency. The existing MD5 skip guard (sidecar `source_hash` check) continues to work since the sidecar path is derived from `out_path`. No other changes needed; the `file_hash` variable is already computed at line 204 of `tool_handlers.py` before `out_name` is constructed.

### 23. `/reset` Ghost Archive Race — ✅ Confirmed, low real-world impact
`clear_all()` wipes `backups/` without calling `_await_all_zips()` first (unlike `restore_turn`, which does). A background `_do_zip` thread from a turn committed just before `/reset` can finish and drop a fresh zip into the now-"clean" directory seconds later. Harmless clutter in practice (no ledger entry points to it, and it'll get pruned eventually), but still a real, easily-fixed race.

**Fix:** `self._await_all_zips()` at the top of `clear_all()`.

**Fix Analysis:** The proposed fix is correct and simple. Looking at `clear_all()` in `turn_state_manager.py` (line 860), it immediately iterates `os.listdir(config.BACKUPS_DIR)` and removes files. Adding `self._await_all_zips()` (which already exists in the class and joins all pending background zip threads with a timeout) as the very first line of `clear_all()` is sufficient. The method exists, is already used by `restore_turn()`, and the one-liner addition fully resolves the race.

### 24. `/restore` Drops Empty Directories — ✅ Confirmed, minor
`_zip_scratch` only ever calls `zf.write(fp, arcname)` for files; an empty directory has no file to trigger an entry, so it's absent from the archive and doesn't get recreated on restore (even though `folder_registry` in the separately-restored registry snapshot still lists it as tracked, causing a minor display/reality mismatch).

**Fix:** explicitly add empty-directory entries, e.g. after the walk, for any directory with no files and no subdirectories, `zf.writestr(rel_dir + '/', '')`.

**Fix Analysis:** The proposed fix is correct. However, `zf.writestr(rel_dir + '/', '')` writes an empty file with a trailing-slash name, which not all unzip implementations interpret as a directory entry. The more portable approach is `zf.mkdir(rel_dir)` (Python 3.11+) or creating a `ZipInfo` with `is_dir()` semantics:
```python
zi = zipfile.ZipInfo(rel_dir + '/')
zf.writestr(zi, '')
```
This is the standard zip directory entry format and is recognized by all extractors. The `_zip_scratch` function already has the `os.walk` loop — add a check after processing each directory: if it has no files and no non-excluded subdirs, write the directory entry.

### 25. `/delete` Corrupts `rerun_of` Pointers — ✅ Confirmed, cosmetic only
`delete_turn`'s ledger rebuild decrements `turn` for entries after the deleted one, but never touches `rerun_of`. If turn 5 was `rerun_of: 3` and turn 2 is deleted, turn 5 becomes turn 4 but still says "rerun of T3" — which, after the shift, is the wrong turn (what's now T3 used to be T4). Only affects the `[↺ rerun of TN]` display in `/turns`, not functionality.

**Fix:** when rebuilding, also adjust `rerun_of`: if it equals the deleted turn, clear it; if it's greater, decrement it.

**Fix Analysis:** The proposed fix is correct. In `delete_turn()` of `turn_state_manager.py`, after the ledger is rebuilt with decremented turn numbers, each entry's `rerun_of` field needs the same offset adjustment. The fix is straightforward: in the same loop that decrements `entry["turn"]`, add:
```python
if "rerun_of" in entry:
    ro = entry["rerun_of"]
    if ro == turn_num:
        del entry["rerun_of"]  # or set to None
    elif ro > turn_num:
        entry["rerun_of"] = ro - 1
```
This is purely cosmetic (only affects the `/turns` display), so no urgency, but it's a clean one-step fix.

### 26. Phantom Deleted Folders Persist — ✅ Confirmed
`reconcile_workspace()` only diffs `file_registry` against disk; `folder_registry` is never reconciled at all, so a directory removed via `bash rm -rf` stays listed in `[WORKSPACE STRUCTURE]` forever.

**Fix:** collect on-disk directories in the same walk (respecting `EXCL_DIRS`) and remove any `folder_registry` entries not present:
```python
on_disk_dirs = { ... }  # gathered from os.walk's `dirs`
stale = self.folder_registry - on_disk_dirs
if stale:
    self.folder_registry -= stale
    self._registry_dirty = True
```

**Fix Analysis:** The proposed fix is correct. The implementation detail: `folder_registry` stores paths as relative strings (relative to `workspace_root`), so `on_disk_dirs` must be built the same way — using `os.path.relpath(os.path.join(root, d), self.workspace_path)` for each directory entry found in the walk. The same `EXCL_DIRS` / `EXCL_PREFIXES` / `EXCL_SUFFIXES` filters applied to the file walk should be applied to the directory collection too, so excluded directories (like `node_modules`) don't get re-added to `on_disk_dirs` and cause false-negative "stale" removals.

### 27. Reasoning Erasure in Multi-Block Thinking — ✅ Confirmed
`extract_thinking_tags` uses `tag_pattern.search()` (first match only) to *capture* thinking, but `tag_pattern.sub("", text)` (all matches) to *strip* it from the visible text. If a model ever emits two separate `<think>...</think>` blocks, the second block's content is deleted from the response and never captured anywhere.

**Fix:** capture all matches:
```python
matches = tag_pattern.findall(text)
if matches:
    thinking = "\n\n---\n\n".join(m[1].strip() for m in matches)
    return thinking, tag_pattern.sub("", text).strip()
return None, text
```

**Fix Analysis:** The proposed fix is correct. One note: `tag_pattern.findall(text)` returns a list of tuples (the capture groups), so `m[1]` is the thinking content captured by the second group — confirm that the regex has exactly two groups `(tag_name, content)` before relying on `m[1]`. The separator `"\n\n---\n\n"` between multiple thinking blocks is a reasonable choice for display in the terminal. The `tag_pattern.sub("", text)` correctly removes all occurrences (already the behavior of `re.sub`), so no change to the stripping side is needed.

---

## 🟡 Category 4: RAG, Search & Cache Failures

### 28. History Search Embeds Only the Scratchpad — ✅ Confirmed
Assistant chat-history content is stored as `<think>...</think>\n\n{answer}` when thinking exists (per `emperor_agent._generate_with_tools`), and that full string flows unmodified into the global `.jsonl`. `_reindex_sessions` truncates assistant text to 1000 chars *before* stripping thinking — for any model with reasoning traces longer than 1000 characters (common), the actual answer never even reaches the embedded passage.

**Fix:** strip thinking (reuse `utils.extract_thinking_tags`) before truncating for the embedding passage:
```python
from utils import extract_thinking_tags
_, a_clean = extract_thinking_tags(a)
passages = [f"USER: {u[:1000]}\nASSISTANT: {a_clean[:1000]}" for u, a in new_turns]
```

**Fix Analysis:** The proposed fix is correct. The `extract_thinking_tags` function is in `utils.py` and is already used elsewhere in the codebase. The list comprehension needs to call it per-turn:
```python
from utils import extract_thinking_tags
passages = [
    f"USER: {u[:1000]}\nASSISTANT: {extract_thinking_tags(a)[1][:1000]}"
    for u, a in new_turns
]
```
Note that `extract_thinking_tags` returns `(thinking, clean_text)` — using `[1]` gives the clean text. This is a one-line change in `_reindex_sessions` at line 212–214 of `search_history_agent.py`.

### 29. `mv` Destroys Semantic Doc Search — ✅ Confirmed
The bash `mv` post-hook calls `remove_file(mv_src)`, which — correctly for a real deletion — deletes the matching `.chunks.json`. But `mv` isn't a deletion; the file (and its content) still exists under the new name, just without its chunk index, and nothing re-embeds it. `doc_search` permanently loses that document until someone manually re-runs `ingest_text`.

**Fix:** in the mv post-hook, if the source has a `.chunks.json` sidecar, rename it alongside the file (updating its internal `"source"` field) instead of deleting it, rather than routing through `remove_file`.

**Fix Analysis:** The proposed fix is directionally correct but the implementation is non-trivial. A `.chunks.json` sidecar stores a list of chunk objects that each contain a `"source"` field (the original file path). On `mv`, all three things need to happen atomically: (1) rename the `.chunks.json` file to match the new filename, (2) update every chunk's `"source"` field to the new path, (3) update the sidecar's top-level `source_hash` key if present. Given the file could have many chunks, a read-modify-write is needed. An alternative simpler fix: in the mv post-hook, after renaming, delete the old `.chunks.json` (current behavior) AND immediately re-embed the file at its new path if possible, or at minimum set a flag that tells the next `doc_search` call to re-ingest it. The cleanest production fix is the rename + field-update approach, but it requires careful implementation.

### 30. Direct Edits Desync Semantic Search — ✅ Confirmed
`_handle_str_replace` only calls `code_search_agent.mark_stale()` on success — there's no equivalent invalidation/re-embed trigger for `doc_search_agent`'s `.chunks.json`. Editing an ingested `.txt` via `str_replace` leaves `doc_search` returning the old, pre-edit passages forever.

**Fix:** after a successful `str_replace` on a `.txt` with a matching `.chunks.json`, either re-run `chunk_and_embed` on it or delete the stale `.chunks.json` so it's at least flagged as needing re-ingestion rather than silently serving stale text.

**Fix Analysis:** The proposed fix is correct, and deleting the stale `.chunks.json` is the right default (re-running `chunk_and_embed` inline would add latency to every `str_replace` call on a `.txt`). The implementation: in `_handle_str_replace`, after `code_search_agent.mark_stale()` is called, check if a `.chunks.json` sidecar exists for the modified file and delete it. The user can then re-run `ingest_text` to rebuild it. This matches the existing pattern where `code_search_agent.mark_stale()` defers the re-index rather than doing it immediately. Add a note to the `str_replace` result message that the doc search index was invalidated, so the model knows to re-ingest.

### 31. Web Extraction Drops Article Intros — ✅ Confirmed
`extract_sections` splits on markdown headers; text appearing *before* the first header lands in `parts[0]` while `current_header` is still `None`, and the `elif part.strip() and current_header:` guard silently discards it.

**Fix:** treat pre-header text as its own section:
```python
elif part.strip():
    header = current_header or "Introduction"
    sections.append({'type': 'section', 'header': header, 'content': part.strip(),
                      'priority': self._calc_priority(header, part)})
```

**Fix Analysis:** The proposed fix is correct. The `"Introduction"` fallback header is a reasonable label for pre-header content. One consideration: `_calc_priority("Introduction", part)` will use the same priority logic as a real header named "Introduction" — if `_calc_priority` weighs header names (e.g. boosting sections whose header matches the query), a synthetic `"Introduction"` label could affect reranking. An alternative: use `current_header or ""` and let the priority calculation handle empty-header sections naturally. This depends on `_calc_priority`'s internals, but the `"Introduction"` default is safe for most cases.

### 32. Empty Search Results Cached for 30 Days — ✅ Confirmed
A zero-result search produces `final = ""`, which gets written to `self.cache[cache_key]` with the standard 30-day TTL — `_is_cache_valid` doesn't check for emptiness, so an identical repeated query is stuck returning nothing for a month.

**Fix:** skip caching when `final` is empty:
```python
if final:
    self.cache[cache_key] = {"timestamp": time.time(), "content": final}
    self._mark_cache_dirty()
return final
```

**Fix Analysis:** The proposed fix is correct and sufficient. No additional changes needed — this is a clean guard that prevents the 30-day TTL from locking in empty results.

### 33. BM25 Hard-Excludes `.txt` — ✅ Confirmed, impact narrower than stated
`_ensure_index_loaded` unconditionally skips all `.txt` files (assuming they're OCR'd PDF output covered by `doc_search`), while `.md` — treated identically by `ingest_text` — is not excluded.

Correction: Gemini's "permanently invisible to `workspace_search`" isn't quite right — this only affects the `semantic=true` BM25-*fallback* path (when Codestral Embed is unavailable). The default grep mode (`semantic=false`, what most calls use) reads files directly off disk and is unaffected.

**Fix:** only exclude a `.txt` if it actually has a corresponding `.chunks.json` (i.e. it was genuinely ingested), not all `.txt` files:
```python
if rel_path.lower().endswith('.txt'):
    sidecar = os.path.join(self.workspace_path, rel_path[:-4] + ".chunks.json")
    if os.path.exists(sidecar):
        continue
```

**Fix Analysis:** The proposed fix is correct. The same sidecar check should be applied consistently in both the BM25 index loader (`_ensure_index_loaded`) and in `_should_index` (Bug #35). Since both bugs share the same root cause and sidecar pattern, the fix can be factored into a helper `_has_doc_sidecar(filepath)` to avoid duplicating the path-construction logic across the two call sites.

### 34. `ingest_text` Causes Duplicate Embedding — ✅ Confirmed
`ingest_text` copies `/uploads/file.txt` → `/workspace/scratch/file.txt`. `CodeSearchAgent._iter_source_files` dedups by *container path* (`/uploads/file.txt` vs `/workspace/scratch/file.txt` — different strings), so both copies get independently chunked and embedded via paid Codestral Embed calls, and both show up as duplicate hits.

**Fix:** dedup by content hash in addition to (or instead of) container path, or explicitly skip an uploads-folder file when an identical-hash copy already exists in scratch.

**Fix Analysis:** The hash-based dedup is the right approach. The implementation: in `CodeSearchAgent._iter_source_files()`, compute an MD5 of each file's content as it's encountered, and skip any file whose hash has already been seen in the current iteration. This handles not just the uploads→scratch copy case but any other accidental duplicates. The MD5 computation adds overhead for large files but is bounded by the number of source files indexed (which is already bounded by `CODE_INDEX_EXTENSIONS`). An alternative cheaper dedup: since `ingest_text` always copies to the exact same scratch path, explicitly skip a file under `UPLOADS_FOLDER` if an identically-named file exists under `SCRATCH_DIR` (a path-based shortcut that covers the common case without hashing).

### 35. Codestral Burns Tokens on Ingested PDFs — ✅ Confirmed
`.txt` is in `CODE_INDEX_EXTENSIONS`, and `_should_index` has no check for a paired `.chunks.json`, so OCR'd PDF transcripts (already chunked/embedded for `doc_search` via Nemotron) get redundantly chunked and sent to Mistral's paid API too.

**Fix:** same sidecar check as #33's fix, applied in `_should_index`:
```python
if ext == ".txt":
    if os.path.exists(os.path.splitext(filepath)[0] + ".chunks.json"):
        return False
```

**Fix Analysis:** The proposed fix is correct. As noted in Bug #33's analysis, factoring this into a shared `_has_doc_sidecar(filepath)` helper used by both this fix and Bug #33's fix will prevent future drift. The fix is otherwise straightforward.

### 36. Uploads Cache Blind to Subdirectory Changes — ✅ Confirmed
`os.path.getmtime(config.UPLOADS_FOLDER)` only reflects changes to files/folders *directly inside* the top-level folder — a file added/edited in `/uploads/subfolder/` doesn't touch the parent's mtime, so the cache never invalidates and the model keeps seeing a stale listing.

**Fix:** compute a cheap aggregate signature over the whole tree (max mtime + file count) instead of a single top-level `getmtime`, or accept the walk cost since it's already bounded by what `_scan_uploads_folder` would do anyway on a cache miss.

**Fix Analysis:** The proposed fix is correct. The aggregate signature approach (max mtime across all files in the uploads tree + file count) is the right balance — it catches subdirectory changes without a full re-scan every turn. Implementation:
```python
def _uploads_signature(folder):
    max_mtime = 0
    count = 0
    for root, _, files in os.walk(folder):
        for fn in files:
            try:
                mt = os.path.getmtime(os.path.join(root, fn))
                if mt > max_mtime:
                    max_mtime = mt
                count += 1
            except OSError:
                pass
    return (max_mtime, count)
```
The cache invalidation check then becomes `current_sig != cached_sig`. This is a bit more expensive than a single `getmtime` but still O(number of upload files) and bounded.

### 37. Uploads Cache Stale After Rollback — ✅ Confirmed
Neither the Ctrl+C rollback path in `start.py` nor `TurnStateManager.restore_turn()` ever resets `agent._uploads_cache`. Since a rollback only touches `scratch/` (not `uploads/`), the cache key (`uploads_mtime`) doesn't change, so a stale "ALREADY INGESTED" note for a now-reverted PDF persists, and the model's `doc_search` calls against it fail.

**Fix:** set `agent._uploads_cache = None` in both rollback code paths, alongside the existing registry/BM25 invalidation calls.

**Fix Analysis:** The proposed fix is correct and minimal. The two rollback paths are: (1) the Ctrl+C interrupt rollback in `start.py`, and (2) `TurnStateManager.restore_turn()`. In both cases, `agent._uploads_cache = None` should be added adjacent to whatever other cache/registry invalidation calls are already present. This is a one-liner in each place.

### 38. Tree-Sitter Drops Short Helper Functions — ✅ Confirmed
The regex-fallback path (`if not final_chunks and used_treesitter:`) only triggers when tree-sitter produced **zero** usable chunks. If a file has one large class (survives the `CODE_CHUNK_MIN_LINES` filter) plus several small standalone functions (each individually filtered out for being under 3 lines), `final_chunks` is non-empty (contains the class), so the fallback never runs — the small functions are lost entirely, neither indexed alone nor merged into anything.

**Fix:** don't apply the min-line filter to tree-sitter-derived function/class/method chunks at all (they're complete semantic units regardless of size, unlike arbitrary regex windows which need a size floor to be meaningful) — or separately regex-chunk whatever line ranges aren't covered by a surviving AST chunk.

**Fix Analysis:** Removing the min-line filter entirely for tree-sitter chunks is the cleanest fix. The `CODE_CHUNK_MIN_LINES` filter exists to prevent noise from arbitrarily-windowed regex chunks (e.g. a 2-line window of `import` statements), not from AST-level units. A helper like `def foo(): pass` is a valid semantic unit even at 2 lines. Implementation: track whether each chunk was tree-sitter derived (already possible since tree-sitter chunks are built in a separate pass) and skip the min-line filter for those. For the regex fallback path, keep the min-line filter as-is.

### 39. C/C++ Headers Downgraded to Regex Chunking — ✅ Confirmed, and slightly worse than described
`_LANG_MAP` has no `.h`/`.hpp` entries, so `detect_language()` returns `'text'` for headers, which isn't in `CHUNK_NODE_TYPES` — tree-sitter is skipped, falling back to dumb line-window chunking. Matches Gemini's claim for `.h`.

One thing Gemini missed: `.hpp` isn't even in `config.CODE_INDEX_EXTENSIONS` (only `.h` is) — so `.hpp` files aren't just badly-chunked, they're **excluded from semantic code search entirely**.

**Fix:** add `'.h': 'c', '.hpp': 'cpp'` to `_LANG_MAP` (this also fixes `CODE_EXTENSIONS`, `UPLOAD_EXTENSIONS`, and workspace_tracker's reconcile set automatically, since they all derive from it), and add `.hpp` to `config.CODE_INDEX_EXTENSIONS`.

**Fix Analysis:** The proposed fix is correct. Two things to verify before applying: (1) confirm that tree-sitter's Python bindings include a C/C++ grammar that correctly handles `.h`/`.hpp` — if not, they'll still fall through to regex chunking at runtime (but at least they'll be correctly classified and indexed). (2) Adding `.hpp` to `CODE_INDEX_EXTENSIONS` while it's still not in `_LANG_MAP` would cause it to be picked up for semantic indexing but chunked via regex fallback — the `_LANG_MAP` addition must come first or simultaneously.

### 40. Code Search Crawls `target/`, `build/`, `dist/` — ✅ Confirmed
`config.CODE_INDEX_EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}` is missing `target`, `build`, `dist`, and `env` — all present in `utils.EXCL_DIRS`, which every *other* exclusion list in the project is built from or matches. Compiled artifacts and bundles get sent to paid Codestral Embed.

**Fix:** either import `utils.EXCL_DIRS` directly for this constant, or add the missing entries.

**Fix Analysis:** Importing `utils.EXCL_DIRS` directly is the right long-term fix — it's the single source of truth already used by `workspace_tracker`, `turn_state_manager`, and `start.py`. Having `code_search_agent.py` define its own partial exclusion set is the exact kind of drift that caused this bug. The change is: replace `CODE_INDEX_EXCLUDE_DIRS = {".git", "__pycache__", ...}` in `config.py` with `from utils import EXCL_DIRS as CODE_INDEX_EXCLUDE_DIRS`, or just reference `utils.EXCL_DIRS` directly inside `code_search_agent.py`'s walk.

---

## 🔵 Category 5: Parsing, Rendering & System Inconsistencies

### 42. Positional Tag Matching Corrupts NIM History — ✅ Confirmed, but doesn't affect execution
`_TAG_RE` in `_convert_to_native_tools` matches *any* XML-like tag in the assistant's prose (not filtered to known tool names, unlike `_parse_pseudo_tools`), then aligns matches to `result_blocks` **purely by list position**. If the model writes something like `I'll edit the <script> tag...` in prose before its actual `<bash>` call, the unrelated `<script>` tag occupies position 0, and the real tool call's result gets paired with `<script>`'s (irrelevant) attributes when reconstructing NIM's `tool_calls[]`.

Worth noting: this only corrupts the *NIM-specific historical reconstruction* sent back to the model on later turns — it runs on messages *after* tools already executed correctly (via the properly-filtered `_parse_pseudo_tools`), so it doesn't cause wrong tool execution, just a subtly wrong self-history for the model to read later.

**Fix:** restrict `_TAG_RE`'s scan to only tags matching known tool names (build the alternation from `GROUP_TOOLS_UNION`, mirroring how `_parse_pseudo_tools` already scopes its search).

**Fix Analysis:** The proposed fix is correct. The implementation: inside `_convert_to_native_tools`, before the `_TAG_RE` definition, build a tool-name alternation the same way `_parse_pseudo_tools` does:
```python
_TOOL_ALT = "|".join(re.escape(name) for name in GROUP_TOOLS_UNION)
_TAG_RE = re.compile(
    rf'<({_TOOL_ALT})([^>]*?)(?:/>|>(.*?)</\1>)',
    re.DOTALL,
)
```
This is a local (inside-function) re-compile per call, but since `_convert_to_native_tools` is called per message pair (not per token), the overhead is negligible. `GROUP_TOOLS_UNION` is already imported/available in `llm_backends.py`.

### 43. `\in` Replacing Before `\inf` → `∈f` — ✅ Confirmed
In the `_bare` dict, `\in` is defined far earlier than `\inf`, and `text.replace()` is a plain substring op — `\in` is a literal prefix of `\inf`, so `\inf` gets partially replaced before its own entry is ever reached. Notably, `\infty` (also prefixed by `\in`) is placed *before* `\in` in the same dict specifically to avoid this — `\inf` was just missed.

**Fix:** move `r'\inf': 'inf'` earlier in the dict, before `r'\in': '\u2208'` (same treatment already given to `\infty`).

**Fix Analysis:** The proposed fix is correct and is the minimal surgical change. To be safe, also audit for other LaTeX prefix collisions in `_bare` — for example `\int` is a prefix of `\intop`, `\iota` of `\iota` (no issue), etc. A more robust long-term fix: sort the keys by descending length before iterating, so longer (more-specific) patterns always replace before shorter prefix patterns. Python dicts preserve insertion order, but a simple sort-by-length of the items list before the replace loop would make ordering errors impossible.

### 44. Greedy Unbraced Superscript Regex — ✅ Confirmed
`\^([0-9+\-=a-zA-Z]+)` is greedy and includes `+`/`-`/`=` in the character class with no boundary logic, so `x^2+y` matches `^2+y` as one exponent, rendering as `x²⁺ʸ` instead of `x² + y` — actively changes the apparent meaning of common physics/math text.

**Fix:** restrict the unbraced form to a much narrower, unambiguous case (a run of digits with an optional leading sign), and require braces for anything more complex:
```python
r'\^\{([0-9+\-=()a-zA-Z]+)\}|\^([+-]?\d+)'
```

**Fix Analysis:** The proposed fix is correct. The updated regex correctly handles:
- Braced: `x^{2+y}` → `x²⁺ʸ` (full expression, braces required)
- Unbraced: `x^2` → `x²` (digits only, safe)
- Does NOT match: `x^2+y` as a single exponent (the `+y` is left as plain text)

One edge case: unbraced negative exponents like `x^-2` — the `[+-]?\d+` form captures this correctly as `-2`. Also confirm that the replacement logic handles the two capture groups (braced group 1, unbraced group 2) correctly — whichever matched will be non-empty, and the other will be an empty string (from `re.findall`'s behavior with alternation groups).

### 45. `url_search` Block-Form Corruption — ✅ Confirmed (defensive gap, not hit by the documented format)
The documented/prompted format for `url_search` is self-closing (`<url_search url="..." context="..."/>`), which parses fine via `_parse_xml_attrs` regardless of `SIMPLE_PARAM`/`STRUCTURED` classification. But `url_search` is filed under `SIMPLE_PARAM`, so *if* a model instead emits it as a block tag with named children (`<url_search><url>...</url><context>...</context></url_search>` — plausible, since other tools like `str_replace` use exactly that shape), `_parse_inner_args` dumps the entire raw inner XML string into `url`, and `context` is lost.

**Fix:** move `url_search` to `STRUCTURED`: `"url_search": ["url", "context"]`.

**Fix Analysis:** The proposed fix is correct. This change only affects the block-tag parsing fallback path — the primary self-closing format `<url_search url="..." context="..."/>` is handled by `_parse_xml_attrs` regardless of `SIMPLE_PARAM`/`STRUCTURED` classification, so existing correct model outputs are unaffected. Moving to `STRUCTURED` only improves behavior when a model emits the block form. No other changes needed.

### 48. Missing Extensions (SVG, JSX/TSX, C#) — ✅ Confirmed
- `UPLOAD_EXTENSIONS` omits `.svg` even though `IMAGE_PREVIEW_EXTENSIONS` supports it — an uploaded SVG never even appears in the `[UPLOADS FOLDER]` listing the model sees.
- `config.CODE_INDEX_EXTENSIONS` omits `.jsx`/`.tsx`/`.cs`, even though `_LANG_MAP` (and therefore syntax highlighting / `CODE_EXTENSIONS` / tracking) already recognizes them — they're visible and trackable, just invisible to `code_search_agent`'s semantic index. This is drift between two extension lists that should stay in sync.

**Fix:** add `.svg` to `UPLOAD_EXTENSIONS`; add `.jsx`, `.tsx`, `.cs` to `config.CODE_INDEX_EXTENSIONS` (or better, derive `CODE_INDEX_EXTENSIONS` from `CODE_EXTENSIONS` the same way `UPLOAD_EXTENSIONS` already does, to prevent this drift recurring).

**Fix Analysis:** The proposed fix is correct. Deriving `CODE_INDEX_EXTENSIONS` from `CODE_EXTENSIONS` is the right structural fix (as also implied by Bug #40's fix). However, not everything in `CODE_EXTENSIONS` should be semantically indexed — e.g., `.csv`, `.env`, `.txt` (without a sidecar) are in `CODE_EXTENSIONS` but aren't useful for semantic code search. So the derivation should use a filter: start from `CODE_EXTENSIONS`, then exclude non-code types. The simplest correct fix: add the missing extensions explicitly now AND add a comment noting they must stay in sync with `CODE_EXTENSIONS`/`_LANG_MAP`, until a proper derivation is implemented.

### 49. Windows Case-Sensitive Path Matching — ✅ Confirmed
`host_to_container_path` compares with `os.path.normpath` only, not `os.path.normcase` — on Windows, a drive-letter or directory casing mismatch (`c:\...` vs `C:\...`) fails the prefix check and falls through to returning the raw host path unchanged. Other parts of the codebase (`tool_handlers._norm`) already guard against exactly this with `normcase`, so this is a real, known-pattern inconsistency, not a novel edge case.

**Fix:** normalize case only for the *comparison*, not the final relpath:
```python
normed = os.path.normcase(os.path.normpath(path))
for h_prefix, c_prefix in _map:
    h_normed = os.path.normcase(os.path.normpath(h_prefix))
    if normed == h_normed or normed.startswith(h_normed + os.sep):
        rel = os.path.relpath(path, h_prefix).replace("\\", "/")
        return f"{c_prefix}/{rel}" if rel != "." else c_prefix
return path
```

**Fix Analysis:** The proposed fix is correct. The key insight — `normcase` for comparison but `os.path.relpath(path, h_prefix)` with the original-case paths for the final result — preserves the actual path casing in the container path string. This matches the exact pattern already used in `tool_handlers._norm`. No other changes needed.

### 51. Bash Truncation Suggests Impossible `view_lines` Call — ✅ Confirmed, low severity
`_smart_truncate`'s bridge message ("use view_lines for the full content") is generic, but when applied to bash stdout/stderr, there is no file to `view_lines` — it's ephemeral process output, not something on disk (unless the command itself redirected to a file).

**Fix:** pass a bash-specific message instead of the generic one, e.g. "output truncated — redirect to a file (`cmd > out.txt`) and view_lines that, or narrow the command's output."

**Fix Analysis:** The proposed fix is correct. Looking at `_smart_truncate` in `tool_handlers.py` (line 63–75), it has a hardcoded bridge message. The cleanest fix is to add an optional `hint` parameter:
```python
def _smart_truncate(text: str, max_chars: int = 8000, head: int = 3000,
                    hint: str = "use view_lines for the full content") -> str:
    ...
    bridge = f"\n[SYSTEM: {omitted:,} chars omitted — {hint}] ...\n"
```
Then the bash handler passes `hint="redirect to a file (cmd > out.txt) and view_lines that, or narrow the command's output"`. All other callers continue to work unchanged with the default `hint`.

### 52. Duplicate `web_date_context()` Injection — ✅ Confirmed, minor
When both `web` and `research` groups are active, the loop in `_base_system` (and identically in `_review_system`) appends the `### REFERENCE DATE` block once per matching group — twice total, verbatim.

**Fix:** track injection with a flag so it only happens once per prompt build:
```python
_date_injected = False
for group in (...):
    ...
    if group in ("web", "research") and not _date_injected:
        prompt += "\n\n" + web_date_context()
        _date_injected = True
```

**Fix Analysis:** The proposed fix is correct. Looking at `_base_system` and `_review_system` in `emperor_agent.py` (lines 300–311 and 322–328), both have the same `if group in ("web", "research"): prompt += web_date_context()` pattern in their loop. The flag approach works, but an even cleaner fix since the loop order is deterministic ("web" comes before "research"): after the loop, check once:
```python
if any(g in self._active_groups for g in ("web", "research")):
    prompt += "\n\n" + web_date_context()
```
This is outside the loop entirely and avoids the flag variable. Either approach is correct.

---

## Summary

Out of everything reviewed, essentially all items are real, reproducible bugs when traced through the actual code paths. The main corrections to Gemini's report:

- **#14, #19, #33** — real bugs, but the stated *impact* is bigger than what actually happens (data isn't fully lost / detection isn't fully disabled / files aren't fully invisible).
- **#5, #6, #9** — real, but the exact failure mode (which exception class, "entire agent" vs. "current turn," "UI freeze" framing) is slightly imprecise.
- **#39** — actually has an extra wrinkle Gemini didn't catch (`.hpp` isn't just badly chunked, it's not indexed at all).

Nothing on the list was an outright false positive — Gemini's review holds up well against the actual source.
