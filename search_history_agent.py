# search_history_agent.py - Cross-session conversation search
"""
Searches past chat sessions (config.CHAT_HISTORIES_DIR/*.jsonl) for turns
relevant to a query. Turn-level vectors are embedded once and cached in a
sidecar file (<session>.jsonl.idx.json) next to each session, so the first
search after a busy session costs one embedding call and every search after
that is free.

Extracted from tool_handlers.py's _handle_search_history, which used to be a
single 335-line method doing session parsing, incremental-index diffing,
embedding, cosine scoring, cross-encoder reranking, and result formatting all
inline (with five nested function definitions). None of those pieces were
unit-testable in isolation without invoking the full tool handler.

This mirrors DocSearchAgent's shape: one class, one public entry point
(search()), and a private method per pipeline stage so each stage can be
read, tested, or changed independently.
"""

import json
import os

import config
from utils import rerank_passages, console

FULL_THRESHOLD = 500   # chars — show the complete turn below this
RERANK_POOL    = 15    # how many top-by-score candidates get reranked


class SearchHistoryAgent:
    """
    Cross-session semantic + keyword search over past conversation turns.

    Pipeline (see search()):
      1. _list_sessions()     — find .jsonl session files
      2. _parse_turns()       — read one session into (user, assistant) pairs
      3. _reindex_sessions()  — embed any turns not yet in the sidecar cache
      4. _score_candidates()  — cosine/keyword-score every turn against query
      5. _rerank()            — cross-encoder rerank the top-scoring pool
      6. _format_results()    — build the string returned to the model
    """

    def __init__(self, histories_dir: str = None):
        self.histories_dir = histories_dir or config.CHAT_HISTORIES_DIR

    # ── Public entry point ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> str:
        if not os.path.isdir(self.histories_dir):
            return (
                "[SYSTEM: search_history] No chat history folder found yet. "
                "History builds automatically as you chat."
            )

        jsonl_files = self._list_sessions()
        if not jsonl_files:
            return "[SYSTEM: search_history] No session files found yet."

        try:
            from utils import _embed, _embed_batched, _cosine_similarity  # noqa: F401 (availability probe)
            embed_available = True
        except Exception:
            embed_available = False

        if embed_available:
            self._reindex_sessions(jsonl_files)

        query_vec = None
        if embed_available:
            try:
                query_vec = _embed([query], input_type="query")[0]
            except Exception as e:
                console.print(f"   [yellow]query embed failed: {e}[/yellow]")

        candidates = self._score_candidates(jsonl_files, query, query_vec)
        if not candidates:
            return f"[SYSTEM: search_history] No matches found for '{query}'."

        final = self._rerank(query, candidates, top_k)
        return self._format_results(query, final)

    # ── Step 1: session discovery ───────────────────────────────────────────

    def _list_sessions(self) -> list:
        return sorted(
            (
                os.path.join(self.histories_dir, fn)
                for fn in os.listdir(self.histories_dir)
                if fn.endswith(".jsonl")
            ),
            reverse=True,
        )

    # ── Step 2: parsing ──────────────────────────────────────────────────────

    def _parse_turns(self, fp: str) -> list:
        """
        Read a .jsonl and return a list of (user_text, assistant_text) pairs.
        Each pair is one conversational turn. Skips malformed lines silently.
        """
        turns = []
        pending_user = None
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    role    = obj.get("role", "")
                    content = obj.get("content", "")
                    if not isinstance(content, str):
                        if isinstance(content, list):
                            # Bug 37: extract only text blocks from multimodal content
                            # so image metadata / JSON structure doesn't pollute the index.
                            text_parts = [
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            ]
                            content = " ".join(text_parts).strip() or "[image-only message]"
                        else:
                            content = str(content)
                    if role == "user":
                        pending_user = content
                    elif role == "assistant" and pending_user is not None:
                        turns.append((pending_user, content))
                        pending_user = None
        except Exception:
            pass
        return turns

    # ── Sidecar cache (one <session>.jsonl.idx.json per session) ────────────

    def _load_sidecar(self, fp: str) -> list:
        """Load existing turn index from sidecar. Returns [] if missing/corrupt."""
        sidecar = fp + ".idx.json"
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Bug 30: filter out non-dict items that would crash .get() calls
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            pass
        return []

    def _save_sidecar(self, fp: str, entries: list):
        """Save turn index to sidecar atomically (tmp+os.replace). Non-fatal on failure."""
        from utils import save_json_atomic
        sidecar = fp + ".idx.json"
        if not save_json_atomic(sidecar, entries):
            console.print(f"   [yellow]idx sidecar save failed for {fp}[/yellow]")

    # ── Step 3: incremental turn-level indexing ─────────────────────────────
    #
    # Only embeds turns that aren't in the sidecar yet, then appends them.
    # This means the first search after a busy session costs one API call;
    # every search after that is free.

    def _reindex_sessions(self, jsonl_files: list) -> None:
        from utils import _embed_batched

        needs_index = []  # (fp, turns, existing_entries)

        for fp in jsonl_files:
            turns    = self._parse_turns(fp)
            existing = self._load_sidecar(fp)

            # 1. Prune sidecar entries if turns were deleted / rolled back
            if len(existing) > len(turns):
                existing = existing[:len(turns)]
                self._save_sidecar(fp, existing)

            # 2. Check for content desync in existing entries (e.g. after /edit or /rerun)
            mismatch_idx = None
            for idx, (u, a) in enumerate(turns[:len(existing)]):
                entry = existing[idx]
                # Bug 30: guard against corrupted (non-dict) sidecar entries
                if not isinstance(entry, dict):
                    mismatch_idx = idx
                    break
                if entry.get("user") != u or entry.get("assistant") != a:
                    mismatch_idx = idx
                    break

            if mismatch_idx is not None:
                existing = existing[:mismatch_idx]
                self._save_sidecar(fp, existing)

            if len(existing) < len(turns):
                needs_index.append((fp, turns, existing))

        if not needs_index:
            return

        total_new = sum(len(t) - len(e) for _, t, e in needs_index)
        console.print(
            f"   [dim]Indexing {total_new} new turn(s) across "
            f"{len(needs_index)} session(s)...[/dim]"
        )

        for fp, turns, existing in needs_index:
            new_turns = turns[len(existing):]  # only unindexed turns

            # Build passage text: "USER: ...\nASSISTANT: ..."
            # Truncate each half to 1000 chars so we stay under the
            # embedding model's 512-token limit (4 chars ≈ 1 token).
            # Bug #28 fix: strip <think>/<reasoning> tags from the assistant
            # text BEFORE truncating, so the 1000-char window captures the
            # actual answer rather than just the reasoning trace (which can be
            # several thousand chars long and would crowd out the real content).
            from utils import extract_thinking_tags
            passages = [
                f"USER: {u[:1000]}\nASSISTANT: {extract_thinking_tags(a)[1][:1000]}"
                for u, a in new_turns
            ]

            # Skip empty passages (guard against corrupt history lines)
            valid_indices = [i for i, p in enumerate(passages) if p.strip()]
            if not valid_indices:
                continue

            valid_passages = [passages[i] for i in valid_indices]

            try:
                vecs = _embed_batched(valid_passages, input_type="passage")
            except Exception as e:
                console.print(
                    f"   [yellow]embed failed for {os.path.basename(fp)}: {e}[/yellow]"
                )
                continue

            # Bug 16: _embed_batched() never raises on a batch failure — it
            # pads the result with None placeholders instead. Every entry in
            # `vecs` corresponds to a non-empty passage (valid_passages was
            # already filtered above), so a None here means embedding
            # genuinely failed for that passage, not that it was empty.
            # Previously these None vecs were persisted to the sidecar as if
            # indexed, which permanently hid those turns from future
            # indexing (len(existing) == len(turns) short-circuits re-index).
            # Skip persisting anything for this file this cycle instead, so
            # `existing` stays shorter than `turns` and the turns are
            # correctly retried on the next indexing pass.
            if any(v is None for v in vecs):
                failed_n = sum(1 for v in vecs if v is None)
                console.print(
                    f"   [yellow]embed batch had {failed_n} failure(s) for "
                    f"{os.path.basename(fp)} — will retry next run[/yellow]"
                )
                continue

            # Build new index entries and append to existing
            vec_map    = {valid_indices[i]: vecs[i] for i in range(len(valid_indices))}
            start_turn = len(existing) + 1  # 1-based turn number

            new_entries = []
            for i, (u, a) in enumerate(new_turns):
                new_entries.append({
                    "turn":      start_turn + i,
                    "user":      u,
                    "assistant": a,
                    "vec":       vec_map.get(i),  # None if passage was empty
                })

            self._save_sidecar(fp, existing + new_entries)

    # ── Step 4: scoring ──────────────────────────────────────────────────────

    def _score_candidates(self, jsonl_files: list, query: str, query_vec) -> list:
        """
        Return a list of (fp, turn_num, user, assistant, score, method) for
        every turn that scores > 0 against the query, across all sessions.
        """
        from utils import _cosine_similarity

        candidates = []

        for fp in jsonl_files:
            entries = self._load_sidecar(fp)
            if not entries:
                # Fallback: if no sidecar at all, parse live and do keyword match
                for turn_num, (u, a) in enumerate(self._parse_turns(fp), 1):
                    text  = (u + " " + a).lower()
                    score = sum(1 for w in query.lower().split() if len(w) > 2 and w in text)
                    score = score / (len(query.split()) + 1)
                    if score > 0:
                        candidates.append((fp, turn_num, u, a, score, "keyword"))
                continue

            for entry in entries:
                vec = entry.get("vec")
                if query_vec is not None and vec is not None:
                    try:
                        score  = float(_cosine_similarity(query_vec, [vec])[0])
                        method = "embed"
                    except Exception:
                        score, method = 0.0, "none"
                else:
                    text  = (entry.get("user", "") + " " + entry.get("assistant", "")).lower()
                    score = sum(1 for w in query.lower().split() if len(w) > 2 and w in text)
                    score  = score / (len(query.split()) + 1)
                    method = "keyword"

                if score > 0:
                    candidates.append((
                        fp,
                        entry.get("turn", 0),
                        entry.get("user", ""),
                        entry.get("assistant", ""),
                        score,
                        method,
                    ))

        return candidates

    # ── Step 5: reranking ────────────────────────────────────────────────────
    #
    # Takes the top RERANK_POOL candidates by embedding/keyword score and
    # reranks them with the NVIDIA cross-encoder. The reranker sees the actual
    # turn text (not a blurry session average), so its relevance signal is
    # much more accurate than the initial score alone.

    def _rerank(self, query: str, candidates: list, top_k: int) -> list:
        top_candidates = sorted(candidates, key=lambda c: c[4], reverse=True)[:RERANK_POOL]

        passages = [
            f"USER: {c[2][:800]}\nASSISTANT: {c[3][:800]}"
            for c in top_candidates
        ]
        order    = rerank_passages(query, passages, label="history turn(s)")
        reranked = [top_candidates[i] for i in order]
        return reranked[:top_k]

    # ── Step 6: formatting ───────────────────────────────────────────────────
    #
    # Full turn shown if user + assistant < FULL_THRESHOLD chars combined.
    # Otherwise a truncated preview + a view_lines hint with exact line
    # numbers so the model can read the full turn in one call without
    # searching for it.

    def _turn_line_range(self, fp: str, turn_num: int) -> tuple:
        """
        Return (start_line, end_line) in the .jsonl for a given 1-based turn number.
        Walks the file counting completed user+assistant pairs so that orphaned
        user lines (e.g. from a crash mid-write before the assistant line was saved)
        do not shift subsequent offsets.
        Returns (1, 2) as a safe fallback if something goes wrong.
        """
        try:
            pair_count   = 0
            pending_line = None
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for i, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    role = obj.get("role", "")
                    if role == "user":
                        pending_line = i   # always take the latest user line
                    elif role == "assistant" and pending_line is not None:
                        pair_count += 1
                        if pair_count == turn_num:
                            return pending_line, i
                        pending_line = None
            return 1, 2
        except Exception:
            return 1, 2

    def _format_results(self, query: str, final: list) -> str:
        lines = [f"[HISTORY SEARCH: '{query}'] — {len(final)} turn(s) matched\n"]

        for rank, (fp, turn_num, user_text, asst_text, score, method) in enumerate(final, 1):
            session_name = os.path.basename(fp)
            combined_len = len(user_text) + len(asst_text)
            start_line, end_line = self._turn_line_range(fp, turn_num)

            lines.append(f"{'─' * 60}")
            lines.append(
                f"[{rank}] {session_name}  |  turn #{turn_num}  |  "
                f"score: {score:.3f} ({method})"
            )
            lines.append(f"     file: {fp}  (lines {start_line}–{end_line})")
            lines.append("")

            if combined_len <= FULL_THRESHOLD:
                # Show complete turn — it's short enough
                lines.append(f"  YOU:   {user_text}")
                lines.append(f"  AGENT: {asst_text}")
            else:
                # Show preview of each half + drill-down hint
                user_preview = user_text[:300].rstrip()
                asst_preview = asst_text[:300].rstrip()
                ellipsis_u   = "..." if len(user_text) > 300 else ""
                ellipsis_a   = "..." if len(asst_text) > 300 else ""
                lines.append(f"  YOU:   {user_preview}{ellipsis_u}")
                lines.append(f"  AGENT: {asst_preview}{ellipsis_a}")
                lines.append("")
                lines.append(
                    f"  ↳ Full turn: view_lines(file=\"{fp}\", "
                    f"start={start_line}, end={end_line})"
                )

            lines.append("")

        lines.append(
            "[SYSTEM: Tip] To read the full session: view_lines(file=<path>, start=1, end=<N>)\n"
            "[SYSTEM: Tip] To search within a session: search_in_file(file=<path>, pattern=<term>)"
        )
        return "\n".join(lines)
