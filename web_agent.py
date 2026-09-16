# web_agent.py
from utils import read_api_key, rerank_passages, load_json, save_json_atomic, extract_keywords, console
import atexit
import json
import os
import re
import requests
import time
from datetime import date
from typing import List, Dict

from linkup import LinkupClient

import config

CACHE_FILE = config.WEB_CACHE_FILE
CACHE_TTL  = 24 * 60 * 60 * 30   # 30 days

# item 16 / SP-17: throttled autosave interval — the cache is otherwise only
# flushed at process exit, so this protects against losing a long session's
# cache if the process dies without a clean exit (kill -9, crash, etc).
CACHE_AUTOSAVE_INTERVAL = 300  # seconds

# ══════════════════════════════════════════════════════════════════════════════
# CONTENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class ContentExtractor:
    """Advanced content extraction with multiple strategies."""

    def __init__(self):
        pass

    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract fenced code blocks."""
        try:
            pattern = re.compile(r'```(\w+)?[ \t\r]*\n(.*?)```', re.DOTALL)
            blocks = []
            for match in pattern.finditer(text):
                language = match.group(1) or 'text'
                code = match.group(2).strip()
                blocks.append({'type': 'code', 'language': language, 'content': code, 'priority': 5})
            return blocks
        except Exception:
            return []

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract markdown sections."""
        try:
            sections = []
            parts = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)

            current_header = None
            for part in parts:
                if re.match(r'^#{1,6}\s+', part):
                    current_header = part.strip()
                elif part.strip():
                    # Bug 31 fix: treat pre-header text as its own section
                    header = current_header or "Introduction"
                    sections.append({
                        'type': 'section',
                        'header': header,
                        'content': part.strip(),
                        'priority': self._calc_priority(header, part)
                    })
            return sections
        except Exception:
            return []

    def _calc_priority(self, header: str, content: str) -> int:
        """Calculate section relevance priority — general-purpose, no topic bias."""
        priority = 5

        # Penalise boilerplate / navigation sections that add no substance
        low_value = ['table of contents', 'navigation', 'footer', 'copyright',
                     'license', 'sidebar', 'breadcrumb', 'cookie']

        h_lower = header.lower()

        if any(kw in h_lower for kw in low_value):
            priority -= 3

        # Boost longer, more substantive sections (likely real content)
        if len(content) > 500:
            priority += 2
        elif len(content) < 50:
            priority -= 1

        return max(1, priority)

    def extract_lists(self, text: str) -> List[Dict[str, str]]:
        """Extract bullet and numbered lists."""
        try:
            lists   = []
            pattern = re.compile(r'((?:^[\s]*[-*+]\s+.+$\n?)+|(?:^\d+\.\s+.+$\n?)+)', re.MULTILINE)
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                if len(content) > 20:
                    lists.append({'type': 'list', 'content': content, 'priority': 6})
            return lists
        except Exception:
            return []

    def extract_tables(self, text: str) -> List[Dict[str, str]]:
        """Extract markdown tables."""
        try:
            tables  = []
            pattern = re.compile(r'(\|.+\|\n\|[\s:|-]+\|\n(?:\|.+\|\n?)+)', re.MULTILINE)
            for match in pattern.finditer(text):
                tables.append({'type': 'table', 'content': match.group(1).strip(), 'priority': 7})
            return tables
        except Exception:
            return []

    def keyword_relevance_score(self, text: str, keywords: List[str]) -> int:
        """Score text relevance against a keyword list."""
        if not keywords:
            return 5
        text_lower = text.lower()
        matches    = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(10, int((matches / len(keywords)) * 10) + 3)

    def deduplicate_content(self, parts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove near-duplicate content blocks."""
        seen   = set()
        unique = []
        for part in parts:
            fingerprint = part['content'][:100].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(part)
        return unique

    def smart_truncate(self, text: str, max_tokens: int = 2000, keywords: List[str] = None) -> str:
        """Intelligently truncate by extracting highest-priority content blocks."""
        all_parts = []
        all_parts.extend(self.extract_code_blocks(text))
        all_parts.extend(self.extract_sections(text))
        all_parts.extend(self.extract_lists(text))
        all_parts.extend(self.extract_tables(text))

        if not all_parts:
            return self._fallback_truncate(text, max_tokens)

        if keywords:
            for part in all_parts:
                score = self.keyword_relevance_score(part['content'], keywords)
                part['priority'] += score

        all_parts = sorted(all_parts, key=lambda x: x['priority'], reverse=True)
        all_parts = self.deduplicate_content(all_parts)

        result    = []
        total     = 0
        max_chars = max_tokens * 4

        for part in all_parts:
            content = part['content']
            if total + len(content) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    result.append(self._format_part(part, content[:remaining] + "\n[...]"))
                break
            result.append(self._format_part(part, content))
            total += len(content)

        final = "\n\n".join(result)
        return final if final.strip() else self._fallback_truncate(text, max_tokens)

    def _format_part(self, part: Dict[str, str], content: str) -> str:
        ptype = part['type']
        if ptype == 'code':
            lang = part.get('language', 'text')
            return f"**CODE ({lang.upper()}):**\n```{lang}\n{content}\n```"
        elif ptype == 'section':
            header = part.get('header', 'Section')
            return f"{header}\n{content}"
        elif ptype == 'list':
            return f"**KEY POINTS:**\n{content}"
        elif ptype == 'table':
            return f"**DATA:**\n{content}"
        return content

    def _fallback_truncate(self, text: str, max_tokens: int) -> str:
        """Simple paragraph-by-paragraph truncation fallback."""
        max_chars  = max_tokens * 4
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        result = []
        total  = 0
        for para in paragraphs:
            if total + len(para) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    result.append(para[:remaining] + "...")
                break
            result.append(para)
            total += len(para)
        return "\n\n".join(result)

    def extract_for_query(self, text: str, query: str, max_tokens: int = 2000) -> str:
        """Extract the most query-relevant content from raw text."""
        console.print(f"[dim]Extracting for query...[/dim]")

        keywords     = extract_keywords(query)
        result       = self.smart_truncate(text, max_tokens, keywords)
        original     = len(text) // 4
        result_tokens = len(result) // 4
        savings      = ((original - result_tokens) / original * 100) if original > 0 else 0

        console.print(f"[dim]Extraction: {original:,} → {result_tokens:,} tokens ({savings:.1f}% saved)[/dim]")
        return result


class DocumentationExtractor(ContentExtractor):
    """ContentExtractor specialised for technical API documentation."""

    def extract_api_info(self, text: str) -> Dict[str, str]:
        """Extract endpoints, parameters, and error codes from documentation."""
        info = {'endpoints': [], 'parameters': [], 'errors': []}

        endpoint_pattern = r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}]+)'
        info['endpoints'] = re.findall(endpoint_pattern, text)

        param_pattern = r'(?:^|\n)[\s]*[-*]?\s*`?(\w+)`?\s*:\s*(.+?)(?=\n|$)'
        info['parameters'] = re.findall(param_pattern, text)

        error_pattern = r'(?:Error|Exception|HTTP)\s*(\d{3,4})[:\s]+(.+?)(?=\n|$)'
        info['errors'] = re.findall(error_pattern, text)

        return info

    def format_for_developer(self, text: str, query: str, max_tokens: int = 2000) -> str:
        """Format documentation with code examples first, then endpoints and parameters."""
        console.print(f"[dim]Developer extraction...[/dim]")

        api_info     = self.extract_api_info(text)
        result_parts = []
        tokens_used  = 0
        max_chars    = max_tokens * 4

        # 1. Code examples (highest priority)
        code_blocks = self.extract_code_blocks(text)
        if code_blocks:
            result_parts.append("## CODE EXAMPLES\n")
            for block in code_blocks[:3]:
                formatted = f"```{block['language']}\n{block['content']}\n```"
                if tokens_used + len(formatted) < max_chars:
                    result_parts.append(formatted)
                    tokens_used += len(formatted)

        # 2. API endpoints
        if api_info['endpoints']:
            endpoints = "## API ENDPOINTS\n" + "\n".join(api_info['endpoints'])
            if tokens_used + len(endpoints) < max_chars:
                result_parts.append(endpoints)
                tokens_used += len(endpoints)

        # 3. Parameters
        if api_info['parameters']:
            params = "## PARAMETERS\n" + "\n".join([f"- `{p[0]}`: {p[1]}" for p in api_info['parameters']])
            if tokens_used + len(params) < max_chars:
                result_parts.append(params)
                tokens_used += len(params)

        # 4. Fill remaining budget with query-relevant content
        if tokens_used < max_chars * 0.7:
            keywords  = extract_keywords(query)
            remaining = (max_chars - tokens_used) // 4
            extra     = self.smart_truncate(text, remaining, keywords)
            result_parts.append(extra)

        final = "\n\n".join(result_parts)
        console.print(f"[dim]Extracted {len(final)//4:,} tokens[/dim]")
        return final


# ══════════════════════════════════════════════════════════════════════════════
# WEB AGENT
# ══════════════════════════════════════════════════════════════════════════════

class WebAgent:
    """
    Agent responsible for web browsing, executing Google searches, reading raw URLs,
    and delegating semantic HTML extraction to the ContentExtractor.
    """
    def __init__(self):
        try:
            self.api_key = read_api_key(config.LINKUP_API_KEY_FILE, silence_warning=True)
            self.client  = LinkupClient(api_key=self.api_key) if self.api_key else None
            if not self.api_key:
                console.print("⚠️  [yellow]WebAgent: Linkup key not found — web search disabled[/yellow]")

            self.cache         = self._load_cache()
            self._cache_dirty       = False   # item 16 / SP-17
            self._cache_last_saved  = time.time()
            atexit.register(self.flush_cache)
            self.extractor     = ContentExtractor()
            self.doc_extractor = DocumentationExtractor()

            try:
                self._nvidia_key = read_api_key(config.NVIDIA_API_KEY_FILE, silence_warning=True)
                if not self._nvidia_key:
                    console.print("⚠️  [yellow]WebAgent: NVIDIA key not found — reranking disabled[/yellow]")
            except Exception:
                self._nvidia_key = None
                console.print("⚠️  [yellow]WebAgent: NVIDIA key not found — reranking disabled[/yellow]")

        except Exception as e:
            console.print(f"🚨 [bold red]Web Agent init failed: {e}[/bold red]")
            raise

    # ── Reranking ──────────────────────────────────────────────────────────────

    def _rerank_snippets(self, query: str, snippets: list) -> list:
        """
        Re-order snippets by relevance to query using NVIDIA NIM cross-encoder.
        Falls back to original order on any error — non-fatal.
        """
        order = rerank_passages(query, snippets, api_key=self._nvidia_key, label="snippets")
        return [snippets[i] for i in order]

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        data = load_json(CACHE_FILE, default={})
        # Bug 9: if the JSON file is corrupted and doesn't deserialize to a
        # dict, silently discard it so every subsequent cache lookup is a miss
        # rather than an AttributeError/TypeError crash.
        return data if isinstance(data, dict) else {}

    def _save_cache(self):
        save_json_atomic(CACHE_FILE, self.cache)
        self._cache_dirty      = False
        self._cache_last_saved = time.time()

    def _mark_cache_dirty(self):
        """
        Mark the cache as having unsaved changes instead of writing the
        entire cache file to disk synchronously after every single search or
        read_url call (item 16 / SP-17). For a session with hundreds of
        cached entries, a full atomic rewrite per call adds up. The cache is
        flushed:
          (a) opportunistically, if more than CACHE_AUTOSAVE_INTERVAL seconds
              have passed since the last save, and
          (b) always, at process exit via atexit (see flush_cache()).
        """
        self._cache_dirty = True
        if time.time() - self._cache_last_saved > CACHE_AUTOSAVE_INTERVAL:
            self._save_cache()

    def flush_cache(self):
        """Final safety-net save, registered with atexit — ensures a clean
        process exit never drops cache entries written since the last
        throttled autosave."""
        if self._cache_dirty:
            try:
                self._save_cache()
            except Exception:
                pass

    def _is_cache_valid(self, key: str) -> bool:
        entry = self.cache.get(key)
        # Bug 9: guard against corrupted/non-dict cache entries that would
        # raise TypeError/KeyError when accessing entry["timestamp"].
        if not isinstance(entry, dict):
            return False
        ts = entry.get("timestamp")
        if not isinstance(ts, (int, float)):
            return False
        return (time.time() - ts) < CACHE_TTL

    # ── Content cleaning ───────────────────────────────────────────────────────

    def _clean_content(self, text: str) -> str:
        """Strip navigation noise and normalise whitespace."""
        try:
            noise = [
                r"\[Skip to content\].*?(\n|$)",
                r"\[Join the.*?waiting list\].*?(\n|$)",
                r"\[Follow.*?on.*?\]\(.*?\)",
                r"\[Subscribe to.*?newsletter\].*?(\n|$)",
                r"Made with \[Material for MkDocs\].*?(\n|$)",
                r"Copyright ©.*?(\n|$)",
            ]
            for pattern in noise:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

            text = re.sub(r'\n{3,}', '\n\n', text)
            text = "\n".join([line.rstrip() for line in text.splitlines()])
            return text.strip()
        except Exception:
            return text

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, query: str, from_date: str = None) -> str:
        """
        Web search with 30-day caching.

        Args:
            query:     search query.
            from_date: optional ISO date string ('YYYY-MM-DD'). When set,
                       Linkup restricts results to content published on or
                       after this date — useful when the model knows
                       recency matters (e.g. "current" notifications, specs,
                       policies) and wants to filter out stale SEO content
                       that would otherwise conflict with fresher sources.
        """
        # FIX #3: guard against None client when Linkup key is not configured
        if not self.client:
            return "[SYSTEM: web search unavailable — Linkup API key not configured. Add key to linkup_api_key.txt and restart.]"

        parsed_from_date = None
        if from_date:
            try:
                parsed_from_date = date.fromisoformat(from_date.strip())
            except ValueError:
                console.print(
                    f"⚠️  [yellow]web_agent.search: invalid from_date '{from_date}' "
                    f"(expected YYYY-MM-DD) — ignoring[/yellow]"
                )

        # from_date is part of the cache key so a dated search never returns
        # a stale unfiltered cache hit (or vice versa).
        cache_key = f"{query.lower().strip()}|from={parsed_from_date.isoformat() if parsed_from_date else ''}"

        if self._is_cache_valid(cache_key):
            console.print(f"📦 [dim]Cache hit: '{query}'[/dim]")
            return self.cache[cache_key]["content"]

        date_note = f" (from {parsed_from_date.isoformat()})" if parsed_from_date else ""
        console.print(f"🌐 [dim]Searching: '{query}'{date_note}...[/dim]")

        try:
            search_kwargs = dict(
                query=query,
                depth="standard",
                output_type="searchResults",
                #maxResults=config.LINKUP_MAX_RESULTS,
            )
            if parsed_from_date:
                search_kwargs["from_date"] = parsed_from_date
            response = self.client.search(**search_kwargs)
        except Exception as e:
            console.print(f"🚨 [bold red]Search error: {e}[/bold red]")
            return f"Error: Search failed - {str(e)}"

        snippets = []
        for result in response.results:
            try:
                url     = getattr(result, 'url', 'N/A')
                content = getattr(result, 'content', '')

                if content:
                    clean  = self._clean_content(content)
                    suffix = "...[use url_search for more]" if len(clean) >= 3000 else ""
                    snippets.append(f"URL: {url}\n[CONTENT]: {clean[:3000]}{suffix}")
            except Exception as e:
                console.print(f"⚠️  [yellow]web_agent.search: skipping malformed result: {e}[/yellow]")
                continue

        snippets = self._rerank_snippets(query, snippets)

        final = "\n\n".join(snippets)
        if len(final) > 21000:
            final = final[:21000] + "\n\n[SYSTEM: TRUNCATED — result exceeded 21,000 chars]"

        # Bug #32 fix: don't cache empty results — a 30-day TTL on an empty
        # string would lock in zero results for a month for the same query.
        if final:
            self.cache[cache_key] = {"timestamp": time.time(), "content": final}
            self._mark_cache_dirty()
        # console.print(f"[dim]{final}[/dim]")  # debug: dumps full snippet content to console
        return final

    def read_url(self, url: str, query_context: str = None, max_tokens: int = 6000) -> str:
        """Read a URL and intelligently extract the most relevant content."""
        # Bug 8: guard against missing API key — avoids sending "Bearer None"
        # which causes an HTTP 401 crash instead of a graceful error message.
        if not self.api_key:
            return ("[SYSTEM: url_search unavailable — Linkup API key not configured. "
                    "Add key to linkup_api_key.txt and restart.]")

        cache_key = f"{url}_{query_context or 'default'}_{max_tokens}"

        if self._is_cache_valid(cache_key):
            console.print(f"📦 [dim]URL cache hit: '{url}'[/dim]")
            return self.cache[cache_key]["content"]

        console.print(f"[dim]Reading: '{url}'...[/dim]")

        try:
            resp = requests.post(
                "https://api.linkup.so/v1/fetch",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"url": url, "renderJs": True},
                timeout=60,
            )
            resp.raise_for_status()
            combined = resp.json().get("markdown", "")
        except Exception as e:
            console.print(f"🚨 [bold red]URL error: {e}[/bold red]")
            return f"Error: URL read failed - {str(e)}"

        is_docs  = self._is_documentation(url, combined)

        if is_docs and query_context:
            console.print(f"📄 [dim]Documentation extraction...[/dim]")
            final = self.doc_extractor.format_for_developer(combined, query_context, max_tokens)
        elif query_context:
            console.print(f"[dim]Query-aware extraction...[/dim]")
            final = self.extractor.extract_for_query(combined, query_context, max_tokens)
        else:
            console.print(f"[dim]Smart truncation...[/dim]")
            final = self.extractor.smart_truncate(combined, max_tokens)

        final = f"[SOURCE: {url}]\n\n{final}"

        # Bug #32 fix: same guard for URL cache — don't cache empty page extractions.
        if final:
            self.cache[cache_key] = {"timestamp": time.time(), "content": final}
            self._mark_cache_dirty()
        # console.print(f"[dim]{final}[/dim]")  # debug: dumps full extracted page content to console
        return final

    def _is_documentation(self, url: str, content: str) -> bool:
        """Heuristic: is this URL a technical documentation page?

        Only returns True when there is strong evidence of a code/API docs page.
        Deliberately conservative — false negatives (missing a docs page) are
        far less harmful than false positives (routing physics/math/news content
        through format_for_developer, which buries prose behind CODE EXAMPLES).

        Checks (in order of reliability):
          1. URL path contains a recognised docs keyword  — very reliable
          2. Page contains ≥4 fenced code-block markers   — requires real density,
             not just a single snippet or the word 'import'
        """
        url_lower = url.lower()

        # FIX #15: use segment-aware matching for 'docs' to avoid false positives
        # like 'prodocs.example.com' or 'my-products.com' matching via raw substring.
        # 'docs.' must be at a subdomain boundary; '/docs' must be a path segment.
        if (re.search(r'(?:^|[./])docs?\.', url_lower) or   # docs. as subdomain
                re.search(r'/docs?(?:/|$)', url_lower) or    # /docs/ or /doc/ as path
                any(ind in url_lower for ind in [
                    'documentation', '/api/', '/reference/',
                    '/guide/', 'readme', '/tutorial/', '/manual/', '/sdk/',
                ])):
            return True

        # Content-based: require genuine code density (≥2 complete fenced blocks).
        # A single ``` or the word 'import'/'API' appears in ordinary articles —
        # we need multiple full blocks to be confident this is a code reference page.
        code_fence_count = len(re.findall(r'```', content[:8000]))
        if code_fence_count >= 4:   # ≥2 complete blocks → open+close pairs
            return True

        return False