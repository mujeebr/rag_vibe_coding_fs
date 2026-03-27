#!/usr/bin/env python3
"""
Web page parser for RAG ingestion.

Best-effort extraction strategy:
1) trafilatura (best for main-text extraction) if installed
2) readability-lxml (article extraction) if installed
3) BeautifulSoup fallback (removes boilerplate tags and gets visible text)

Usage:
  python parsing/parse_website.py "https://example.com/some-article" --out parsing/parsed_results.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple
from urllib.parse import urljoin, urlparse


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base = (parsed.netloc + parsed.path).strip("/")
    if not base:
        base = "page"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return base[:120]


def _same_site(a: str, b: str) -> bool:
    pa = urlparse(a)
    pb = urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ParsedPage:
    url: str
    fetched_at_utc: str
    title: Optional[str]
    text: str
    extractor: str

    def to_txt(self) -> str:
        header = [
            f"URL: {self.url}",
            f"Fetched-At (UTC): {self.fetched_at_utc}",
            f"Extractor: {self.extractor}",
        ]
        if self.title:
            header.append(f"Title: {self.title}")
        header_txt = "\n".join(header)
        body = _normalize_whitespace(self.text)
        return f"{header_txt}\n\n{body}\n"


def fetch_url(url: str, timeout_s: int = 30, user_agent: str = "RAGParser/1.0 (+https://example.com)") -> Tuple[str, str]:
    """
    Returns (final_url, html).
    Uses requests if available; otherwise falls back to urllib.
    """
    try:
        import requests  # type: ignore

        resp = requests.get(
            url,
            timeout=timeout_s,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.url, resp.text
    except ModuleNotFoundError:
        pass

    from urllib.request import Request, urlopen

    req = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout_s) as r:  # nosec - user-controlled URL is expected here
        final_url = r.geturl()
        raw = r.read()
        # best-effort decode
        try:
            html = raw.decode("utf-8")
        except UnicodeDecodeError:
            html = raw.decode("latin-1", errors="replace")
        return final_url, html


def extract_main_text(html: str, url: str) -> ParsedPage:
    fetched_at = datetime.now(timezone.utc).isoformat()

    # 1) trafilatura
    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            output_format="txt",
        )
        if downloaded and downloaded.strip():
            title = None
            try:
                import bs4  # type: ignore  # noqa: F401
                from bs4 import BeautifulSoup  # type: ignore

                soup = BeautifulSoup(html, "html.parser")
                if soup.title and soup.title.get_text(strip=True):
                    title = soup.title.get_text(strip=True)
            except Exception:
                title = None
            return ParsedPage(url=url, fetched_at_utc=fetched_at, title=title, text=downloaded, extractor="trafilatura")
    except ModuleNotFoundError:
        pass
    except Exception:
        # fall through to other extractors
        pass

    # 2) readability-lxml
    try:
        from readability import Document  # type: ignore

        doc = Document(html)
        title = doc.short_title() or None
        summary_html = doc.summary(html_partial=True)

        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text("\n", strip=True)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", summary_html)

        if text and text.strip():
            return ParsedPage(url=url, fetched_at_utc=fetched_at, title=title, text=text, extractor="readability-lxml")
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    # 3) BeautifulSoup fallback
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None

        main = soup.find("main")
        if main is None:
            main = soup.find("article")
        if main is None:
            main = soup.body or soup

        text = main.get_text("\n", strip=True)
        return ParsedPage(url=url, fetched_at_utc=fetched_at, title=title, text=text, extractor="bs4-fallback")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency for HTML parsing. Install at least one of:\n"
            "- trafilatura\n"
            "- readability-lxml + beautifulsoup4\n"
            "- beautifulsoup4\n"
        ) from e


def write_output(out_path: str, page: ParsedPage) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page.to_txt())


def _append_output(out_path: str, pages: Iterable[ParsedPage]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(pages):
            if i:
                f.write("\n" + ("=" * 80) + "\n\n")
            f.write(page.to_txt())


def _parse_sitemap_urls(xml_text: str) -> list[str]:
    import xml.etree.ElementTree as ET

    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return urls

    for el in root.iter():
        if el.tag.lower().endswith("loc") and el.text:
            loc = el.text.strip()
            if _is_http_url(loc):
                urls.append(loc)
    # preserve order while deduping
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def discover_urls(start_url: str, timeout_s: int, user_agent: str, max_pages: int) -> list[str]:
    """
    Best-effort URL discovery:
    - Prefer sitemap.xml (and sitemap index) if present.
    - Fallback: single-page link extraction from the start_url.
    """
    candidates: list[str] = []
    sitemap_urls = [urljoin(start_url, "/sitemap.xml"), urljoin(start_url, "/sitemap_index.xml")]
    for sm in sitemap_urls:
        try:
            final_sm, xml_text = fetch_url(sm, timeout_s=timeout_s, user_agent=user_agent)
            if not _same_site(start_url, final_sm):
                continue
            found = _parse_sitemap_urls(xml_text)
            if found:
                # if it is a sitemap index, it may contain other sitemap URLs; pull one level deep
                nested: list[str] = []
                for u in found[: min(len(found), 50)]:
                    if u.endswith(".xml"):
                        try:
                            _, inner = fetch_url(u, timeout_s=timeout_s, user_agent=user_agent)
                            nested.extend(_parse_sitemap_urls(inner))
                        except Exception:
                            continue
                candidates = nested or found
                break
        except Exception:
            continue

    if not candidates:
        # Fallback: extract internal links from the start page only
        final_url, html = fetch_url(start_url, timeout_s=timeout_s, user_agent=user_agent)
        base = final_url
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                u = urljoin(base, href)
                if not _is_http_url(u):
                    continue
                if _same_site(base, u):
                    candidates.append(u)
        except ModuleNotFoundError:
            # very rough fallback
            for m in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
                u = urljoin(base, m)
                if _is_http_url(u) and _same_site(base, u):
                    candidates.append(u)

    # Normalize + dedupe + cap
    seen: set[str] = set()
    out: list[str] = []
    for u in candidates:
        if len(out) >= max_pages:
            break
        if "#" in u:
            u = u.split("#", 1)[0]
        if u not in seen and _same_site(start_url, u):
            seen.add(u)
            out.append(u)

    # Ensure start_url is first
    if start_url not in seen:
        out.insert(0, start_url)
    return out[:max_pages]


def crawl_site(
    start_url: str,
    out_path: str,
    timeout_s: int,
    user_agent: str,
    max_pages: int,
    delay_s: float,
) -> int:
    import time

    urls = discover_urls(start_url, timeout_s=timeout_s, user_agent=user_agent, max_pages=max_pages)
    pages: list[ParsedPage] = []

    for idx, u in enumerate(urls, start=1):
        try:
            final_url, html = fetch_url(u, timeout_s=timeout_s, user_agent=user_agent)
            page = extract_main_text(html, url=final_url)
            pages.append(page)
            print(f"[{idx}/{len(urls)}] Parsed: {final_url}")
        except Exception as e:
            print(f"[{idx}/{len(urls)}] Skipped: {u} ({type(e).__name__}: {e})", file=sys.stderr)
        if delay_s > 0 and idx < len(urls):
            time.sleep(delay_s)

    _append_output(out_path, pages)

    # Also optionally write per-page files next to the combined output
    try:
        out_dir = os.path.dirname(out_path) or "."
        per_page_dir = os.path.join(out_dir, "pages")
        os.makedirs(per_page_dir, exist_ok=True)
        for page in pages:
            name = f"{_safe_filename_from_url(page.url)}__{_stable_id(page.url)}.txt"
            write_output(os.path.join(per_page_dir, name), page)
    except Exception:
        pass

    print(f"Wrote combined parsed text to: {out_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch a URL (or crawl a site), extract main text, write to .txt for RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python parsing/parse_website.py "https://example.com/blog/post" --out parsing/parsed_results.txt
              python parsing/parse_website.py "https://fullstackacademy.in" --crawl --max-pages 200 --out parsing/fullstackacademy.txt

            Notes:
              - Respect robots.txt / site terms for any website you scrape.
              - If a site is JS-rendered, consider switching to Playwright-based fetching.
            """
        ).strip(),
    )
    p.add_argument("url", help="Web page URL to fetch and parse.")
    p.add_argument(
        "--out",
        default="parsing/parsed_results.txt",
        help="Output .txt file path (default: parsing/parsed_results.txt).",
    )
    p.add_argument(
        "--crawl",
        action="store_true",
        help="If set, discover and parse multiple pages from the same site (prefers sitemap.xml).",
    )
    p.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl when --crawl is set (default: 100).")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds for --crawl (default: 0.5).")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds (default: 30).")
    p.add_argument(
        "--user-agent",
        default="RAGParser/1.0 (+https://example.com)",
        help="User-Agent header for requests (default: RAGParser/1.0 ...).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    url = args.url.strip()
    if not _is_http_url(url):
        print("Error: URL must start with http:// or https://", file=sys.stderr)
        return 2

    if args.crawl:
        return crawl_site(
            start_url=url,
            out_path=args.out,
            timeout_s=args.timeout,
            user_agent=args.user_agent,
            max_pages=max(1, args.max_pages),
            delay_s=max(0.0, args.delay),
        )

    final_url, html = fetch_url(url, timeout_s=args.timeout, user_agent=args.user_agent)
    page = extract_main_text(html, url=final_url)
    write_output(args.out, page)

    # Optional convenience: also write a URL-derived file next to the chosen output.
    # This helps when batching and you want unique filenames.
    try:
        import os

        out_dir = os.path.dirname(args.out) or "."
        derived = os.path.join(out_dir, f"{_safe_filename_from_url(final_url)}.txt")
        if os.path.abspath(derived) != os.path.abspath(args.out):
            write_output(derived, page)
    except Exception:
        pass

    print(f"Wrote parsed text to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

