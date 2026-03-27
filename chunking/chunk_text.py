#!/usr/bin/env python3
"""
Chunk parsed website text for RAG + Chroma ingestion.

Input format:
- The output produced by parsing/parse_website.py (single page or multi-page combined)
- Pages are separated by a line of 80 '=' characters (written by _append_output)
- Each page begins with header lines like:
    URL: ...
    Fetched-At (UTC): ...
    Extractor: ...
    Title: ... (optional)
  followed by a blank line, then the main text.

Outputs:
- JSONL file with one record per chunk:
    {"id": "...", "document": "...", "metadata": {...}}
  This is directly compatible with Chroma's add() pattern:
    collection.add(ids=[...], documents=[...], metadatas=[...])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


SEPARATOR_RE = re.compile(r"^\s*={20,}\s*$")


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Character-based chunking with overlap.
    Returns list of {"text": ..., "start": ..., "end": ...}.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = _normalize(text)
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)
    step = chunk_size - overlap

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append({"text": chunk, "start": start, "end": end})
        if end == n:
            break
        start += step

    return chunks


@dataclass
class ParsedDoc:
    url: Optional[str]
    fetched_at_utc: Optional[str]
    extractor: Optional[str]
    title: Optional[str]
    text: str


def _split_pages(raw: str) -> List[str]:
    lines = raw.splitlines()
    pages: List[List[str]] = [[]]
    for line in lines:
        if SEPARATOR_RE.match(line):
            if pages[-1]:
                pages.append([])
            continue
        pages[-1].append(line)
    return ["\n".join(p).strip() for p in pages if "\n".join(p).strip()]


def _parse_page(page_text: str) -> ParsedDoc:
    """
    Parses the header block and body from one page section.
    If headers aren't present, treats whole thing as body.
    """
    url = fetched_at_utc = extractor = title = None

    lines = page_text.splitlines()
    body_start = 0
    for i, line in enumerate(lines[:30]):  # headers are always at the top
        if not line.strip():
            body_start = i + 1
            break
        if line.startswith("URL:"):
            url = line[len("URL:") :].strip() or None
        elif line.startswith("Fetched-At (UTC):"):
            fetched_at_utc = line[len("Fetched-At (UTC):") :].strip() or None
        elif line.startswith("Extractor:"):
            extractor = line[len("Extractor:") :].strip() or None
        elif line.startswith("Title:"):
            title = line[len("Title:") :].strip() or None
        else:
            # If we see non-header content before a blank line, assume no headers.
            if i == 0:
                body_start = 0
                break
    body = "\n".join(lines[body_start:]).strip()
    if body_start == 0 and (url or fetched_at_utc or extractor or title):
        # still ensure we don't drop content
        body = "\n".join(lines).strip()
    return ParsedDoc(url=url, fetched_at_utc=fetched_at_utc, extractor=extractor, title=title, text=body)


def to_chroma_jsonl_records(
    docs: Iterable[ParsedDoc],
    chunk_size: int,
    overlap: int,
    source_file: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for doc_i, doc in enumerate(docs):
        chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        total = len(chunks)
        for chunk_i, ch in enumerate(chunks):
            doc_url = doc.url or "unknown"
            rec_id = _stable_id(source_file, doc_url, str(doc_i), str(chunk_i), str(ch["start"]))
            metadata: Dict[str, Any] = {
                "source": source_file,
                "source_type": "web",
                "url": doc.url,
                "title": doc.title,
                "extractor": doc.extractor,
                "fetched_at_utc": doc.fetched_at_utc,
                "doc_index": doc_i,
                "chunk_index": chunk_i,
                "total_chunks": total,
                "chunk_start": ch["start"],
                "chunk_end": ch["end"],
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
            records.append({"id": rec_id, "document": ch["text"], "metadata": metadata})
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunk parsed text into Chroma-ingestable JSONL.")
    p.add_argument(
        "--in",
        dest="in_path",
        default="parsing/parsed_results.txt",
        help="Input parsed .txt file (default: parsing/parsed_results.txt).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        default="chunking/chunks.jsonl",
        help="Output JSONL file (default: chunking/chunks.jsonl).",
    )
    p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters (default: 1000).")
    p.add_argument("--overlap", type=int, default=100, help="Overlap in characters (default: 100).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.exists(args.in_path):
        print(f"Error: input file not found: {args.in_path}", file=sys.stderr)
        return 2

    with open(args.in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    pages = _split_pages(raw)
    docs = [_parse_page(p) for p in pages]
    records = to_chroma_jsonl_records(
        docs=docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        source_file=os.path.basename(args.in_path),
    )
    write_jsonl(args.out_path, records)

    print(f"Wrote {len(records)} chunks to: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

