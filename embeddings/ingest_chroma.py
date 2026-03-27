#!/usr/bin/env python3
"""
Embed JSONL chunks and ingest into a persistent ChromaDB vectorstore.

Input:
  - chunking/chunks.jsonl (one JSON per line: {"id","document","metadata"})

Output:
  - Persistent Chroma directory (default: chroma_db/)

Example:
  python embeddings/ingest_chroma.py \
    --in chunking/chunks.jsonl \
    --persist-dir chroma_db \
    --collection rag_chunks \
    --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            yield obj


def batch_iter(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _existing_ids(collection, ids: List[str]) -> set[str]:
    """
    Chroma doesn't provide a direct bulk 'exists' API; we use get(ids=...).
    If an id doesn't exist, it just won't be returned.
    """
    if not ids:
        return set()
    try:
        got = collection.get(ids=ids, include=[])
        return set(got.get("ids") or [])
    except Exception:
        # If collection is empty or backend differs, be conservative and return none.
        return set()


def embed_texts(model, texts: List[str], batch_size: int) -> List[List[float]]:
    # sentence-transformers returns numpy arrays; convert to vanilla lists for Chroma.
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=(len(texts) >= 100),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def load_records(in_path: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for rec in read_jsonl(in_path):
        rid = rec.get("id")
        doc = rec.get("document")
        meta = rec.get("metadata")
        if not isinstance(rid, str) or not rid:
            raise ValueError("Record missing non-empty string 'id'")
        if not isinstance(doc, str):
            raise ValueError(f"Record {rid} has non-string 'document'")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError(f"Record {rid} has non-dict 'metadata'")
        ids.append(rid)
        documents.append(doc)
        metadatas.append(meta)

    return ids, documents, metadatas


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed chunk JSONL and persist to ChromaDB.")
    p.add_argument("--in", dest="in_path", default="chunking/chunks.jsonl", help="Input JSONL chunks file.")
    p.add_argument("--persist-dir", default="chroma_db", help="Directory to persist Chroma data (default: chroma_db).")
    p.add_argument("--collection", default="rag_chunks", help="Chroma collection name (default: rag_chunks).")
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (default: all-MiniLM-L6-v2).",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load models from local cache / local path (no network).",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding + upserts (default: 64).")
    p.add_argument(
        "--reset",
        action="store_true",
        help="If set, deletes and recreates the collection (DANGEROUS).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.exists(args.in_path):
        print(f"Error: input file not found: {args.in_path}", file=sys.stderr)
        return 2

    # Lazy imports (so users can run help without deps)
    import chromadb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    os.makedirs(args.persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=args.persist_dir)

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = load_records(args.in_path)
    if not ids:
        print("No records found to ingest.")
        return 0

    try:
        model = SentenceTransformer(args.model, local_files_only=args.local_files_only)
    except Exception as e:
        msg = (
            f"Error loading embedding model '{args.model}'.\n"
            "- If you're behind a restricted network/proxy, run with internet access once so the model can download.\n"
            "- Or pre-download the model and pass a local folder path via --model.\n"
            "- Or rerun with --local-files-only once the model is cached.\n"
        )
        print(msg, file=sys.stderr)
        print(f"Underlying error: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    total = len(ids)
    ingested = 0
    skipped = 0

    for idxs in batch_iter(list(range(total)), args.batch_size):
        batch_ids = [ids[i] for i in idxs]
        batch_docs = [documents[i] for i in idxs]
        batch_metas = [metadatas[i] for i in idxs]

        exists = _existing_ids(collection, batch_ids)
        to_add = [(i, bid) for i, bid in zip(idxs, batch_ids) if bid not in exists]
        if not to_add:
            skipped += len(batch_ids)
            continue

        add_ids = [ids[i] for i, _ in to_add]
        add_docs = [documents[i] for i, _ in to_add]
        add_metas = [metadatas[i] for i, _ in to_add]

        embs = embed_texts(model, add_docs, batch_size=min(args.batch_size, 128))
        collection.add(ids=add_ids, documents=add_docs, metadatas=add_metas, embeddings=embs)
        ingested += len(add_ids)

        done = ingested + skipped
        print(f"Ingested {ingested}/{total} (skipped {skipped}, processed {done})")

    print(f"Done. Persisted Chroma at: {args.persist_dir} (collection: {args.collection})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

