#!/usr/bin/env python3
"""
RAG chat:
- loads prompt template from generation/config.yaml
- embeds user query (sentence-transformers)
- retrieves relevant chunks from persistent ChromaDB
- calls LLM (Ollama or OpenAI-compatible) OR prints the final prompt (prompt_only)
- maintains chat history as JSONL per session in generation/history/

Run:
  rag_web/bin/python generation/chat_rag.py --query "What courses do you offer?"

Interactive:
  rag_web/bin/python generation/chat_rag.py --session demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_dotenv_if_present() -> None:
    """
    Loads environment variables from .env if python-dotenv is installed.
    Safe no-op if dotenv isn't available or file doesn't exist.
    """
    if not os.path.exists(".env"):
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=".env", override=False)
    except Exception:
        return


def _read_yaml(path: str) -> Dict[str, Any]:
    import yaml  # PyYAML (already a transitive dep of chromadb)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml root in {path}; expected mapping")
    return data


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _history_path(history_dir: str, session: str) -> str:
    safe = "".join(c for c in session if c.isalnum() or c in ("-", "_"))[:80] or "default"
    return os.path.join(history_dir, f"{safe}.jsonl")


def load_history(history_file: str, max_turns: int) -> List[Dict[str, Any]]:
    if not os.path.exists(history_file):
        return []
    turns: List[Dict[str, Any]] = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except Exception:
                continue
    if max_turns > 0:
        turns = turns[-max_turns:]
    return turns


def append_history(history_file: str, role: str, content: str) -> None:
    _ensure_dir(os.path.dirname(history_file) or ".")
    rec = {"ts_utc": _utc_now(), "role": role, "content": content}
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def format_history_for_prompt(turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in turns:
        role = t.get("role", "unknown")
        content = t.get("content", "")
        if not isinstance(content, str):
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float]


def retrieve(
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    query: str,
    top_k: int,
    local_files_only: bool,
) -> List[RetrievedChunk]:
    # Reduce noisy HF/transformers logs in terminal
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import chromadb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(collection_name)

    model = SentenceTransformer(embedding_model_name, local_files_only=local_files_only)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]

    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks: List[RetrievedChunk] = []
    for i in range(min(len(ids), len(docs), len(metas))):
        chunks.append(
            RetrievedChunk(
                id=str(ids[i]),
                document=str(docs[i] or ""),
                metadata=metas[i] if isinstance(metas[i], dict) else {},
                distance=float(dists[i]) if i < len(dists) and dists[i] is not None else None,
            )
        )
    return chunks


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        url = ch.metadata.get("url")
        title = ch.metadata.get("title")
        header_bits = [f"[{i}]"]
        if title:
            header_bits.append(f"title={title}")
        if url:
            header_bits.append(f"url={url}")
        header = " ".join(header_bits)
        parts.append(f"{header}\n{ch.document}".strip())
    return "\n\n---\n\n".join(parts).strip()


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


def naive_answer(question: str, chunks: List[RetrievedChunk], max_lines: int = 10) -> str:
    """
    Fallback answer for 'prompt_only' mode: extract likely-relevant lines from retrieved chunks.
    This avoids dumping the whole prompt to the terminal when an LLM isn't configured.
    """
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", question)]
    keywords = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]

    picked: List[str] = []
    seen: set[str] = set()

    def consider(line: str) -> None:
        l = line.strip()
        if not l or len(l) < 4:
            return
        l_norm = re.sub(r"\s+", " ", l)
        if l_norm.lower() in seen:
            return
        seen.add(l_norm.lower())
        picked.append(l_norm)

    for ch in chunks:
        for line in (ch.document or "").splitlines():
            line_l = line.lower()
            if not keywords:
                consider(line)
                continue
            if any(k in line_l for k in keywords):
                consider(line)
        if len(picked) >= max_lines:
            break

    if not picked:
        return "I don't know based on the retrieved context. Try asking a more specific question."

    urls = []
    for ch in chunks:
        u = ch.metadata.get("url")
        if isinstance(u, str) and u and u not in urls:
            urls.append(u)
    sources = "\n".join(f"- {u}" for u in urls[:5])

    return "Here’s what I found in the knowledge base:\n\n" + "\n".join(f"- {l}" for l in picked[:max_lines]) + (
        f"\n\nSources:\n{sources}" if sources else ""
    )


def call_ollama(base_url: str, model: str, prompt: str) -> str:
    import requests  # type: ignore

    url = base_url.rstrip("/") + "/api/generate"
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip()


def call_openai_compatible(base_url: str, api_key: str, model: str, prompt: str) -> str:
    import requests  # type: ignore

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query ChromaDB + generate an answer with an LLM.")
    p.add_argument("--config", default="generation/config.yaml", help="Path to config.yaml.")
    p.add_argument("--session", default="default", help="Chat session name (history file key).")
    p.add_argument("--query", default=None, help="Single query. If omitted, runs interactive mode.")
    p.add_argument("--local-files-only", action="store_true", help="Only load embedding model from local cache.")
    p.add_argument("--debug", action="store_true", help="Print the full constructed prompt/context for debugging.")
    return p


def run_one(cfg: Dict[str, Any], session: str, question: str, local_files_only: bool, debug: bool) -> str:
    app = cfg.get("app", {})
    history_cfg = cfg.get("history", {})
    llm_cfg = cfg.get("llm", {})
    prompt_cfg = cfg.get("prompt", {})

    persist_dir = str(app.get("persist_dir", "chroma_db"))
    collection = str(app.get("collection", "rag_chunks"))
    embedding_model = str(app.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    top_k = int(app.get("top_k", 5))

    history_dir = str(history_cfg.get("dir", "generation/history"))
    max_turns = int(history_cfg.get("max_turns_in_prompt", 10))
    history_file = _history_path(history_dir, session)

    turns = load_history(history_file, max_turns=max_turns)
    chat_history = format_history_for_prompt(turns)

    chunks = retrieve(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model_name=embedding_model,
        query=question,
        top_k=top_k,
        local_files_only=local_files_only,
    )
    context = build_context(chunks)

    template = str(prompt_cfg.get("template", "{context}\n\nQ: {question}\nA:"))
    final_prompt = template.format(context=context, chat_history=chat_history, question=question)

    provider = str(llm_cfg.get("provider", "prompt_only"))
    if provider == "prompt_only":
        answer = final_prompt if debug else naive_answer(question, chunks)
    elif provider == "ollama":
        o = llm_cfg.get("ollama", {}) or {}
        answer = call_ollama(base_url=str(o.get("base_url", "http://localhost:11434")), model=str(o.get("model", "llama3.1")), prompt=final_prompt)
    elif provider == "openai_compatible":
        o = llm_cfg.get("openai_compatible", {}) or {}
        base_url = str(o.get("base_url", "https://api.openai.com/v1"))
        key_env = str(o.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {key_env}")
        answer = call_openai_compatible(base_url=base_url, api_key=api_key, model=str(o.get("model", "gpt-4o-mini")), prompt=final_prompt)
    else:
        raise ValueError(f"Unknown llm.provider: {provider}")

    append_history(history_file, "user", question)
    append_history(history_file, "assistant", answer)
    return answer


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _load_dotenv_if_present()
    cfg = _read_yaml(args.config)

    if args.query is not None:
        out = run_one(cfg, session=args.session, question=args.query, local_files_only=args.local_files_only, debug=args.debug)
        print(out)
        return 0

    # interactive
    print("Enter questions (blank line to exit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            a = run_one(cfg, session=args.session, question=q, local_files_only=args.local_files_only, debug=args.debug)
            print(a)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

