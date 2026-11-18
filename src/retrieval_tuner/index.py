from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


def load_chunks(docs_dir: Path, size: int = 280, overlap: int = 40) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        step = max(size - overlap, 1)
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append({"doc_id": path.stem, "text": text[start:end], "id": f"{path.stem}:{idx}"})
            if end == len(text):
                break
            start += step
            idx += 1
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        n = len(tokenized) or 1
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        self.vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            length = len(tokens) or 1
            self.vectors.append({t: (c / length) * self.idf[t] for t, c in tf.items()})

    def search(self, query: str, k: int = 3) -> list[dict]:
        q_tokens = tokenize(query)
        tf = Counter(q_tokens)
        length = len(q_tokens) or 1
        q_vec = {t: (c / length) * self.idf.get(t, 0.0) for t, c in tf.items()}
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            keys = set(q_vec) | set(vec)
            if not keys:
                score = 0.0
            else:
                dot = sum(q_vec.get(x, 0.0) * vec.get(x, 0.0) for x in keys)
                na = math.sqrt(sum(v * v for v in q_vec.values()) or 1)
                nb = math.sqrt(sum(v * v for v in vec.values()) or 1)
                score = dot / (na * nb)
            scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
