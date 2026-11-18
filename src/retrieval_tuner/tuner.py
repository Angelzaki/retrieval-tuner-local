from __future__ import annotations

import json
from pathlib import Path

from retrieval_tuner.index import Retriever, load_chunks

CASES = [
    ("¿Cuál es el descuento máximo en A1?", "politica_a1"),
    ("¿A2 exige MFA?", "requisitos_a2"),
    ("¿Cuál es el límite comercial?", "limite_comercial"),
    ("política de viajes interplanetarios 2099", None),
]


def evaluate(retriever: Retriever, top_k: int, min_score: float) -> dict:
    hits = 0
    abstain_ok = 0
    total = 0
    for query, expected in CASES:
        total += 1
        ranked = retriever.search(query, k=top_k)
        filtered = [r for r in ranked if r["score"] >= min_score]
        if expected is None:
            abstain_ok += int(len(filtered) == 0 or filtered[0]["score"] < min_score)
            if len(filtered) == 0:
                hits += 1
            continue
        doc_ids = [r["doc_id"] for r in filtered]
        hits += int(expected in doc_ids)
    return {
        "top_k": top_k,
        "min_score": min_score,
        "hit_rate": round(hits / total, 4),
        "hits": hits,
        "total": total,
    }


def grid_search(docs_dir: Path) -> dict:
    chunks = load_chunks(docs_dir)
    retriever = Retriever(chunks)
    grid = []
    for top_k in (1, 2, 3, 5):
        for min_score in (0.05, 0.10, 0.12, 0.18, 0.25):
            grid.append(evaluate(retriever, top_k, min_score))
    best = max(grid, key=lambda x: (x["hit_rate"], -x["top_k"], x["min_score"]))
    return {"best": best, "grid": grid, "chunks": len(chunks)}


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_tune.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best = report["best"]
    lines = [
        "# Retrieval Tuner",
        "",
        f"best: top_k={best['top_k']} min_score={best['min_score']} hit_rate={best['hit_rate']:.0%}",
        "",
        "| top_k | min_score | hit_rate |",
        "|---:|---:|---:|",
    ]
    for row in sorted(report["grid"], key=lambda x: x["hit_rate"], reverse=True)[:10]:
        lines.append(f"| {row['top_k']} | {row['min_score']} | {row['hit_rate']:.0%} |")
    (out_dir / "retrieval_tune.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
