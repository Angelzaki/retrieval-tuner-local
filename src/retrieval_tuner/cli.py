from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from retrieval_tuner.tuner import grid_search, write_report

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval-tuner")
    parser.add_argument("--docs", default="samples/docs")
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)

    report = grid_search(Path(args.docs))
    write_report(report, Path(args.results))

    table = Table(title="Retrieval Tuner · top 5")
    table.add_column("top_k", justify="right")
    table.add_column("min_score", justify="right")
    table.add_column("hit_rate", justify="right")
    for row in sorted(report["grid"], key=lambda x: x["hit_rate"], reverse=True)[:5]:
        table.add_row(str(row["top_k"]), str(row["min_score"]), f"{row['hit_rate']:.0%}")
    console.print(table)
    best = report["best"]
    console.print(
        f"best: top_k=[bold]{best['top_k']}[/bold] min_score=[bold]{best['min_score']}[/bold] "
        f"({best['hit_rate']:.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
