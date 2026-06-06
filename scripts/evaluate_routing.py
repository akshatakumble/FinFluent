"""Evaluate supervisor LLM routing accuracy against a labeled query set."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller.central_controller import route_user_query  # noqa: E402


def load_eval_set(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((row["query"].strip(), row["expected"].strip().lower()))
    return rows


def evaluate(path: Path) -> int:
    cases = load_eval_set(path)
    correct = 0
    print(f"Routing evaluation — {len(cases)} queries\n")
    print(f"{'Query':<45} {'Expected':<10} {'Predicted':<10} OK")
    print("-" * 80)

    for query, expected in cases:
        predicted = route_user_query(query)
        ok = predicted == expected
        correct += int(ok)
        mark = "yes" if ok else "NO"
        short_query = (query[:42] + "...") if len(query) > 45 else query
        print(f"{short_query:<45} {expected:<10} {predicted:<10} {mark}")

    accuracy = correct / len(cases) if cases else 0.0
    print("-" * 80)
    print(f"Accuracy: {correct}/{len(cases)} ({accuracy:.1%})")
    return 0 if accuracy >= 0.8 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinFluent intent routing.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data" / "routing_eval.csv"),
        help="CSV with columns: query, expected",
    )
    args = parser.parse_args()
    raise SystemExit(evaluate(Path(args.dataset)))
