from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator
from src.evaluation.benchmark_generator import generate_benchmark_file

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GENERATED_CASE_FILE = EVALUATION_DIR / "generated_csv_benchmarks.json"
GENERATED_USER_CASE_FILE = EVALUATION_DIR / "generated_user_csv_benchmarks.json"
COMPARISON_CASE_FILE = EVALUATION_DIR / "sample_comparison_cases.json"
MULTILINGUAL_CASE_FILE = EVALUATION_DIR / "multilingual_financial_cases.json"
DEFAULT_CASE_FILES = [
    EVALUATION_DIR / "benchmark_queries.json",
    EVALUATION_DIR / "sample_csv_cases.json",
    EVALUATION_DIR / "sample_rag_cases.json",
    EVALUATION_DIR / "sample_tool_selection_cases.json",
    GENERATED_CASE_FILE,
    GENERATED_USER_CASE_FILE,
    COMPARISON_CASE_FILE,
    MULTILINGUAL_CASE_FILE,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible financial chatbot benchmark suite.")
    parser.add_argument("--check-only", action="store_true", help="Do not rewrite committed report files.")
    parser.add_argument("--include-uploaded", action="store_true", help="Regenerate CSV cases from root upload fixtures.")
    parser.add_argument("--minimum-accuracy", type=float, default=100.0)
    args = parser.parse_args()

    case_files = list(DEFAULT_CASE_FILES)
    if args.include_uploaded:
        upload_csv_paths = sorted((PROJECT_ROOT / "data" / "uploads").glob("*.csv"))
    else:
        upload_csv_paths = []
    if upload_csv_paths:
        generate_benchmark_file(upload_csv_paths, GENERATED_CASE_FILE)

    missing_case_files = [str(path) for path in case_files if not path.exists()]
    if missing_case_files:
        raise SystemExit("Missing committed benchmark files: {0}".format(", ".join(missing_case_files)))

    evaluator = Evaluator()
    report = evaluator.evaluate_files(case_files)

    json_path = EVALUATION_DIR / "evaluation_report.json"
    markdown_path = PROJECT_ROOT / "docs" / "evaluation_report.md"
    if not args.check_only:
        json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
        markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    print("Evaluation complete")
    print("Total cases: {0}".format(report.total_cases))
    print("Passed cases: {0}".format(report.passed_cases))
    print("Accuracy: {0:.2f}%".format(report.accuracy_percentage))
    if report.failed_cases:
        print("Failed cases:")
        for item in report.failed_cases:
            print("- {0}: {1}".format(item["case_id"], "; ".join(item["failures"])))
    if not args.check_only:
        print("JSON report: {0}".format(json_path))
        print("Markdown report: {0}".format(markdown_path))
    if report.accuracy_percentage < args.minimum_accuracy or report.failed_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
