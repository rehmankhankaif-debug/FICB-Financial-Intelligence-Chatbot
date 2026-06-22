from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator
from src.evaluation.benchmark_generator import generate_benchmark_file

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GENERATED_CASE_FILE = EVALUATION_DIR / "generated_csv_benchmarks.json"
DEFAULT_CASE_FILES = [
    EVALUATION_DIR / "benchmark_queries.json",
    EVALUATION_DIR / "sample_csv_cases.json",
    EVALUATION_DIR / "sample_rag_cases.json",
    EVALUATION_DIR / "sample_tool_selection_cases.json",
]


def main() -> None:
    upload_csv_paths = sorted((PROJECT_ROOT / "data" / "uploads").glob("*.csv"))
    case_files = list(DEFAULT_CASE_FILES)
    if upload_csv_paths:
        generate_benchmark_file(upload_csv_paths, GENERATED_CASE_FILE)
        case_files.append(GENERATED_CASE_FILE)

    evaluator = Evaluator()
    report = evaluator.evaluate_files(case_files)

    json_path = EVALUATION_DIR / "evaluation_report.json"
    markdown_path = PROJECT_ROOT / "docs" / "evaluation_report.md"
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
    print("JSON report: {0}".format(json_path))
    print("Markdown report: {0}".format(markdown_path))


if __name__ == "__main__":
    main()
