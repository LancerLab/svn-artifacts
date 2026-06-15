#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = WORKSPACE_ROOT / "benchmark" / "choreo"
TRITON_ROOT = WORKSPACE_ROOT / "benchmark" / "triton"
GENERATED_CASE = Path("benchmark/triton/cases/generated_case.py")
GENERATED_CASE_ROOT = TRITON_ROOT

sys.path.insert(0, str((TRITON_ROOT / "cases").resolve()))

from triton_case_common import describe_case, parse_case  # noqa: E402
from binary_family import BINARY_RUNTIME_ASSERTIONS  # noqa: E402
from unary_family import UNARY_RUNTIME_ASSERTIONS  # noqa: E402


IMPLEMENTED_UNARY = {"relu", "sigmoid", "gelu"}
IMPLEMENTED_BINARY = {"elemwise_add"}


def main() -> int:
    manifest_path = TRITON_ROOT / "manifest.csv"
    status_path = TRITON_ROOT / "status.csv"
    choreo_cases = sorted(path for path in CASE_ROOT.rglob("*.co") if path.stem[:1].isdigit())

    manifest_rows = []
    status_rows = []

    for case_path in choreo_cases:
        relative_case = case_path.relative_to(WORKSPACE_ROOT).as_posix()
        parsed = parse_case(case_path)
        metadata = describe_case(case_path)
        triton_case, runner_args, implementation_status, validation_status, explicit_assertions, generated_note = generate_case_runner(parsed, relative_case, metadata)

        manifest_rows.append(
            {
                "category": parsed.category,
                "case_name": parsed.case_name,
                "choreo_case": relative_case,
                "triton_case": triton_case,
                "expected": "success",
                "runner_args": runner_args,
                "notes": metadata["notes"] or generated_note,
            }
        )
        status_rows.append(
            {
                "category": parsed.category,
                "case_name": parsed.case_name,
                "choreo_case": relative_case,
                "triton_case": triton_case,
                "operator_view_equivalent": "yes" if metadata["valid_contract"] else "no",
                "input_shape_contract": metadata["contract"],
                "explicit_assertion_count": explicit_assertions,
                "implementation_status": implementation_status if metadata["valid_contract"] else "generated-contract-mismatch",
                "validation_status": validation_status if metadata["valid_contract"] else "excluded-invalid-contract",
                "notes": metadata["notes"] or generated_note,
            }
        )

    write_csv(
        manifest_path,
        [
            "category",
            "case_name",
            "choreo_case",
            "triton_case",
            "expected",
            "runner_args",
            "notes",
        ],
        manifest_rows,
    )
    write_csv(
        status_path,
        [
            "category",
            "case_name",
            "choreo_case",
            "triton_case",
            "operator_view_equivalent",
            "input_shape_contract",
            "explicit_assertion_count",
            "implementation_status",
            "validation_status",
            "notes",
        ],
        status_rows,
    )

    print(f"Wrote {manifest_path}")
    print(f"Wrote {status_path}")
    print(f"Covered {len(choreo_cases)} benchmark/choreo cases")
    return 0


def generate_case_runner(parsed, relative_case: str, metadata: dict[str, object]) -> tuple[str, str, str, str, int, str]:
    if parsed.category in IMPLEMENTED_UNARY and metadata["valid_contract"]:
        target = GENERATED_CASE_ROOT / parsed.category / f"{parsed.case_name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_unary_entrypoint(relative_case, parsed.category), encoding="utf-8")
        target.chmod(0o755)
        return (
            target.relative_to(WORKSPACE_ROOT).as_posix(),
            "--device cuda",
            "implemented-generated-unary-kernel",
            "compile-pending",
            int(metadata["explicit_assertions"]) + UNARY_RUNTIME_ASSERTIONS,
            "Generated Triton unary kernel entrypoint",
        )

    if parsed.category in IMPLEMENTED_BINARY and metadata["valid_contract"]:
        target = GENERATED_CASE_ROOT / parsed.category / f"{parsed.case_name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_binary_entrypoint(relative_case), encoding="utf-8")
        target.chmod(0o755)
        return (
            target.relative_to(WORKSPACE_ROOT).as_posix(),
            "--device cuda",
            "implemented-generated-binary-kernel",
            "compile-pending",
            int(metadata["explicit_assertions"]) + BINARY_RUNTIME_ASSERTIONS,
            "Generated Triton binary add entrypoint",
        )

    if metadata["valid_contract"]:
        target = GENERATED_CASE_ROOT / parsed.category / f"{parsed.case_name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_host_entrypoint(relative_case), encoding="utf-8")
        target.chmod(0o755)
        return (
            target.relative_to(WORKSPACE_ROOT).as_posix(),
            "--device cpu",
            "implemented-generated-host-entrypoint",
            "generated-unvalidated",
            int(metadata["explicit_assertions"]),
            "Generated concrete host-backed entrypoint",
        )

    target = GENERATED_CASE_ROOT / parsed.category / f"{parsed.case_name}.py"
    return (
        target.relative_to(WORKSPACE_ROOT).as_posix(),
        "--device cpu",
        "implemented-standalone",
        "generated-unvalidated",
        int(metadata["explicit_assertions"]),
        "Standalone host-backed Triton case (contract-invalid in parser)",
    )


def render_unary_entrypoint(relative_case: str, category: str) -> str:
    return f'''#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parents[2]
if str(CASES_DIR) not in sys.path:
    sys.path.insert(0, str(CASES_DIR))

from unary_family import bound_case_main


if __name__ == "__main__":
    sys.exit(bound_case_main("{relative_case}", "{category}"))
'''


def render_binary_entrypoint(relative_case: str) -> str:
    return f'''#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parents[2]
if str(CASES_DIR) not in sys.path:
    sys.path.insert(0, str(CASES_DIR))

from binary_family import bound_case_main


if __name__ == "__main__":
    sys.exit(bound_case_main("{relative_case}"))
'''


def render_host_entrypoint(relative_case: str) -> str:
    return f'''#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parents[2]
if str(CASES_DIR) not in sys.path:
    sys.path.insert(0, str(CASES_DIR))

from generated_case import bound_case_main


if __name__ == "__main__":
    sys.exit(bound_case_main("{relative_case}"))
'''


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())