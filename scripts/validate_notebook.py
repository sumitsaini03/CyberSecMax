#!/usr/bin/env python3
"""Validate notebook JSON structure and the required Colab workflow sections."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_TEXT = (
    "Verify the actual runtime GPU",
    "Install the pinned candidate stack",
    "Mount persistent Google Drive",
    "Hugging Face authentication and terms",
    "Inspect the datasets before choosing a profile",
    "Baseline evaluation and preflight smoke test",
    "Full QLoRA training",
    "Inspect post-training results",
    "Troubleshooting",
)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Invalid notebook JSON: {exc}"]
    if notebook.get("nbformat") != 4:
        errors.append("Notebook must use nbformat 4.")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        return errors + ["Notebook must contain a non-empty cells list."]
    sources: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in {"markdown", "code"}:
            errors.append(f"Cell {index} has unsupported cell_type.")
        source = cell.get("source")
        if isinstance(source, list):
            if not all(isinstance(line, str) for line in source):
                errors.append(f"Cell {index} source list must contain only strings.")
            content = "".join(source)
        elif isinstance(source, str):
            content = source
        else:
            errors.append(f"Cell {index} is missing string source.")
            content = ""
        sources.append(content)
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                errors.append(f"Code cell {index} must not contain an execution count.")
            if cell.get("outputs") != []:
                errors.append(f"Code cell {index} must not contain saved outputs.")
    all_text = "\n".join(sources)
    for required in REQUIRED_TEXT:
        if required not in all_text:
            errors.append(f"Notebook is missing required section: {required}")
    if ("train_cyber_lora.py" not in all_text or "--preflight-only" not in all_text
            or "--config configs/experiment.yaml" not in all_text):
        errors.append("Notebook must invoke the primary script for preflight and full training.")
    if "HF_TOKEN" not in all_text or "userdata.get" not in all_text:
        errors.append("Notebook must show Colab Secret guidance without embedding a token.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default=str(Path(__file__).resolve().parents[1] / "SecurityMaxx.ipynb"))
    args = parser.parse_args()
    errors = validate(Path(args.notebook))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {args.notebook} is valid nbformat 4 JSON and contains the required workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
