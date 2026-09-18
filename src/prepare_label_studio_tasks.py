"""Create Label Studio tasks from one plain-text prompt per line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT = Path("annotation/label-studio/tasks.json")


class TaskPreparationError(ValueError):
    """A prompt file cannot be converted into valid Label Studio tasks."""


def prepare_tasks(prompts: Iterable[str]) -> list[dict[str, dict[str, str]]]:
    """Build deterministic Label Studio task objects from prompt strings."""

    tasks: list[dict[str, dict[str, str]]] = []
    for line_no, prompt in enumerate(prompts, 1):
        if not isinstance(prompt, str):
            raise TaskPreparationError(f"prompt line {line_no}: expected text")
        text = prompt.strip()
        if text:
            tasks.append({"data": {"text": text}})
    if not tasks:
        raise TaskPreparationError("prompt input contains no non-empty prompts")
    return tasks


def seed_tasks(input_path: Path, output_path: Path) -> int:
    """Read prompts from ``input_path`` and write a Label Studio task array."""

    try:
        prompts = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TaskPreparationError(f"cannot read prompts from {input_path}: {exc}") from exc

    tasks = prepare_tasks(prompts)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise TaskPreparationError(f"cannot write tasks to {output_path}: {exc}") from exc
    return len(tasks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create Label Studio tasks from one prompt per line."
    )
    parser.add_argument("prompts", type=Path, help="UTF-8 text file with one prompt per line")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"task JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    try:
        count = seed_tasks(args.prompts, args.output)
    except TaskPreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {count} Label Studio tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
