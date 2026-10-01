"""Redact local absolute paths leaked into committed notebook outputs.

Python warnings print the full path of the file that raised them
(``C:\\Users\\<name>\\AppData\\Local\\Temp\\ipykernel_4172\\329.py:82: ...``), so
every executed notebook that hit a warning publishes the author's username and
directory layout. The outputs themselves are the evidence behind the FINDINGS
documents, and some of those warnings carry analysis content (e.g. statsmodels'
CollinearityWarning "Cointegration test is not reliable"), so neither clearing
outputs (``nbstripout``) nor deleting the warnings is acceptable.

Instead the directory part of each path is cut out of ``stderr`` stream
outputs, leaving ``329.py:82: FutureWarning: ...``. Stdout and display outputs
are never modified; ``--check`` still reports paths found there.

Files are rewritten the way Jupyter writes them (indent=1, sorted keys,
non-ASCII kept), so the diff contains only the redacted lines.

Usage:
    python -m scripts.strip_notebook_paths            # redact in place
    python -m scripts.strip_notebook_paths --check    # exit 1 if any path remains (CI)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

NOTEBOOKS_DIR = Path("notebooks")

LOCAL_DIR_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:\\(?:[^\\\s]+\\)+|/(?:home|Users)/(?:[^/\s]+/)+"
)
"""The directory part of a Windows drive path or a POSIX home path.

Matched against decoded output text, never against serialised JSON: there
``"Name:\\n"`` would read as drive ``e:`` followed by a backslash."""


def _output_text(value: Any) -> str:
    """Concatenate every string inside an output (text, data, traceback...)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_output_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_output_text(v) for v in value)
    return ""


def redact_notebook(nb: dict[str, Any]) -> int:
    """Cut local directories out of stderr stream outputs of *nb*, in place.

    Returns the number of lines changed.
    """
    changed = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stderr":
                continue
            text = output.get("text", [])
            lines = [text] if isinstance(text, str) else text
            redacted = [LOCAL_DIR_RE.sub("", line) for line in lines]
            changed += sum(a != b for a, b in zip(lines, redacted))
            output["text"] = redacted[0] if isinstance(text, str) else redacted
    return changed


def find_local_paths(nb: dict[str, Any]) -> int:
    """Count outputs of any type that still contain a local absolute path."""
    return sum(
        1
        for cell in nb.get("cells", [])
        for output in cell.get("outputs", [])
        if LOCAL_DIR_RE.search(_output_text(output))
    )


def dump_notebook(nb: dict[str, Any]) -> str:
    """Serialise *nb* exactly as Jupyter does."""
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks containing local paths and exit 1; modify nothing.",
    )
    parser.add_argument("root", nargs="?", type=Path, default=NOTEBOOKS_DIR)
    args = parser.parse_args(argv if argv is not None else [])

    dirty = 0
    for path in sorted(args.root.rglob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        if args.check:
            hits = find_local_paths(nb)
            if hits:
                print(f"{path}: {hits} output(s) contain a local path")
                dirty += 1
            continue
        changed = redact_notebook(nb)
        if changed:
            path.write_text(dump_notebook(nb), encoding="utf-8", newline="\n")
            print(f"{path}: redacted {changed} line(s)")
        remaining = find_local_paths(nb)
        if remaining:
            print(
                f"{path}: {remaining} non-stderr output(s) still contain a path - fix by hand"
            )

    if args.check and dirty:
        print(
            f"{dirty} notebook(s) leak local paths - run "
            "`python -m scripts.strip_notebook_paths`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
