"""Fail when a document cites a repository file or directory that does not exist.

Documentation here drifts the same way every time: a file is moved, split or
deleted, and the ADRs, C4 pages and README that cite it keep pointing at the
old location. The 2026-09 audit found 20+ such references (``src/ml/trainer.py``,
``storage/base.py``, ``allegro.py``...), each of which sends a reader to a file
that is not there. This check makes that class of drift fail CI instead of
waiting for the next audit.

What is checked, in every tracked ``*.md`` except CHANGELOG.md (a history,
where old paths are correct by definition):

- backtick spans that start with a top-level project directory
  (``src/``, ``app/``, ``docs/``...) — must be a tracked file or directory;
- backtick spans that are a bare file name with a code/config extension
  (``registry.py``) — some tracked file must have that name;
- relative Markdown link targets — must resolve to a tracked file or directory.

Deliberate exceptions:

- paths ignored by ``.gitignore`` (``docker/.env``) are files the docs tell the
  reader to create, so they are not expected to exist;
- names in :data:`EXTERNAL_NAMES` are remote files, not repository files;
- a line ending in ``<!-- doc-paths: historical -->`` describes something that
  was removed on purpose and is skipped.

Usage:
    python -m scripts.check_doc_paths
"""

import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

PROJECT_DIRS = (
    ".github/",
    "app/",
    "configs/",
    "docker/",
    "docs/",
    "frontend/",
    "notebooks/",
    "scripts/",
    "src/",
    "tests/",
)

EXTERNAL_NAMES = frozenset(
    {
        # MTGJson bulk downloads, cited as data sources.
        "AllPrintings.json",
        "AllPrices.json",
        "AllPricesToday.json",
    }
)

HISTORICAL_MARKER = "<!-- doc-paths: historical -->"

EXCLUDED_DOCS = frozenset({"CHANGELOG.md"})

_BACKTICK_RE = re.compile(r"`([^`\s]+)`")
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
_BARE_NAME_RE = re.compile(r"[\w.-]+\.(?:py|sql|json|ya?ml|ts|tsx|ipynb)")


@dataclass(frozen=True)
class Broken:
    doc: str
    line: int
    reference: str

    def __str__(self) -> str:
        return f"{self.doc}:{self.line}: {self.reference}"


def _git_ls_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return {line for line in out.splitlines() if line}


def _is_gitignored(path: str) -> bool:
    # No trailing slash: for a path that does not exist, `git check-ignore`
    # reports `dir/` as ignored by a blank .gitignore line.
    path = path.rstrip("/")
    return (
        subprocess.run(["git", "check-ignore", "-q", path], check=False).returncode == 0
    )


class RepoIndex:
    """Tracked files, the directories that contain them, and their base names."""

    def __init__(self, files: set[str]) -> None:
        self.files = files
        self.dirs: set[str] = set()
        for f in files:
            parts = f.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                self.dirs.add("/".join(parts[:i]))
        self.names = {f.rsplit("/", 1)[-1] for f in files}

    def exists(self, path: str) -> bool:
        path = path.rstrip("/")
        if path in self.files or path in self.dirs:
            return True
        # `src/monitoring/retraining._compare_and_promote` — module.attribute
        module, _, attr = path.rpartition(".")
        return bool(attr) and f"{module}.py" in self.files


def _clean(token: str) -> str:
    """Strip `:line`, `#anchor`, `(args)` and trailing punctuation."""
    return re.split(r"[:#(]", token, maxsplit=1)[0].rstrip(".,;")


def check_doc(
    doc: str,
    text: str,
    index: RepoIndex,
    is_ignored: Callable[[str], bool] = _is_gitignored,
) -> list[Broken]:
    broken: list[Broken] = []
    doc_dir = str(PurePosixPath(doc).parent)
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.rstrip().endswith(HISTORICAL_MARKER):
            continue
        for token in _BACKTICK_RE.findall(line):
            ref = _clean(token)
            if not ref or any(ch in ref for ch in "*{}<>$"):
                continue
            if "/" not in ref:
                if (
                    _BARE_NAME_RE.fullmatch(ref)
                    and ref not in index.names
                    and ref not in EXTERNAL_NAMES
                ):
                    broken.append(Broken(doc, lineno, ref))
                continue
            if ref.startswith(PROJECT_DIRS) and not index.exists(ref):
                if not is_ignored(ref):
                    broken.append(Broken(doc, lineno, ref))
        for target in _LINK_RE.findall(line):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = os.path.normpath(os.path.join(doc_dir, rel)).replace(os.sep, "/")
            if not index.exists(resolved) and not is_ignored(resolved):
                broken.append(Broken(doc, lineno, f"link {target}"))
    return broken


def main() -> int:
    index = RepoIndex(_git_ls_files())
    docs = sorted(
        f for f in index.files if f.endswith(".md") and f not in EXCLUDED_DOCS
    )
    broken: list[Broken] = []
    for doc in docs:
        with open(doc, encoding="utf-8") as fh:
            broken.extend(check_doc(doc, fh.read(), index))
    for b in broken:
        print(b)
    print(f"Checked {len(docs)} documents: {len(broken)} broken reference(s).")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
