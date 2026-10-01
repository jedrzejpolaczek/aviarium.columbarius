"""Tests for scripts/check_doc_paths.py."""

from scripts.check_doc_paths import HISTORICAL_MARKER, RepoIndex, check_doc

INDEX = RepoIndex(
    {
        "README.md",
        "docs/adr/ADR-001-x.md",
        "src/ml/training/trainer.py",
        "src/monitoring/retraining.py",
        "scripts/run_pipeline.py",
    }
)


def _never_ignored(_path: str) -> bool:
    return False


def _refs(doc, text, is_ignored=_never_ignored):
    return [b.reference for b in check_doc(doc, text, INDEX, is_ignored)]


def test_existing_files_directories_and_line_suffixes_pass():
    text = (
        "See `src/ml/training/trainer.py`, `src/ml/training/trainer.py:252`,\n"
        "`src/ml/` and `scripts/run_pipeline.py#main`.\n"
    )

    assert _refs("README.md", text) == []


def test_moved_file_and_missing_directory_are_reported():
    text = "Trainer lives in `src/ml/trainer.py`; metrics in `src/ml/metrics/`.\n"

    assert _refs("README.md", text) == ["src/ml/trainer.py", "src/ml/metrics/"]


def test_bare_file_name_must_exist_somewhere():
    text = "`trainer.py` exists, `allegro.py` never did.\n"

    assert _refs("README.md", text) == ["allegro.py"]


def test_external_download_names_are_allowed():
    assert _refs("README.md", "Reads `AllPricesToday.json`.\n") == []


def test_module_attribute_reference_resolves_to_its_module():
    text = "`src/monitoring/retraining._compare_and_promote` gates promotion.\n"

    assert _refs("README.md", text) == []


def test_relative_links_resolve_against_the_document_directory():
    text = (
        "[ok](../../README.md) [bad](ADR-999-gone.md) [anchor](#x) [web](https://x.y)\n"
    )

    assert _refs("docs/adr/ADR-001-x.md", text) == ["link ADR-999-gone.md"]


def test_gitignored_paths_are_files_the_reader_creates():
    text = "Copy it to `docker/.env`.\n"

    assert _refs("README.md", text, is_ignored=lambda p: p == "docker/.env") == []


def test_historical_marker_skips_the_line():
    text = f"Migrated by `scripts/migrate_old.py` (removed). {HISTORICAL_MARKER}\n"

    assert _refs("README.md", text) == []


def test_non_path_backticks_are_ignored():
    text = "Run `make check`, set `flag: true`, call `retrain()`.\n"

    assert _refs("README.md", text) == []


def test_reports_carry_the_line_number():
    text = "fine\n`src/nope.py`\n"

    broken = check_doc("README.md", text, INDEX, _never_ignored)

    assert [(b.line, str(b)) for b in broken] == [(2, "README.md:2: src/nope.py")]
