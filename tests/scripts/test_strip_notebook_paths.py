"""Tests for scripts/strip_notebook_paths.py."""

import json

from scripts.strip_notebook_paths import (
    dump_notebook,
    find_local_paths,
    main,
    redact_notebook,
)

WARNING_WITH_PATH = (
    "C:\\Users\\someone\\AppData\\Local\\Temp\\ipykernel_4172\\329.py:12: "
    "CollinearityWarning: Cointegration test is not reliable in this case.\n"
)


def _stream(name, *lines):
    return {"name": name, "output_type": "stream", "text": list(lines)}


def _notebook(*outputs):
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "c1",
                "metadata": {},
                "outputs": list(outputs),
                "source": ["x = 1\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _texts(nb):
    return [o["text"] for o in nb["cells"][0]["outputs"]]


def test_redact_keeps_warning_text_and_drops_only_the_directories():
    nb = _notebook(_stream("stderr", WARNING_WITH_PATH, "  t, p, cv = coint(a, b)\n"))

    changed = redact_notebook(nb)

    assert changed == 1
    assert _texts(nb) == [
        [
            "329.py:12: CollinearityWarning: Cointegration test is not reliable "
            "in this case.\n",
            "  t, p, cv = coint(a, b)\n",
        ]
    ]
    assert find_local_paths(nb) == 0


def test_redact_handles_venv_and_posix_paths():
    nb = _notebook(
        _stream(
            "stderr",
            "D:\\proj\\.venv\\Lib\\site-packages\\pymc\\x.py:5: UserWarning: w\n",
            "/home/someone/proj/y.py:7: FutureWarning: f\n",
        )
    )

    redact_notebook(nb)

    assert _texts(nb) == [["x.py:5: UserWarning: w\n", "y.py:7: FutureWarning: f\n"]]


def test_stdout_is_never_modified_but_still_reported():
    nb = _notebook(_stream("stdout", "loaded /home/someone/data.csv\n"))

    assert redact_notebook(nb) == 0
    assert _texts(nb) == [["loaded /home/someone/data.csv\n"]]
    assert find_local_paths(nb) == 1


def test_escaped_newline_after_colon_is_not_mistaken_for_a_drive_path():
    # "Name:\n" serialised to JSON reads as `e:\n` — must not count as a path.
    nb = _notebook(_stream("stderr", "Name:\n", "value\tother\n"))

    assert redact_notebook(nb) == 0
    assert find_local_paths(nb) == 0


def test_dump_matches_jupyter_format():
    nb = _notebook(_stream("stdout", "zażółć\n"))

    out = dump_notebook(nb)

    assert out.endswith("}\n")
    assert '\n "cells": [' in out  # indent=1
    assert "zażółć" in out  # non-ASCII kept, not \u-escaped
    assert json.loads(out) == nb


def test_main_redacts_dirty_notebook_and_check_passes_afterwards(tmp_path):
    path = tmp_path / "nb.ipynb"
    path.write_text(
        dump_notebook(_notebook(_stream("stderr", WARNING_WITH_PATH))), encoding="utf-8"
    )

    assert main(["--check", str(tmp_path)]) == 1
    assert main([str(tmp_path)]) == 0
    assert main(["--check", str(tmp_path)]) == 0


def test_main_leaves_clean_notebook_byte_identical(tmp_path):
    path = tmp_path / "nb.ipynb"
    original = dump_notebook(_notebook(_stream("stdout", "ok\n")))
    path.write_text(original, encoding="utf-8", newline="\n")

    main([str(tmp_path)])

    assert path.read_text(encoding="utf-8") == original
