"""Unit tests for scripts/kill_proc.py.

Only the non-interactive `report()` helper is tested here, against real
processes. main()'s interactive kill path is intentionally left untested
since exercising it would require actually killing a process.
"""

import os

from scripts.kill_proc import report


def test_report_returns_process_for_current_pid():
    proc = report(os.getpid())

    assert proc is not None
    assert proc.pid == os.getpid()


def test_report_returns_none_for_nonexistent_pid():
    proc = report(999999)

    assert proc is None
