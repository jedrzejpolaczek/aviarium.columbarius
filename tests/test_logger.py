"""Unit tests for src/logger.py.

Tests cover PlainFormatter, ColorFormatter, ProgressStreamHandler, ProgressLogger,
and setup_logging.
"""

import io
import logging
import pathlib
from collections.abc import Generator

import pytest

from src.logger import (
    PROGRESS,
    ColorFormatter,
    PlainFormatter,
    ProgressLogger,
    ProgressStreamHandler,
    get_logger,
    setup_logging,
)


@pytest.fixture()
def isolated_root_logger() -> Generator[logging.Logger, None, None]:
    """Save and restore root logger handlers/level around each test.

    setup_logging() clears root handlers to avoid duplicates; this fixture
    ensures pytest's log-capture handler is reinstated after each test.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# PlainFormatter
# ---------------------------------------------------------------------------


class TestPlainFormatter:
    def test_format_returns_string(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        assert isinstance(formatter.format(record), str)

    def test_format_contains_message(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "my message", (), None)
        assert "my message" in formatter.format(record)

    def test_format_contains_no_ansi_codes(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert "\033[" not in formatter.format(record)

    def test_format_contains_level_name(self) -> None:
        formatter = PlainFormatter()
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert "WARNING" in formatter.format(record)


# ---------------------------------------------------------------------------
# ColorFormatter
# ---------------------------------------------------------------------------


class TestColorFormatter:
    def test_format_returns_string(self) -> None:
        formatter = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        result = formatter.format(record)
        assert isinstance(result, str)

    def test_format_contains_message(self) -> None:
        formatter = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "my message", (), None)
        result = formatter.format(record)
        assert "my message" in result

    def test_format_contains_ansi_codes(self) -> None:
        formatter = ColorFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = formatter.format(record)
        assert "\033[" in result

    def test_format_unknown_level_uses_reset(self) -> None:
        formatter = ColorFormatter()
        record = logging.LogRecord("test", 999, "", 0, "msg", (), None)
        result = formatter.format(record)
        assert isinstance(result, str)

    def test_format_progress_level(self) -> None:
        formatter = ColorFormatter()
        record = logging.LogRecord("test", PROGRESS, "", 0, "progress msg", (), None)
        result = formatter.format(record)
        assert "progress msg" in result


# ---------------------------------------------------------------------------
# ProgressStreamHandler
# ---------------------------------------------------------------------------


class TestProgressStreamHandler:
    def test_default_stream_is_stderr(self) -> None:
        import sys

        handler = ProgressStreamHandler()
        assert handler.stream is sys.stderr

    def test_custom_stream_accepted(self) -> None:
        stream = io.StringIO()
        handler = ProgressStreamHandler(stream)
        assert handler.stream is stream

    def test_has_progress_initially_false(self) -> None:
        handler = ProgressStreamHandler(io.StringIO())
        assert handler._has_progress is False

    def test_emit_info_in_non_tty_writes_line(self) -> None:
        stream = io.StringIO()
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(ColorFormatter())
        record = logging.LogRecord("test", logging.INFO, "", 0, "info line", (), None)
        handler.emit(record)
        assert "info line" in stream.getvalue()

    def test_emit_progress_in_non_tty_writes_line(self) -> None:
        stream = io.StringIO()
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(ColorFormatter())
        record = logging.LogRecord("test", PROGRESS, "", 0, "progress line", (), None)
        handler.emit(record)
        assert "progress line" in stream.getvalue()

    def test_emit_progress_in_tty_uses_carriage_return(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True  # type: ignore[method-assign]
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(ColorFormatter())
        record = logging.LogRecord("test", PROGRESS, "", 0, "tty progress", (), None)
        handler.emit(record)
        output = stream.getvalue()
        assert "\r" in output

    def test_emit_sets_has_progress_flag_in_tty(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True  # type: ignore[method-assign]
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(ColorFormatter())
        record = logging.LogRecord("test", PROGRESS, "", 0, "p", (), None)
        handler.emit(record)
        assert handler._has_progress is True

    def test_emit_info_clears_progress_line_in_tty(self) -> None:
        """INFO after PROGRESS in TTY should clear the progress line first."""
        stream = io.StringIO()
        stream.isatty = lambda: True  # type: ignore[method-assign]
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(ColorFormatter())
        handler._has_progress = True

        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "info after progress", (), None
        )
        handler.emit(record)

        output = stream.getvalue()
        assert "\r" in output  # _CLEAR contains \r
        assert handler._has_progress is False


# ---------------------------------------------------------------------------
# ProgressLogger
# ---------------------------------------------------------------------------


class TestProgressLogger:
    def test_get_logger_returns_progress_logger(self) -> None:
        logger = get_logger("test.progress")
        assert isinstance(logger, ProgressLogger)

    def test_progress_method_exists(self) -> None:
        logger = get_logger("test.progress2")
        assert hasattr(logger, "progress")
        assert callable(logger.progress)

    def test_progress_method_logs_at_progress_level(self) -> None:
        stream = io.StringIO()
        handler = ProgressStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger = get_logger("test.level_check")
        logger.setLevel(PROGRESS)
        logger.addHandler(handler)
        logger.propagate = False
        try:
            logger.progress("test message")
            assert "test message" in stream.getvalue()
        finally:
            logger.removeHandler(handler)

    def test_progress_method_noop_when_level_above_progress(self) -> None:
        stream = io.StringIO()
        handler = ProgressStreamHandler(stream)
        logger = get_logger("test.noop")
        logger.setLevel(
            logging.INFO
        )  # INFO=20 > PROGRESS=15, so progress is suppressed
        logger.addHandler(handler)
        logger.propagate = False
        try:
            logger.progress("should not appear")
            assert "should not appear" not in stream.getvalue()
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_adds_progress_stream_handler(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        setup_logging()
        assert any(
            isinstance(h, ProgressStreamHandler) for h in isolated_root_logger.handlers
        )

    def test_console_handler_level_is_progress_by_default(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        setup_logging()
        handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, ProgressStreamHandler)
        ]
        assert handlers[0].level == PROGRESS

    def test_custom_level_applied_to_console_handler(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        setup_logging(level=logging.WARNING)
        handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, ProgressStreamHandler)
        ]
        assert handlers[0].level == logging.WARNING

    def test_returns_none_without_log_dir(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        assert setup_logging() is None

    def test_creates_log_file_when_log_dir_provided(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        result = setup_logging(log_dir=tmp_path)
        assert result is not None
        assert result.exists()
        assert result.parent == tmp_path
        assert result.suffix == ".log"

    def test_file_handler_added_at_debug_level(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        setup_logging(log_dir=tmp_path)
        file_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

    def test_file_handler_uses_plain_formatter(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        setup_logging(log_dir=tmp_path)
        file_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert isinstance(file_handlers[0].formatter, PlainFormatter)

    def test_file_handler_is_rotating_with_expected_limits(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        import logging.handlers

        setup_logging(log_dir=tmp_path)
        file_handlers = [
            h
            for h in isolated_root_logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 10_000_000
        assert file_handlers[0].backupCount == 5

    def test_prunes_old_log_files_beyond_keep_last(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        for i in range(12):
            (tmp_path / f"pipeline_2026-01-{i + 1:02d}_00-00-00.log").write_text("x")

        setup_logging(log_dir=tmp_path, keep_last_logs=10)

        remaining = sorted(tmp_path.glob("pipeline_*.log"))
        # 12 pre-existing + 1 new file created by this call = 13; pruned to 10.
        assert len(remaining) == 10

    def test_keeps_all_log_files_when_under_the_limit(
        self, isolated_root_logger: logging.Logger, tmp_path: pathlib.Path
    ) -> None:
        (tmp_path / "pipeline_2026-01-01_00-00-00.log").write_text("x")

        setup_logging(log_dir=tmp_path, keep_last_logs=10)

        remaining = list(tmp_path.glob("pipeline_*.log"))
        assert len(remaining) == 2  # the pre-existing file + the new one

    def test_root_level_set_to_debug(
        self, isolated_root_logger: logging.Logger
    ) -> None:
        setup_logging()
        assert isolated_root_logger.level == logging.DEBUG
