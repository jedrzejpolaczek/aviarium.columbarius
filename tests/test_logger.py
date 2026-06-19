"""Unit tests for src/logger.py.

Tests cover ColorFormatter, ProgressStreamHandler, ProgressLogger, and setup_logging.
"""

import io
import logging


from src.logger import (
    PROGRESS,
    ColorFormatter,
    ProgressLogger,
    ProgressStreamHandler,
    get_logger,
    setup_logging,
)


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
    """setup_logging wraps basicConfig, which is a no-op when root already has handlers.

    We test the arguments passed to basicConfig rather than the root-logger state,
    because pytest's log-capture plugin always holds a handler on root during tests.
    """

    def test_setup_logging_passes_progress_handler_to_basicconfig(self) -> None:
        from unittest.mock import patch

        with patch("logging.basicConfig") as mock_bc:
            setup_logging()
        _, kwargs = mock_bc.call_args
        handlers = kwargs.get("handlers", [])
        assert any(isinstance(h, ProgressStreamHandler) for h in handlers)

    def test_setup_logging_default_level_is_progress(self) -> None:
        from unittest.mock import patch

        with patch("logging.basicConfig") as mock_bc:
            setup_logging()
        _, kwargs = mock_bc.call_args
        assert kwargs.get("level") == PROGRESS

    def test_setup_logging_custom_level_forwarded(self) -> None:
        from unittest.mock import patch

        with patch("logging.basicConfig") as mock_bc:
            setup_logging(level=logging.WARNING)
        _, kwargs = mock_bc.call_args
        assert kwargs.get("level") == logging.WARNING
