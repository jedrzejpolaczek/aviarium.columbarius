"""Logging configuration with a custom PROGRESS level and colour output.

Exposes:
    PROGRESS          — integer log level (15, between DEBUG and INFO) for
                        ephemeral in-place status lines.
    ProgressLogger    — logging.Logger subclass with a .progress() method.
    ProgressStreamHandler — StreamHandler that renders PROGRESS records as
                        overwriting terminal lines in a TTY; falls back to
                        plain lines when output is redirected.
    ColorFormatter    — Formatter that colourises level names via ANSI codes.
    get_logger(name)  — typed drop-in for logging.getLogger(name); returns a
                        ProgressLogger so .progress() is visible to mypy.
    setup_logging()   — configure root logging with colour output and PROGRESS
                        support (call once at application entry point).
"""

import logging
import logging.handlers
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

# ── Colour palette (ANSI codes) ───────────────────────────────────────────────
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREY = "\033[90m"
_BRED = "\033[1;31m"
_CLEAR = "\r\033[K"  # carriage-return + erase to end of line

# ── PROGRESS level (between DEBUG=10 and INFO=20) ─────────────────────────────
# Use for ephemeral status messages that overwrite each other on one terminal
# line (e.g. "Downloading deck 3/8 — Burn").  INFO and above are persistent.
PROGRESS = 15
logging.addLevelName(PROGRESS, "PROGRESS")

_LEVEL_COLOURS = {
    logging.DEBUG: _GREY,
    PROGRESS: _GREY,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BRED,
}


# ── Formatter ─────────────────────────────────────────────────────────────────


class PlainFormatter(logging.Formatter):
    """Plain-text formatter for file output — no ANSI codes."""

    _FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)


class ColorFormatter(logging.Formatter):
    """Formatter that colourises the level name and logger name via ANSI codes."""

    _FMT = "%(asctime)s {lvl}%(levelname)-8s{rst} {name}%(name)s{rst} — %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        """Pre-build per-level formatters so format() does not allocate on every call."""
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)
        self._formatters: dict[int, logging.Formatter] = {
            level: logging.Formatter(
                self._FMT.format(lvl=colour, rst=_RESET, name=_CYAN),
                datefmt=self._DATE_FMT,
            )
            for level, colour in _LEVEL_COLOURS.items()
        }
        self._default_formatter = logging.Formatter(
            self._FMT.format(lvl=_RESET, rst=_RESET, name=_CYAN),
            datefmt=self._DATE_FMT,
        )

    def format(self, record: logging.LogRecord) -> str:
        """Return the formatted record with ANSI colour injected for the log level."""
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)


# ── Handler ───────────────────────────────────────────────────────────────────


class ProgressStreamHandler(logging.StreamHandler[TextIO]):
    """StreamHandler that renders PROGRESS records as in-place overwriting lines.

    In a real TTY:
        PROGRESS messages overwrite the current line (no trailing newline, grey).
        Any higher-level record first clears the progress line, then prints
        normally with a newline.

    When output is redirected (pipes, CI logs, file capture):
        PROGRESS messages are printed as regular lines so the log is still
        informative; the overwrite behaviour is suppressed because \\r and
        \\033[K land in the file as raw bytes.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        """Initialise the handler; defaults to stderr when no stream is given."""
        super().__init__(stream or sys.stderr)
        self._has_progress = False

    def emit(self, record: logging.LogRecord) -> None:
        """Write *record*, overwriting the current line for PROGRESS records in a TTY."""
        try:
            msg = self.format(record)
            stream = self.stream
            is_progress = record.levelno == PROGRESS
            # Check if we straming to terminal
            is_teletypewriter = getattr(stream, "isatty", lambda: False)()

            if is_progress and is_teletypewriter:
                # Overwrite current line in grey — no trailing newline.
                stream.write(f"{_CLEAR}{_GREY}{msg}{_RESET}")
                stream.flush()
                self._has_progress = True
            else:
                if self._has_progress and is_teletypewriter:
                    # Clear the lingering progress line before the real message.
                    stream.write(_CLEAR)
                    self._has_progress = False
                stream.write(msg + self.terminator)
                stream.flush()
        except Exception:
            self.handleError(record)


# ── Logger subclass with .progress() ─────────────────────────────────────────


class ProgressLogger(logging.Logger):
    """logging.Logger subclass that exposes a .progress() convenience method."""

    def progress(self, message: object, *args: object, **kwargs: Any) -> None:
        """Log at PROGRESS level (ephemeral overwriting line)."""
        if self.isEnabledFor(PROGRESS):
            self._log(PROGRESS, message, args, **kwargs)


logging.setLoggerClass(ProgressLogger)


def get_logger(name: str) -> ProgressLogger:
    """Return a ProgressLogger for *name*.

    Drop-in replacement for ``logging.getLogger(name)`` that gives mypy a
    typed handle with the .progress() method.
    """
    return logging.getLogger(name)  # type: ignore[return-value]


# ── Public setup ──────────────────────────────────────────────────────────────


_LOG_FILE_NAME_RE = re.compile(
    r"^pipeline_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.log(\.\d+)?$"
)


def _prune_old_log_files(log_dir: Path, keep_last: int) -> None:
    """Delete all but the *keep_last* most recent timestamped log file groups.

    Mirrors scripts/backup_data.py's `_prune_old_snapshots` pattern: only
    files matching the `pipeline_<timestamp>.log` name (optionally followed
    by RotatingFileHandler's own `.1`, `.2`, ... backup suffix) are
    considered prunable, so an unrelated file placed in `log_dir` is never
    touched. Files are grouped by their `<timestamp>` (captured by the first
    regex group) rather than pruned individually, so a run whose single
    invocation grows past `maxBytes` and produces `pipeline_<ts>.log.1`,
    `.log.2`, etc. has all of those siblings pruned together with their
    base file — otherwise the `.log.N` siblings would become orphaned
    (their base file gone, but the suffixed files themselves untouched
    forever) once the base file aged out of `keep_last`.

    Grouping by timestamp is needed in addition to RotatingFileHandler's
    own maxBytes/backupCount rotation, since setup_logging() creates a
    *new* timestamped file (and thus a new group) on every call rather
    than appending to one rotating file across calls.
    """
    if keep_last <= 0:
        return
    groups: dict[str, list[Path]] = {}
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        match = _LOG_FILE_NAME_RE.match(p.name)
        if match:
            groups.setdefault(match.group(1), []).append(p)

    for timestamp in sorted(groups)[:-keep_last]:
        for old in groups[timestamp]:
            old.unlink(missing_ok=True)


def setup_logging(
    level: int = PROGRESS,
    log_dir: Path | str | None = None,
    keep_last_logs: int = 90,
) -> Path | None:
    """Configure root logging with colour console output and optional file output.

    The default level is PROGRESS (15) so both ephemeral progress lines and
    persistent INFO/WARNING/ERROR messages are shown.  Pass logging.INFO to
    suppress progress lines entirely (e.g. in automated tests).

    When *log_dir* is given a timestamped log file is created there via a
    ``RotatingFileHandler`` (10 MB per file, 5 backups) at DEBUG level using
    plain-text formatting (no ANSI codes). Since each call creates a new
    timestamped file (and thus a new "group" — see ``_prune_old_log_files``),
    *keep_last_logs* additionally prunes older timestamped groups beyond that
    count, together with any ``.log.1``/``.log.2``/... rotation siblings that
    group produced, so ``log_dir`` doesn't grow without bound across repeated
    invocations. Every script in this project that calls ``setup_logging``
    with the same ``log_dir`` shares one pruning pool — with `make pipeline`,
    `make monitor`, and `make backup` all invoked daily per the documented
    cron schedule, the default of 90 covers roughly a month of runs, not 90
    days of a single script's runs. The path to the new log file is returned
    so callers can display it to the user.
    """
    console_handler = ProgressStreamHandler()
    console_handler.setFormatter(ColorFormatter())
    console_handler.setLevel(level)

    handlers: list[logging.Handler] = [console_handler]
    log_file: Path | None = None

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(PlainFormatter())
        handlers.append(file_handler)
        _prune_old_log_files(log_dir, keep_last_logs)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_file
