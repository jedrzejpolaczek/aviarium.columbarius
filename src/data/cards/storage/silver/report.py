"""Transform issue report writer for the Silver tier."""

import json
from datetime import datetime
from pathlib import Path

from src.logger import get_logger


logger = get_logger(__name__)


def write_report(issues: list[dict[str, object]], report_path: str) -> None:
    """Write the transformation issues to a JSON file.

    Creates parent directories if they do not exist. Overwrites any
    existing report at the same path.

    Args:
        issues: Collected issue dicts from the transformation pipeline.
        report_path: File path where the report will be written.
    """
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_issues": len(issues),
        "issues": issues,
    }
    try:
        Path(report_path).write_text(json.dumps(report, indent=2))
    except OSError as e:
        logger.warning("Could not write transform report to %s: %s", report_path, e)
        return
    logger.info("Transform report written to %s (%d issues)", report_path, len(issues))
