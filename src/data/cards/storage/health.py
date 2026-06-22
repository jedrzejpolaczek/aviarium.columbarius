import datetime
from dataclasses import dataclass
from typing import Literal

import duckdb

from src.data.cards.storage.base.storage import get_tables
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    layer: str
    status: Literal["PASS", "FAIL"]
    detail: str
