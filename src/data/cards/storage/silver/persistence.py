"""Backward-compatibility shim for Silver-tier write helpers.

SilverWriter has been merged into DuckDBWriter in src.data.cards.storage.base.
This module re-exports DuckDBWriter under the legacy SilverWriter name so that
existing ``from src.data.cards.storage.silver.persistence import SilverWriter``
call sites continue to work without modification.

Use DuckDBWriter directly for new code.
"""

from src.data.cards.storage.base import DuckDBWriter as SilverWriter

__all__ = ["SilverWriter"]
