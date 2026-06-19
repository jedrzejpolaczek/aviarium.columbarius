"""Backward-compatibility shim for Gold-tier write helpers.

GoldWriter has been merged into DuckDBWriter in src.data.cards.storage.base.
This module re-exports DuckDBWriter under the legacy GoldWriter name so that
existing ``from src.data.cards.storage.gold.writers import GoldWriter`` call
sites continue to work without modification.

Gold tables are always fully replaced on each pipeline run because
window-function features span the entire history and cannot be patched
incrementally. Use DuckDBWriter.full_load() for all Gold writes.

Use DuckDBWriter directly for new code.
"""

from src.data.cards.storage.base import DuckDBWriter as GoldWriter

__all__ = ["GoldWriter"]
