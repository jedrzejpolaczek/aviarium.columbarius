"""Silver-tier storage package.

Public API re-exported here so callers use the stable path:
    from src.data.cards.storage.silver import SilverStorage
"""

from src.data.cards.storage.silver.storage import SilverStorage

__all__ = ["SilverStorage"]
