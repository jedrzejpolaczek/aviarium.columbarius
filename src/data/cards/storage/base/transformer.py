"""Base class for transformation-layer storage tiers (Silver and Gold)."""

from abc import abstractmethod

from src.data.cards.storage.base.storage import BaseStorage
from src.logger import get_logger


logger = get_logger(__name__)


class TransformStorage(BaseStorage):
    """Base for transformation-layer storage (Silver and Gold tiers).

    Subclasses implement _pipeline(update) which is called by the
    public populate() and update() entry points.
    """

    @abstractmethod
    def _pipeline(self, update: bool) -> None:
        """Run the transformation pipeline.

        Args:
            update: If True, upsert into existing tables. If False, full rebuild.
        """

    def populate(self) -> None:
        """Full rebuild of all tables."""
        logger.info("Starting %s populate (full rebuild)", self.__class__.__name__)
        self._pipeline(update=False)

    def update(self) -> None:
        """Incremental update of all tables."""
        logger.info("Starting %s update (incremental)", self.__class__.__name__)
        self._pipeline(update=True)
