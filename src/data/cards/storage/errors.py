"""Exception hierarchy for the card data storage layer."""


class StorageError(Exception):
    pass


class StorageConnectionError(StorageError):
    pass


class StorageWriteError(StorageError):
    pass
