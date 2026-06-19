"""Exception hierarchy for the card data sources pipeline."""


class SourceError(Exception):
    pass


class SourceNotRegisteredError(SourceError):
    pass


class SourceDownloadError(SourceError):
    pass


class SourceLoadError(SourceError):
    pass
