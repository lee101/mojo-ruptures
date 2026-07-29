class BadSegmentationParameters(Exception):
    """Raised when the requested partition cannot satisfy its constraints."""


class NotEnoughPoints(Exception):
    """Raised when a segment is shorter than a cost's minimum size."""

