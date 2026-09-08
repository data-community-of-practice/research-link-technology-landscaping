"""
Exception classes for PyRLA
"""


class RLAError(Exception):
    """Base exception class for PyRLA"""
    pass


class RLAAuthenticationError(RLAError):
    """Raised when authentication fails"""
    pass


class RLANotFoundError(RLAError):
    """Raised when a resource is not found"""
    pass


class RLAValidationError(RLAError):
    """Raised when request validation fails"""
    pass


class RLAServerError(RLAError):
    """Raised when server returns 5xx error"""
    pass


class RLARateLimitError(RLAError):
    """Raised when rate-limited (HTTP 429)"""
    def __init__(self, message: str, retry_after: int = 30, response_data: dict | None = None):
        self.retry_after = retry_after
        self.response_data = response_data or {}
        super().__init__(message)
