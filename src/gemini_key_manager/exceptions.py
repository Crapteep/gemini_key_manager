
class GeminiManagerError(Exception):
    """Base exception for the library."""
    pass


class NoAvailableKeysError(GeminiManagerError):
    """Raised when no keys are available to serve a request."""
    pass


class MaxRetriesExceededError(GeminiManagerError):
    """Raised when a request fails after all retry attempts."""
    pass


class InvalidModelError(GeminiManagerError):
    """Raised when an invalid model is requested."""
    pass


class RateLimitError(GeminiManagerError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(GeminiManagerError):
    """Raised when authentication fails."""
    pass


class ServerError(GeminiManagerError):
    """Raised when server returns an error."""
    pass