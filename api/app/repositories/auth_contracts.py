"""Authentication provider contract; adapters own token verification."""

from typing import Any, Protocol


class AuthenticationError(Exception):
    """Token is absent, invalid, expired, revoked, or unverifiable."""


class AuthenticationProviderUnavailable(Exception):
    """The configured identity provider cannot currently verify tokens."""


class AuthenticationProvider(Protocol):
    def verify_token(self, token: str) -> dict[str, Any]: ...
