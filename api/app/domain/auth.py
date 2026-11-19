"""Storage and provider independent authenticated identity records."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthenticatedUser:
    firebase_uid: str
    email: str | None = None
    email_verified: bool | None = None
    claims: dict[str, Any] | None = None
    application_user_id: str | None = None
