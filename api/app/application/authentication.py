"""Application level identity mapping and synchronization."""

from datetime import datetime, timezone
from uuid import uuid4

from api.app.domain.auth import AuthenticatedUser


class AuthenticationService:
    def __init__(self, provider, user_repository=None):
        self.provider = provider
        self.user_repository = user_repository

    def authenticate(self, token: str) -> AuthenticatedUser:
        decoded = self.provider.verify_token(token)
        uid = decoded.get("uid") or decoded.get("sub")
        if not isinstance(uid, str) or not uid.strip():
            from api.app.repositories.auth_contracts import AuthenticationError
            raise AuthenticationError("Invalid authentication token.")
        return AuthenticatedUser(
            firebase_uid=uid,
            email=decoded.get("email") if isinstance(decoded.get("email"), str) else None,
            email_verified=decoded.get("email_verified") if isinstance(decoded.get("email_verified"), bool) else None,
            claims={key: decoded[key] for key in ("role", "tenant", "email_verified") if key in decoded},
        )

    async def synchronize_user(self, identity: AuthenticatedUser) -> AuthenticatedUser:
        if self.user_repository is None:
            return identity
        user = await self.user_repository.get_by_firebase_uid(identity.firebase_uid)
        now = datetime.now(timezone.utc)
        if user is None:
            from api.app.domain.entities import User
            user = User(
                id=str(uuid4()),
                email=identity.email,
                created_at=now,
                attributes={"firebase_uid": identity.firebase_uid, "email_verified": identity.email_verified},
            )
            try:
                user = await self.user_repository.create(user)
            except Exception:
                # Concurrent first logins may race; resolve via the unique Firebase UID.
                user = await self.user_repository.get_by_firebase_uid(identity.firebase_uid)
                if user is None:
                    raise
        else:
            changes = {"attributes": {**user.attributes, "firebase_uid": identity.firebase_uid,
                                       "email_verified": identity.email_verified}}
            if identity.email is not None and identity.email != user.email:
                changes["email"] = identity.email
            user = await self.user_repository.update(user.id, changes) or user
        return AuthenticatedUser(
            firebase_uid=identity.firebase_uid, email=identity.email,
            email_verified=identity.email_verified, claims=identity.claims,
            application_user_id=user.id,
        )
