from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from api.app.application.authentication import AuthenticationService
from api.app.auth_routes import get_authenticated_user
from api.app.domain.auth import AuthenticatedUser
from api.app.domain.entities import User
from api.app.infrastructure.firebase.auth import FirebaseAuthenticationProvider
from api.app.infrastructure.firebase.test_auth import TestAuthenticationProvider
from api.app.repositories.auth_contracts import AuthenticationError, AuthenticationProviderUnavailable
from api.app.settings import Settings


def run_async(function, *args):
    import anyio
    return anyio.run(function, *args, backend="asyncio")


def settings(**kwargs):
    values = dict(checkpoint_path=Path("checkpoint.pt"), max_upload_size_bytes=1024,
                  max_request_size_bytes=2048, allowed_origins=("http://localhost:3000",))
    values.update(kwargs)
    return Settings(**values)


def test_auth_endpoint_accepts_token_and_syncs_user(monkeypatch):
    async def inline_threadpool(func, *args):
        return func(*args)
    monkeypatch.setattr("api.app.auth_routes.run_in_threadpool", inline_threadpool)
    repository = MagicMock()
    repository.get_by_firebase_uid = AsyncMock(return_value=None)
    repository.create = AsyncMock(side_effect=lambda user: user)
    service = AuthenticationService(TestAuthenticationProvider({"good": {"uid": "firebase-1", "email": "a@example.test", "email_verified": True}}), repository)
    request = MagicMock()
    request.app.state.authentication_service = service
    user = run_async(get_authenticated_user, request, __import__("fastapi.security", fromlist=["HTTPAuthorizationCredentials"]).HTTPAuthorizationCredentials(scheme="Bearer", credentials="good"))
    assert user.firebase_uid == "firebase-1"
    assert user.application_user_id != "firebase-1"
    assert user.email == "a@example.test"
    assert user.email_verified is True
    repository.create.assert_awaited_once()


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer bad"}])
def test_auth_endpoint_rejects_missing_or_invalid_token(headers):
    from fastapi.security import HTTPAuthorizationCredentials
    provider = TestAuthenticationProvider({"good": {"uid": "u"}})
    request = MagicMock()
    request.app.state.authentication_service = AuthenticationService(provider)
    auth = None
    if "Authorization" in headers:
        scheme, token = headers["Authorization"].split(" ", 1)
        if scheme.lower() == "bearer":
            auth = HTTPAuthorizationCredentials(scheme=scheme, credentials=token)
    with pytest.raises(HTTPException) as raised:
        run_async(get_authenticated_user, request, auth)
    assert raised.value.status_code == 401


@pytest.mark.parametrize("error", [AuthenticationError("expired"), AuthenticationError("revoked")])
def test_service_surfaces_expired_and_revoked_verification_errors(error):
    provider = MagicMock()
    provider.verify_token.side_effect = error
    with pytest.raises(AuthenticationError):
        AuthenticationService(provider).authenticate("token")


def test_firebase_provider_reuses_app_and_verifies_with_revocation_check(monkeypatch):
    import firebase_admin
    from firebase_admin import auth
    app = object()
    get_app = MagicMock(return_value=app)
    initialize = MagicMock()
    verify = MagicMock(return_value={"uid": "u", "email": "u@example.test"})
    monkeypatch.setattr(firebase_admin, "get_app", get_app)
    monkeypatch.setattr(firebase_admin, "initialize_app", initialize)
    monkeypatch.setattr(auth, "verify_id_token", verify)
    monkeypatch.setattr(firebase_admin.credentials, "Certificate", MagicMock(return_value=object()))
    provider = FirebaseAuthenticationProvider("project", "service.json")
    assert provider.verify_token("signed") == {"uid": "u", "email": "u@example.test"}
    assert provider.verify_token("signed") ["uid"] == "u"
    get_app.assert_called_once()
    initialize.assert_not_called()
    assert verify.call_args.kwargs["check_revoked"] is True


def test_firebase_initializes_once_when_no_default_app(monkeypatch):
    import firebase_admin
    from firebase_admin import auth
    app = object()
    get_app = MagicMock(side_effect=[ValueError("no app"), app])
    initialize = MagicMock(return_value=app)
    verify = MagicMock(return_value={"uid": "u"})
    monkeypatch.setattr(firebase_admin, "get_app", get_app)
    monkeypatch.setattr(firebase_admin, "initialize_app", initialize)
    monkeypatch.setattr(auth, "verify_id_token", verify)
    monkeypatch.setattr(firebase_admin.credentials, "Certificate", MagicMock(return_value=object()))
    provider = FirebaseAuthenticationProvider("project", "service.json")
    provider.verify_token("a")
    provider.verify_token("b")
    initialize.assert_called_once()
    assert initialize.call_args.kwargs["options"] == {"projectId": "project"}


def test_firebase_unavailable_and_invalid_token_are_classified(monkeypatch):
    import firebase_admin
    from firebase_admin import auth
    monkeypatch.setattr(firebase_admin, "get_app", MagicMock(side_effect=RuntimeError("down")))
    with pytest.raises(AuthenticationProviderUnavailable):
        FirebaseAuthenticationProvider().verify_token("token")
    monkeypatch.setattr(firebase_admin, "get_app", MagicMock(return_value=object()))
    monkeypatch.setattr(auth, "verify_id_token", MagicMock(side_effect=ValueError("expired")))
    with pytest.raises(AuthenticationError):
        FirebaseAuthenticationProvider().verify_token("token")
    monkeypatch.setattr(auth, "verify_id_token", MagicMock(side_effect=auth.CertificateFetchError("offline", OSError("offline"))))
    with pytest.raises(AuthenticationProviderUnavailable):
        FirebaseAuthenticationProvider().verify_token("token")


def test_synchronization_updates_existing_user_and_resolves_duplicate_race():
    async def scenario():
        repo = MagicMock()
        repo.get_by_firebase_uid = AsyncMock(side_effect=[None, User("internal-1", "x@test", datetime.now(timezone.utc)), None, User("internal-2", None, datetime.now(timezone.utc))])
        repo.create = AsyncMock(side_effect=[RuntimeError("duplicate"), User("internal-2", None, datetime.now(timezone.utc))])
        repo.update = AsyncMock(return_value=None)
        service = AuthenticationService(TestAuthenticationProvider({"t": {"uid": "uid", "email": "x@test"}}), repo)
        identity = service.authenticate("t")
        await service.synchronize_user(identity)
        existing = User("internal-1", "x@test", datetime.now(timezone.utc), {"firebase_uid": "uid"})
        repo.get_by_firebase_uid = AsyncMock(side_effect=[existing, None])
        await service.synchronize_user(identity)
        repo.update.assert_awaited_once()
    run_async(scenario)


def test_test_provider_is_rejected_in_production():
    with pytest.raises(ValueError, match="disabled in production"):
        settings(auth_provider="test", app_environment="production")
