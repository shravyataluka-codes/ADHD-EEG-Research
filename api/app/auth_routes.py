"""Minimal authenticated identity endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.app.repositories.auth_contracts import AuthenticationError, AuthenticationProviderUnavailable

router = APIRouter(prefix="/api/v2/auth", tags=["authentication"])
bearer = HTTPBearer(auto_error=False)


async def get_authenticated_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Bearer token required.", headers={"WWW-Authenticate": "Bearer"})
    service = request.app.state.authentication_service
    try:
        identity = await run_in_threadpool(service.authenticate, credentials.credentials)
    except AuthenticationError as error:
        raise HTTPException(status_code=401, detail=str(error), headers={"WWW-Authenticate": "Bearer"}) from error
    except AuthenticationProviderUnavailable as error:
        raise HTTPException(status_code=503, detail="Authentication provider unavailable.") from error
    try:
        return await service.synchronize_user(identity)
    except Exception as error:
        raise HTTPException(status_code=503, detail="User identity synchronization unavailable.") from error


@router.get("/me")
async def me(user=Depends(get_authenticated_user)):
    return {
        "firebase_uid": user.firebase_uid,
        "user_id": user.application_user_id,
        "email": user.email,
        "email_verified": user.email_verified,
    }
