"""Firebase Admin SDK token verification, isolated to infrastructure."""

from pathlib import Path

from api.app.repositories.auth_contracts import AuthenticationError, AuthenticationProviderUnavailable


class FirebaseAuthenticationProvider:
    def __init__(self, project_id=None, credentials_path=None):
        self.project_id = project_id
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._app = None

    def _get_app(self):
        if self._app is not None:
            return self._app
        try:
            import firebase_admin
            from firebase_admin import credentials
            try:
                app = firebase_admin.get_app()
            except ValueError:
                options = {"projectId": self.project_id} if self.project_id else None
                credential = credentials.Certificate(str(self.credentials_path)) if self.credentials_path else credentials.ApplicationDefault()
                app = firebase_admin.initialize_app(credential, options=options)
            self._app = app
            return app
        except Exception as error:
            raise AuthenticationProviderUnavailable("Firebase authentication is unavailable.") from error

    def verify_token(self, token):
        if not isinstance(token, str) or not token.strip():
            raise AuthenticationError("Bearer token is required.")
        try:
            from firebase_admin import auth
            decoded = auth.verify_id_token(token, app=self._get_app(), check_revoked=True)
            return decoded
        except AuthenticationError:
            raise
        except Exception as error:
            # Firebase token exception classes vary by SDK version; classify verification
            # failures as 401 and initialization/network failures as unavailable above.
            from firebase_admin import auth
            if isinstance(error, auth.CertificateFetchError):
                raise AuthenticationProviderUnavailable("Firebase authentication is unavailable.") from error
            if isinstance(error, (
                auth.InvalidIdTokenError,
                auth.ExpiredIdTokenError,
                auth.RevokedIdTokenError,
                auth.UserDisabledError,
                ValueError,
                KeyError,
            )):
                raise AuthenticationError("Invalid or expired authentication token.") from error
            raise AuthenticationProviderUnavailable("Firebase authentication is unavailable.") from error
