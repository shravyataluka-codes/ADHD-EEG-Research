"""Explicit local/test-only authentication adapter."""

from api.app.repositories.auth_contracts import AuthenticationError


class TestAuthenticationProvider:
    __test__ = False

    def __init__(self, tokens=None):
        self.tokens = dict(tokens or {})

    def verify_token(self, token):
        identity = self.tokens.get(token)
        if identity is None:
            raise AuthenticationError("Invalid authentication token.")
        return dict(identity)
