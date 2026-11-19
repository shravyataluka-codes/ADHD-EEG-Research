# Phase 4.2: Firebase Authentication

## Firebase project setup

Create or select a Firebase project in the Firebase console. Enable the sign-in methods the client will use under **Authentication → Sign-in method**. The client obtains a Firebase ID token after sign-in and sends it as `Authorization: Bearer <ID_TOKEN>`.

## Provider configuration

The backend uses Firebase Admin SDK token verification with revocation checks enabled. It reuses the default Admin app when already initialized and otherwise initializes it once. Set these values in the local root `.env` (never commit that file):

```dotenv
AUTH_PROVIDER=firebase
FIREBASE_PROJECT_ID=YOUR_FIREBASE_PROJECT_ID
FIREBASE_CREDENTIALS_PATH=YOUR_FIREBASE_SERVICE_ACCOUNT_JSON_PATH
```

Install backend dependencies with `pip install -r api/requirements-api.txt`. In production, use a securely mounted service-account file or the platform's Application Default Credentials; do not put credentials in source control, client code, or this documentation.

## Add real credentials locally

In Firebase project settings, create a service-account key only if the deployment environment does not provide Application Default Credentials. Download the JSON directly into a private location outside this repository, restrict file permissions to the backend process, then set `FIREBASE_PROJECT_ID` to the project ID and `FIREBASE_CREDENTIALS_PATH` to that JSON file's absolute path. Keep `.env` and the key outside Git. The example values are placeholders; no credentials are included here.

When a credentials path is omitted, the Admin SDK uses Application Default Credentials. Project ID may also be omitted if the environment can determine it.

If a Phase 4.1 database already created the non-sparse `uq_users_email` index, migrate that index once before deploying this version: drop `uq_users_email` from the `users` collection and let startup recreate it as a sparse unique index. This permits Firebase accounts without an email claim. Startup also creates a sparse unique index on `attributes.firebase_uid` to prevent duplicate internal user mappings.

## Local authentication testing

Run the focused, credential-free tests from the repository root:

```powershell
python -m pytest api/tests/test_authentication.py api/tests/test_persistence_foundation.py
```

For application-level local smoke tests, `AUTH_PROVIDER=test` enables the explicit in-memory test provider only when `APP_ENV` is not `production`. The current built-in test provider has no accepted tokens; inject `TestAuthenticationProvider({"local-token": {"uid": "local-user"}})` through `create_app(..., authentication_provider=...)` in a local harness or test. Never set this provider in production. The `/api/v2/auth/me` endpoint returns only Firebase UID, internal user ID when MongoDB is connected, email, and email verification state. User records are found by the verified Firebase UID and synchronized without trusting frontend-supplied identifiers.
