# Drushti Detect frontend

Next.js App Router frontend for Drushti Detect, an exploratory EEG ADHD vs
Control research application. The browser uploads a CSV and displays the
response from the existing FastAPI service; it contains no EEG preprocessing,
model inference, or explanation implementation.

## Requirements

- Node.js 22 or newer
- The existing FastAPI backend running at `http://localhost:8000`

## Run locally

From the project root, start the backend using the existing Python 3.11
environment:

```powershell
$env:CORS_ALLOWED_ORIGINS = "http://localhost:3000"
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location web
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Set
`NEXT_PUBLIC_API_BASE_URL` in `web/.env.local` if the backend uses another
address. The API must allow the frontend origin through its
`CORS_ALLOWED_ORIGINS` setting.

## Input contract

Upload a `.csv` file of at most 10 MiB with exactly two numeric channel
columns. The supported model sampling rate is 256 Hz. The frontend checks the
extension, file size, and sampling rate; the backend remains authoritative for
CSV parsing, signal validation, preprocessing, predictions, and attributions.
CSV header behavior follows the backend parser.

The uploaded file is held in page memory for the request and is not placed in
`localStorage`, `sessionStorage`, or browser-managed persistent storage.

## Tests and build

```powershell
npm test
npm run lint
npm run build
```

## Research limitations

This is an exploratory research tool, not a clinical diagnostic system.
Displayed probabilities are not calibrated confidence estimates. Participant
identities have not been verified, subject-independent generalization has not
been established, and attributions are not proof of causation.
