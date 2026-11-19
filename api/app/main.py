"""FastAPI application for the Drushti Detect EEG research tool."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.inference import FinalFusionInference
from src.preprocessing import RecordingRejectedError
from src.ui_service import UIInputError

from .ingestion.routes import router as ingestion_router
from .schemas import AnalysisResponse, HealthResponse, ReadyResponse
from .service import UploadTooLargeError, analyze_csv_upload
from .settings import Settings


LOGGER = logging.getLogger(__name__)


class RequestTooLargeError(Exception):
    """Request body exceeded the configured ASGI receive limit."""


class RequestBodyLimitMiddleware:
    """Enforce a body limit for both declared and streamed request bodies."""

    def __init__(self, app, max_body_size_bytes):
        self.app = app
        self.max_body_size_bytes = max_body_size_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return
            if declared_size > self.max_body_size_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Request exceeds the configured size limit."},
                )
                await response(scope, receive, send)
                return

        received_size = 0

        async def limited_receive():
            nonlocal received_size
            message = await receive()
            if message["type"] == "http.request":
                received_size += len(message.get("body", b""))
                if received_size > self.max_body_size_bytes:
                    scope.setdefault("state", {})["request_body_too_large"] = True
                    raise RequestTooLargeError()
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings=None, predictor_loader=None):
    """Build the service; loader injection keeps API tests checkpoint-free."""
    configuration = settings or Settings.from_env()
    loader = predictor_loader or FinalFusionInference.from_checkpoint

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.model_load_error_type = None
        app.state.recordings = {}
        app.state.recordings_lock = asyncio.Lock()
          # Phase 2: temporary in-memory recording store
        try:
            app.state.predictor = loader(
                str(configuration.checkpoint_path),
                device="cpu",
            )
        except Exception as error:
            app.state.model_load_error_type = type(error).__name__
            LOGGER.error(
                "Final fusion checkpoint initialization failed (%s).",
                app.state.model_load_error_type,
            )
        app.state.inference_semaphore = asyncio.Semaphore(
            configuration.max_concurrent_inference
        )
        yield

    app = FastAPI(
        title="Drushti Detect API",
        description=(
            "Drushti Detect provides research-only EEG inference and Integrated "
            "Gradients explanations. Not a clinical diagnostic service."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = configuration

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_size_bytes=configuration.max_request_size_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configuration.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Phase 2: mount EEG ingestion router
    app.include_router(ingestion_router)

    @app.exception_handler(RequestTooLargeError)
    async def request_too_large_error_handler(_request, _error):
        return JSONResponse(
            status_code=413,
            content={"detail": "Request exceeds the configured size limit."},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler_safe(request, error):
        if request.scope.get("state", {}).get("request_body_too_large"):
            return JSONResponse(
                status_code=413,
                content={"detail": "Request exceeds the configured size limit."},
            )
        return await http_exception_handler(request, error)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_request, _error):
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "Invalid request. Provide a CSV file and a numeric "
                    "sampling_rate in Hz."
                )
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(request: Request):
        if request.app.state.predictor is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "The inference model is not ready."},
            )
        return ReadyResponse()

    @app.post("/api/v1/analyze", response_model=AnalysisResponse)
    async def analyze(
        request: Request,
        file: UploadFile = File(..., description="CSV with two numeric channels."),
        sampling_rate: float = Form(..., description="Sampling rate in Hz."),
    ):
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="The inference model is not ready.",
            )

        semaphore = request.app.state.inference_semaphore
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=configuration.inference_wait_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise HTTPException(
                status_code=503,
                detail="Inference capacity is busy; retry the request shortly.",
            ) from error

        try:
            content = await file.read(configuration.max_upload_size_bytes + 1)
            if len(content) > configuration.max_upload_size_bytes:
                raise UploadTooLargeError(
                    "Upload exceeds the configured file-size limit."
                )
            return await run_in_threadpool(
                analyze_csv_upload,
                predictor,
                file.filename or "",
                content,
                sampling_rate,
                configuration,
            )
        except UploadTooLargeError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except UIInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RecordingRejectedError as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Recording rejected by EEG quality checks "
                    f"({error.reason}; channel slots {error.channel_slots})."
                ),
            ) from error
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.error("Recording analysis failed (%s).", type(error).__name__)
            raise HTTPException(
                status_code=500,
                detail="Analysis failed due to an internal inference error.",
            ) from None
        finally:
            try:
                await file.close()
            finally:
                semaphore.release()

    return app


app = create_app()
