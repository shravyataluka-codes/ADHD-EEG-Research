"""Environment-backed settings for the FastAPI service."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "deep_learning"
    / "cell_01"
    / "final_fusion"
    / "run_20261116T092136Z"
    / "final_fusion_checkpoint.pt"
)

DEFAULT_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
REQUEST_MULTIPART_OVERHEAD_BYTES = 64 * 1024

# Phase 2 ingestion limits
DEFAULT_INGESTION_MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_RECORDINGS_IN_MEMORY = 10
DEFAULT_MAX_TOTAL_RECORDING_MEMORY_BYTES = 512 * 1024 * 1024  # 512 MB
DEFAULT_MAX_WAVEFORM_DISPLAY_POINTS = 2000


def _positive_int(value, name):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _positive_float(value, name, allow_zero=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number.") from error
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _parse_origins(raw_origins):
    origins = tuple(
        origin.strip().rstrip("/")
        for origin in raw_origins.split(",")
        if origin.strip()
    )
    if not origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must contain an explicit origin.")
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins, "
                "not wildcards or URL paths."
            )
    return origins


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    max_request_size_bytes: int = (
        DEFAULT_MAX_UPLOAD_SIZE_BYTES + REQUEST_MULTIPART_OVERHEAD_BYTES
    )
    max_recording_duration_seconds: float = 60.0
    allowed_origins: tuple = ("http://localhost:3000",)
    max_concurrent_inference: int = 1
    inference_wait_timeout_seconds: float = 0.5
    # Phase 2 ingestion limits
    ingestion_max_upload_size_bytes: int = DEFAULT_INGESTION_MAX_UPLOAD_SIZE_BYTES
    max_recordings_in_memory: int = DEFAULT_MAX_RECORDINGS_IN_MEMORY
    max_total_recording_memory_bytes: int = DEFAULT_MAX_TOTAL_RECORDING_MEMORY_BYTES
    max_waveform_display_points: int = DEFAULT_MAX_WAVEFORM_DISPLAY_POINTS
    mongodb_uri: str | None = None
    mongodb_database: str = "drushti_detect"
    mongodb_required: bool = False

    def __post_init__(self):
        if not isinstance(self.checkpoint_path, Path):
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        _positive_int(self.max_upload_size_bytes, "MAX_UPLOAD_SIZE_BYTES")
        _positive_int(self.max_request_size_bytes, "MAX_REQUEST_SIZE_BYTES")
        if self.max_request_size_bytes < self.max_upload_size_bytes:
            raise ValueError(
                "MAX_REQUEST_SIZE_BYTES must be at least MAX_UPLOAD_SIZE_BYTES."
            )
        _positive_float(
            self.max_recording_duration_seconds,
            "MAX_RECORDING_DURATION_SECONDS",
        )
        _positive_int(
            self.max_concurrent_inference,
            "MAX_CONCURRENT_INFERENCE",
        )
        _positive_float(
            self.inference_wait_timeout_seconds,
            "INFERENCE_WAIT_TIMEOUT_SECONDS",
            allow_zero=True,
        )
        _parse_origins(",".join(self.allowed_origins))
        # Phase 2 ingestion limit validation
        _positive_int(self.ingestion_max_upload_size_bytes, "INGESTION_MAX_UPLOAD_SIZE_BYTES")
        _positive_int(self.max_recordings_in_memory, "MAX_RECORDINGS_IN_MEMORY")
        _positive_int(self.max_total_recording_memory_bytes, "MAX_TOTAL_RECORDING_MEMORY_BYTES")
        _positive_int(self.max_waveform_display_points, "MAX_WAVEFORM_DISPLAY_POINTS")
        if self.mongodb_uri is not None:
            uri = self.mongodb_uri.strip()
            if not uri or uri == "YOUR_MONGODB_ATLAS_CONNECTION_STRING":
                raise ValueError("MONGODB_URI must be a valid MongoDB connection string.")
            if not uri.startswith(("mongodb://", "mongodb+srv://")):
                raise ValueError("MONGODB_URI must use mongodb:// or mongodb+srv://.")
            object.__setattr__(self, "mongodb_uri", uri)
        if not self.mongodb_database.strip():
            raise ValueError("MONGODB_DATABASE must not be empty.")
        if self.mongodb_required and not self.mongodb_uri:
            raise ValueError("MONGODB_URI is required when MONGODB_REQUIRED is enabled.")

    @classmethod
    def from_env(cls):
        app_environment = os.getenv("APP_ENV", "development").strip().lower()
        raw_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        if raw_origins is None:
            if app_environment in {"prod", "production"}:
                raise ValueError(
                    "CORS_ALLOWED_ORIGINS must be explicitly configured in production."
                )
            raw_origins = "http://localhost:3000"

        max_upload_size = _positive_int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", DEFAULT_MAX_UPLOAD_SIZE_BYTES),
            "MAX_UPLOAD_SIZE_BYTES",
        )
        max_request_size = _positive_int(
            os.getenv(
                "MAX_REQUEST_SIZE_BYTES",
                max_upload_size + REQUEST_MULTIPART_OVERHEAD_BYTES,
            ),
            "MAX_REQUEST_SIZE_BYTES",
        )

        return cls(
            checkpoint_path=Path(
                os.getenv("EEG_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH))
            ),
            max_upload_size_bytes=max_upload_size,
            max_request_size_bytes=max_request_size,
            max_recording_duration_seconds=_positive_float(
                os.getenv("MAX_RECORDING_DURATION_SECONDS", "60"),
                "MAX_RECORDING_DURATION_SECONDS",
            ),
            allowed_origins=_parse_origins(raw_origins),
            max_concurrent_inference=_positive_int(
                os.getenv("MAX_CONCURRENT_INFERENCE", "1"),
                "MAX_CONCURRENT_INFERENCE",
            ),
            inference_wait_timeout_seconds=_positive_float(
                os.getenv("INFERENCE_WAIT_TIMEOUT_SECONDS", "0.5"),
                "INFERENCE_WAIT_TIMEOUT_SECONDS",
                allow_zero=True,
            ),
            # Phase 2 ingestion limits
            ingestion_max_upload_size_bytes=_positive_int(
                os.getenv(
                    "INGESTION_MAX_UPLOAD_SIZE_BYTES",
                    DEFAULT_INGESTION_MAX_UPLOAD_SIZE_BYTES,
                ),
                "INGESTION_MAX_UPLOAD_SIZE_BYTES",
            ),
            max_recordings_in_memory=_positive_int(
                os.getenv("MAX_RECORDINGS_IN_MEMORY", DEFAULT_MAX_RECORDINGS_IN_MEMORY),
                "MAX_RECORDINGS_IN_MEMORY",
            ),
            max_total_recording_memory_bytes=_positive_int(
                os.getenv(
                    "MAX_TOTAL_RECORDING_MEMORY_BYTES",
                    DEFAULT_MAX_TOTAL_RECORDING_MEMORY_BYTES,
                ),
                "MAX_TOTAL_RECORDING_MEMORY_BYTES",
            ),
            max_waveform_display_points=_positive_int(
                os.getenv("MAX_WAVEFORM_DISPLAY_POINTS", DEFAULT_MAX_WAVEFORM_DISPLAY_POINTS),
                "MAX_WAVEFORM_DISPLAY_POINTS",
            ),
            mongodb_uri=os.getenv("MONGODB_URI") or None,
            mongodb_database=os.getenv("MONGODB_DATABASE", "drushti_detect"),
            mongodb_required=os.getenv("MONGODB_REQUIRED", "false").strip().lower()
            in {"1", "true", "yes"},
        )
