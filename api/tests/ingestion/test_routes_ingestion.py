"""Tests for api.app.ingestion.routes — the FastAPI endpoints.

All test data is synthetic.  No real research recordings are used.
Tests use ``fastapi.testclient.TestClient`` with injected settings and a
mock predictor loader so that PyTorch is never loaded.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy.io
from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.settings import Settings


# ---------------------------------------------------------------------------
# App / client helpers
# ---------------------------------------------------------------------------


def _settings(**overrides) -> Settings:
    defaults = dict(
        checkpoint_path=Path("unused.pt"),
        max_upload_size_bytes=2 * 1024 * 1024,
        max_request_size_bytes=3 * 1024 * 1024,
        max_recording_duration_seconds=60,
        allowed_origins=("http://localhost:3000",),
        max_concurrent_inference=1,
        inference_wait_timeout_seconds=0.1,
        ingestion_max_upload_size_bytes=10 * 1024 * 1024,
        max_recordings_in_memory=10,
        max_total_recording_memory_bytes=50 * 1024 * 1024,
        max_waveform_display_points=2000,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client(settings=None) -> TestClient:
    return TestClient(
        create_app(
            settings or _settings(),
            predictor_loader=lambda *_a, **_kw: SimpleNamespace(
                preprocessing_configuration={
                    "sampling_rate_hz": 256,
                    "edge_trim_seconds": 1,
                    "epoch_duration_seconds": 4,
                }
            ),
        )
    )


# ---------------------------------------------------------------------------
# Synthetic upload builders
# ---------------------------------------------------------------------------


def _csv_bytes(n_samples=100, n_channels=2, header=True):
    rng = np.random.default_rng(42)
    vals = rng.standard_normal((n_samples, n_channels))
    lines = []
    if header:
        lines.append(",".join(f"ch{i}" for i in range(n_channels)))
    for row in vals:
        lines.append(",".join(f"{v:.8f}" for v in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mat_v5_bytes(variable_name="eeg", shape=(500, 4)):
    rng = np.random.default_rng(7)
    buf = io.BytesIO()
    scipy.io.savemat(buf, {variable_name: rng.standard_normal(shape).astype(np.float64)})
    return buf.getvalue()


def _mat_hdf5_bytes(variable_name="eeg", shape=(500, 4)):
    rng = np.random.default_rng(13)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset(variable_name, data=rng.standard_normal(shape).astype(np.float64))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Upload: CSV
# ---------------------------------------------------------------------------


class TestCSVUpload:

    def test_successful_csv_upload(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source_format"] == "csv"
        assert body["n_channels"] == 2
        assert body["n_samples"] == 100
        assert body["recording_id"]

    def test_csv_upload_no_sampling_rate_still_succeeds_with_warning(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sampling_rate_hz"] is None
        assert body["import_status"] == "warnings"

    def test_csv_upload_multichannel(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("eeg8.csv", _csv_bytes(n_channels=8), "text/csv")},
                data={"sampling_rate_hz": "512"},
            )
        assert resp.status_code == 200
        assert resp.json()["n_channels"] == 8


# ---------------------------------------------------------------------------
# Upload: MAT
# ---------------------------------------------------------------------------


class TestMATUpload:

    def test_successful_mat_v5_upload(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("eeg.mat", _mat_v5_bytes(), "application/octet-stream")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source_format"] == "mat"
        assert body["provenance"] == "mat_v5"

    def test_successful_mat_hdf5_upload(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("eeg.mat", _mat_hdf5_bytes(), "application/octet-stream")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provenance"] == "mat_v7.3_hdf5"

    def test_mat_v5_ambiguous_needs_variable_name(self):
        rng = np.random.default_rng(7)
        buf = io.BytesIO()
        scipy.io.savemat(buf, {
            "eeg_a": rng.standard_normal((2000, 2)).astype(np.float64),
            "eeg_b": rng.standard_normal((2000, 3)).astype(np.float64),
        })
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("multi.mat", buf.getvalue(), "application/octet-stream")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 422
        assert "candidate" in resp.json()["detail"].lower()

    def test_mat_v5_explicit_variable_selection(self):
        rng = np.random.default_rng(7)
        buf = io.BytesIO()
        scipy.io.savemat(buf, {
            "eeg_a": rng.standard_normal((2000, 2)).astype(np.float64),
            "eeg_b": rng.standard_normal((2000, 3)).astype(np.float64),
        })
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("multi.mat", buf.getvalue(), "application/octet-stream")},
                data={"sampling_rate_hz": "256", "variable_name": "eeg_a"},
            )
        assert resp.status_code == 200
        assert resp.json()["n_channels"] == 2


# ---------------------------------------------------------------------------
# Upload: invalid files
# ---------------------------------------------------------------------------


class TestInvalidUpload:

    def test_unsupported_extension_rejected(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("data.txt", b"hello world", "text/plain")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 415

    def test_empty_csv_rejected(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("empty.csv", b"", "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 422

    def test_non_numeric_csv_rejected(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("bad.csv", b"a,b\n1.0,oops\n", "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 422

    def test_invalid_mat_magic_rejected(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("bad.mat", b"not a mat file" + b"\x00" * 100, "application/octet-stream")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 415  # UnsupportedFormatError → 415

    def test_oversized_upload_rejected(self):
        settings = _settings(ingestion_max_upload_size_bytes=1024)
        with _client(settings) as client:
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("big.csv", _csv_bytes(n_samples=500), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Recording listing
# ---------------------------------------------------------------------------


class TestRecordingListing:

    def test_list_empty(self):
        with _client() as client:
            resp = client.get("/api/v2/recordings")
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

    def test_list_after_upload(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            assert upload.status_code == 200
            resp = client.get("/api/v2/recordings")
        assert resp.json()["total_count"] == 1
        assert resp.json()["recordings"][0]["source_format"] == "csv"


# ---------------------------------------------------------------------------
# Metadata retrieval
# ---------------------------------------------------------------------------


class TestMetadataRetrieval:

    def test_get_recording_metadata(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}")
        assert resp.status_code == 200
        assert resp.json()["recording_id"] == rid

    def test_nonexistent_recording_returns_404(self):
        with _client() as client:
            resp = client.get("/api/v2/recordings/nonexistent-id-12345")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Validation endpoint
# ---------------------------------------------------------------------------


class TestValidation:

    def test_validation_endpoint(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/validation")
        assert resp.status_code == 200
        body = resp.json()
        assert body["import_status"] in ("ok", "warnings")
        assert isinstance(body["warnings"], list)


# ---------------------------------------------------------------------------
# Signal quality endpoint
# ---------------------------------------------------------------------------


class TestQuality:

    def test_quality_endpoint(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/quality")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_channels"] == 2
        assert len(body["channel_stats"]) == 2


# ---------------------------------------------------------------------------
# Waveform endpoint
# ---------------------------------------------------------------------------


class TestWaveform:

    def test_waveform_full_recording(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/waveform")
        assert resp.status_code == 200
        body = resp.json()
        assert body["time_axis_unit"] == "seconds"
        assert len(body["channels"]) == 2

    def test_waveform_channel_selection(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(n_channels=4), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/waveform?channels=0,2")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["channels"]) == 2
        assert body["channels"][0]["channel_index"] == 0
        assert body["channels"][1]["channel_index"] == 2

    def test_waveform_invalid_channel(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/waveform?channels=99")
        assert resp.status_code == 422

    def test_waveform_time_window(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(n_samples=512), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/waveform?start_s=0&end_s=0.5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["channels"][0]["n_original_samples"] == 128

    def test_waveform_nonexistent_recording_404(self):
        with _client() as client:
            resp = client.get("/api/v2/recordings/fake-id/waveform")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Channel stats endpoint
# ---------------------------------------------------------------------------


class TestStats:

    def test_stats_endpoint(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_channels"] == 2
        assert len(body["signal_min"]) == 2
        assert len(body["signal_max"]) == 2


# ---------------------------------------------------------------------------
# PSD endpoint
# ---------------------------------------------------------------------------


class TestPSD:

    def test_psd_endpoint(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(n_samples=512), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/psd")
        assert resp.status_code == 200
        body = resp.json()
        assert body["method"] == "welch"
        assert len(body["channels"]) == 2
        assert len(body["assumptions"]) > 0

    def test_psd_without_rate_fails(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                # no sampling_rate_hz
            )
            rid = upload.json()["recording_id"]
            resp = client.get(f"/api/v2/recordings/{rid}/psd")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# MAT inspection endpoint
# ---------------------------------------------------------------------------


class TestMatInspection:

    def test_inspect_mat_v5(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/inspect-mat",
                files={"file": ("eeg.mat", _mat_v5_bytes(), "application/octet-stream")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "mat_v5"
        assert "eeg" in body["numeric_candidates"]

    def test_inspect_mat_hdf5(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/inspect-mat",
                files={"file": ("eeg.mat", _mat_hdf5_bytes(), "application/octet-stream")},
            )
        assert resp.status_code == 200
        assert resp.json()["format"] == "mat_v7.3_hdf5"

    def test_inspect_non_mat_rejected(self):
        with _client() as client:
            resp = client.post(
                "/api/v2/recordings/inspect-mat",
                files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
            )
        assert resp.status_code == 415


# ---------------------------------------------------------------------------
# Phase 1 backward compatibility
# ---------------------------------------------------------------------------


class TestPhase1Unchanged:

    def test_health_still_works(self):
        with _client() as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_still_works(self):
        with _client() as client:
            resp = client.get("/ready")
        assert resp.status_code == 200

    def test_v1_analyze_still_rejects_bad_csv(self):
        with _client() as client:
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("bad.txt", b"not csv", "text/plain")},
                data={"sampling_rate": "256"},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Storage limits
# ---------------------------------------------------------------------------


class TestStorageLimits:

    def test_recording_count_limit(self):
        settings = _settings(max_recordings_in_memory=2)
        with _client(settings) as client:
            for i in range(2):
                resp = client.post(
                    "/api/v2/recordings/upload",
                    files={"file": (f"r{i}.csv", _csv_bytes(), "text/csv")},
                    data={"sampling_rate_hz": "256"},
                )
                assert resp.status_code == 200, f"Upload {i} failed: {resp.text}"
            # Third upload should be rejected.
            resp = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("overflow.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        assert resp.status_code == 507

    def test_storage_note_present_in_metadata(self):
        with _client() as client:
            upload = client.post(
                "/api/v2/recordings/upload",
                files={"file": ("test.csv", _csv_bytes(), "text/csv")},
                data={"sampling_rate_hz": "256"},
            )
        body = upload.json()
        assert "temporary" in body["storage_note"].lower()
        assert "non-persistent" in body["storage_note"].lower()
# ---------------------------------------------------------------------------
# Regression: decoded recording memory limits
# ---------------------------------------------------------------------------

class TestDecodedMemoryLimits:

    def test_decoded_array_memory_limit(self):
        """A small upload can expand into a larger decoded array."""
        settings = _settings(
            max_total_recording_memory_bytes=1000,
        )

        with _client(settings) as client:
            response = client.post(
                "/api/v2/recordings/upload",
                files={
                    "file": (
                        "large_decoded.csv",
                        _csv_bytes(n_samples=100, n_channels=4),
                        "text/csv",
                    )
                },
                data={"sampling_rate_hz": "256"},
            )

            assert response.status_code == 507

    def test_recording_not_stored_when_memory_limit_exceeded(self):
        settings = _settings(
            max_total_recording_memory_bytes=1000,
        )

        with _client(settings) as client:
            client.post(
                "/api/v2/recordings/upload",
                files={
                    "file": (
                        "large.csv",
                        _csv_bytes(n_samples=100, n_channels=4),
                        "text/csv",
                    )
                },
                data={"sampling_rate_hz": "256"},
            )

            response = client.get("/api/v2/recordings")

            assert response.status_code == 200
            assert response.json()["total_count"] == 0

