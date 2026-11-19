import numpy as np
import pytest

from api.app.ingestion.recording import EEGRecording


@pytest.fixture
def recording():
    signal = np.zeros((2048, 2), dtype=np.float64)
    return EEGRecording("rec-1", "sample.csv", "csv", "csv", 2, 2048, None,
        256.0, [None, None], None, False, 0, "CSV rows are samples", signal.shape,
        signal, None, {}, "ok")
