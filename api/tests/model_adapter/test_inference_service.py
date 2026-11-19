from types import SimpleNamespace
from unittest.mock import patch
import asyncio

from api.app.model_adapter.service import AnalysisService, AnalysisError


class State:
    def __init__(self, recording, predictor=object()):
        import asyncio
        self.recordings = {recording.recording_id: recording}
        self.predictor = predictor
        self.analyses = {}
        self.analyses_lock = asyncio.Lock()


def test_success_stores_and_retrieves_analysis(recording):
    app = SimpleNamespace(state=State(recording))
    output = SimpleNamespace(status="ok", summary={"mean_probabilities": {"ADHD": .7, "Control": .3}},
        epoch_predictions=(), candidate_epoch_count=1, rejected_epochs=(), edge_trim_samples_per_side=256,
        epoch_samples=1024, discarded_tail_samples=0)
    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)
    with patch("api.app.model_adapter.service.run_in_threadpool", side_effect=direct_call), \
         patch("api.app.model_adapter.inference.predict_recording", return_value=output):
        response = asyncio.run(AnalysisService(app).analyze("rec-1", False))
    assert response.predicted_class == "ADHD"
    assert asyncio.run(AnalysisService(app).get(response.analysis_id)).recording_id == "rec-1"


def test_missing_predictor(recording):
    app = SimpleNamespace(state=State(recording, None))
    with __import__("pytest").raises(AnalysisError):
        asyncio.run(AnalysisService(app).analyze("rec-1", False))
