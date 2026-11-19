from unittest.mock import patch

import pytest

from api.app.model_adapter import explainability


def test_phase1_explainability_wrapper_delegates(recording):
    with patch("api.app.model_adapter.explainability.explain_recording", return_value="explanation") as fn:
        assert explainability.explain_recording(object(), recording.array) == "explanation"
        fn.assert_called_once()
