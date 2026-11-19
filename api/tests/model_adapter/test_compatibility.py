import numpy as np

from api.app.model_adapter.compatibility import validate_recording_compatibility
from api.app.model_adapter.preprocessing import canonical_trials


def test_valid_recording_and_unknown_orientation(recording):
    assert validate_recording_compatibility(recording) == []
    recording.observed_time_axis = "unknown"
    assert any("orientation" in e for e in validate_recording_compatibility(recording))


def test_rejects_rate_channel_nan_and_short_data(recording):
    recording.sampling_rate_hz = None
    recording.array[0, 0] = np.nan
    recording.n_channels = 3
    recording.array = recording.array[:100]
    errors = validate_recording_compatibility(recording)
    assert len(errors) == 3


def test_rejects_insufficient_continuous_duration(recording):
    recording.array = recording.array[:100]
    assert any("1,536" in error for error in validate_recording_compatibility(recording))


def test_epoched_trials_remain_separate(recording):
    recording.n_trials = 2
    recording.array_3d = np.stack([recording.array, recording.array])
    trials = canonical_trials(recording)
    assert len(trials) == 2
    assert trials[0].shape == (2048, 2)
    assert not np.shares_memory(trials[0], trials[1])


def test_rejects_short_epoched_trial(recording):
    recording.n_trials = 2
    recording.array_3d = np.zeros((2, 1000, 2))
    assert validate_recording_compatibility(recording)
