"""Streamlit entry point for exploratory EEG prediction and XAI."""

import hashlib
import logging
from pathlib import Path

import streamlit as st

from src.inference import FinalFusionInference
from src.preprocessing import RecordingRejectedError
from src.ui_service import (
    UIInputError,
    analyze_recording,
    parse_sampling_rate,
    parse_uploaded_recording,
)
from ui.components.disclaimer import render_research_disclaimer
from ui.components.epochs import render_epoch_results, render_explanation_summary
from ui.components.features import render_feature_attributions
from ui.components.raw_attributions import render_raw_attributions
from ui.components.summary import render_prediction_summary


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "deep_learning"
    / "cell_01"
    / "final_fusion"
    / "run_20261116T092136Z"
    / "final_fusion_checkpoint.pt"
)


@st.cache_resource(show_spinner="Loading the saved fusion checkpoint...")
def _load_predictor(checkpoint_path):
    return FinalFusionInference.from_checkpoint(
        checkpoint_path,
        device="cpu",
    )


def _render_results(analysis):
    render_prediction_summary(st, analysis)
    render_epoch_results(
        st,
        analysis,
        sampling_rate_hz=analysis.sampling_rate_hz,
    )
    render_explanation_summary(st, analysis.explanation)

    left, right = st.columns(2)
    with left:
        render_feature_attributions(st, analysis.explanation)
    with right:
        render_raw_attributions(st, analysis.explanation)


def main():
    st.set_page_config(
        page_title="ADHD–HC EEG Research",
        page_icon="🧠",
        layout="wide",
    )
    st.title("ADHD vs Healthy Control EEG Research")
    st.caption(
        "Upload one two-channel-slot recording for exploratory model output "
        "and attribution review."
    )
    render_research_disclaimer(st)
    render_research_disclaimer(st.sidebar)

    st.subheader("Recording input")
    uploaded_file = st.file_uploader(
        "Upload a CSV or NPY recording",
        type=["csv", "npy"],
        help=(
            "CSV: exactly two numeric columns, with an optional header. "
            "NPY: numeric array with shape [samples, 2]. MAT is not supported "
            "in Version 1."
        ),
    )
    sampling_rate_text = st.text_input(
        "Sampling rate (Hz)",
        placeholder="Enter the sampling rate, e.g. 256",
        help="Must match the saved model's preprocessing rate.",
    )
    st.caption(
        "Columns are channel slots 1 and 2; electrode identities are unknown. "
        "Recordings must be long enough for edge trimming and one complete epoch."
    )

    current_key = None
    file_content = None
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        current_key = hashlib.sha256(file_content).hexdigest() + sampling_rate_text
    if st.session_state.get("analysis_key") != current_key:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_key", None)

    if st.button("Analyze recording", type="primary"):
        if uploaded_file is None or file_content is None:
            st.error("Upload a CSV or NPY recording first.")
        else:
            try:
                sampling_rate_hz = parse_sampling_rate(sampling_rate_text)
                recording = parse_uploaded_recording(
                    uploaded_file.name,
                    file_content,
                )
                with st.spinner(
                    "Validating, predicting, and calculating explanations..."
                ):
                    predictor = _load_predictor(str(CHECKPOINT_PATH))
                    analysis = analyze_recording(
                        predictor,
                        recording,
                        sampling_rate_hz=sampling_rate_hz,
                    )
                st.session_state["analysis"] = analysis
                st.session_state["analysis_key"] = current_key
            except UIInputError as error:
                st.error(str(error))
            except RecordingRejectedError as error:
                st.error(f"Recording rejected by EEG quality checks: {error}")
            except (OSError, ValueError, FloatingPointError, RuntimeError) as error:
                LOGGER.exception("EEG recording analysis failed.")
                st.error(f"Could not analyze this recording: {error}")

    analysis = st.session_state.get("analysis")
    if analysis is not None and st.session_state.get("analysis_key") == current_key:
        st.divider()
        _render_results(analysis)


if __name__ == "__main__":
    main()
