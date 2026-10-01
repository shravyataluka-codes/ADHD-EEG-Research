"""
================================================================================
EEG TRANSFORMER-BASED ADHD CLASSIFICATION
RESEARCH PROTOTYPE APPLICATION
================================================================================
Branch: prototype/dummy-ml-integration
Scope: Prototype & Application Layer (Independent of ML Research Pipeline)

Flow:
User ──► EEG Input / Demo Trial ──► Input Validation ──► Model Interface ──► Mock Prediction
================================================================================
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Ensure repository root is on path so imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prototype.data.dummy_generator import (
    generate_synthetic_eeg_trial,
    generate_synthetic_eeg_dataframe,
    STANDARD_TIMEPOINTS,
    STANDARD_CHANNELS,
    STANDARD_DTYPE
)
from prototype.model.model_interface import ModelAdapter, predict
from prototype.model.mock_model import CLASS_NAMES, EXPECTED_SHAPE
from prototype.utils.visualization import (
    plot_eeg_channel_waveform,
    plot_multi_channel_montage,
    plot_prediction_probabilities,
    plot_optional_explanation
)

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & RESEARCH THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EEG-Transformer ADHD Research Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Clean Professional CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e3a8a;
    }

    /* Medical Disclaimer Banner */
    .disclaimer-banner {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        margin-bottom: 18px;
    }

    /* Status Badges */
    .demo-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
    }
    .active-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }

    /* Integration Callout Box */
    .contract-box {
        background: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "eeg_dataframe" not in st.session_state:
    st.session_state["eeg_dataframe"] = None
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "No Trial Loaded"
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "input_validation_status" not in st.session_state:
    st.session_state["input_validation_status"] = "Pending Data"


def load_demo_trial(seed: int = 42):
    """Loads a synthetic EEG trial into session state conforming to [385, 56] float32."""
    df = generate_synthetic_eeg_dataframe(seed=seed)
    st.session_state["eeg_dataframe"] = df
    st.session_state["data_source"] = f"Synthetic Demo Trial (Seed={seed})"
    st.session_state["input_validation_status"] = "Valid [385, 56] float32"
    # Execute inference via model interface
    st.session_state["prediction_result"] = ModelAdapter.predict_safe(df)


# Auto-load demo trial on first launch
if st.session_state["eeg_dataframe"] is None:
    load_demo_trial(seed=42)


# -----------------------------------------------------------------------------
# HEADER & GLOBAL RESEARCH DISCLAIMER
# -----------------------------------------------------------------------------
h_col1, h_col2 = st.columns([0.88, 0.12])
with h_col1:
    st.title("EEG Transformer-Based ADHD Classification")
    st.markdown("##### *EEG Signal Analysis and Research Prototype (Application Layer)*")
with h_col2:
    logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=75)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>IMPORTANT RESEARCH & MEDICAL DISCLAIMER:</b> 
    This prototype is intended for research and educational purposes only. 
    Predictions are model outputs and should <b>not</b> be considered a medical diagnosis. 
    This system does not claim to definitively diagnose ADHD or any clinical psychiatric disorder.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & INTERFACE STATUS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🧭 Prototype Navigation")
nav = st.sidebar.radio(
    label="Select View:",
    options=[
        "1. Dashboard",
        "2. EEG Upload & Demo",
        "3. Signal Inspection",
        "4. Model Prediction",
        "5. Model Information",
        "6. About Project"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Interface Status")

service_status = ModelAdapter.get_service_status()
if not service_status["is_real_model_active"]:
    st.sidebar.markdown('<span class="demo-badge">DEMO / MOCK PREDICTION</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="active-badge">REAL TRANSFORMER ACTIVE</span>', unsafe_allow_html=True)

st.sidebar.caption(f"**Mode:** {service_status['mode']}")
st.sidebar.caption(f"**Contract Shape:** `{service_status['expected_shape']}`")
st.sidebar.caption(f"**Contract Dtype:** `{service_status['expected_dtype']}`")
st.sidebar.caption(f"**Current Source:** {st.session_state['data_source']}")

st.sidebar.markdown("---")
st.sidebar.caption("ADHD EEG Research Project © 2026")


# =============================================================================
# VIEW 1: DASHBOARD
# =============================================================================
if nav == "1. Dashboard":
    st.subheader("System Dashboard & Quick Status")
    st.markdown("""
    This research dashboard demonstrates the complete end-to-end integration workflow for an 
    **EEG Transformer** model classifying ADHD from multi-channel EEG signals.
    """)

    # Overview Cards
    df = st.session_state["eeg_dataframe"]
    timepoints = len(df) if df is not None else STANDARD_TIMEPOINTS
    channels = (len(df.columns) - 1) if df is not None and "timestamp" in df.columns else STANDARD_CHANNELS

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        st.markdown('<div class="metric-card"><div class="metric-label">Input Format</div><div class="metric-val">EEG Trial</div></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Contract Shape</div><div class="metric-val">[{timepoints}, {channels}]</div></div>', unsafe_allow_html=True)
    with d3:
        st.markdown('<div class="metric-card"><div class="metric-label">Contract Dtype</div><div class="metric-val">float32</div></div>', unsafe_allow_html=True)
    with d4:
        st.markdown('<div class="metric-card"><div class="metric-label">Architecture</div><div class="metric-val">Transformer</div></div>', unsafe_allow_html=True)
    with d5:
        st.markdown('<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-val">HC/ADD/ADHD</div></div>', unsafe_allow_html=True)
    with d6:
        st.markdown('<div class="metric-card"><div class="metric-label">Service Status</div><div style="margin-top:6px;"><span class="demo-badge">MOCK MODE</span></div></div>', unsafe_allow_html=True)

    st.info("ℹ️ **Parallel Integration Notice:** The values displayed in this prototype represent the established interface contract. The real preprocessing and Transformer training pipelines are running independently on dedicated ML research branches.")

    # High-level architecture flow
    st.markdown("### 🔄 Prototype Integration Flow")
    st.markdown("""
    ```
    ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
    │    User / EEG Input    │ ──►  │    Input Validation    │ ──►  │    Model Interface     │ ──►  │     Mock Prediction    │
    │  (Demo or CSV Trial)   │      │   ([385, 56], float32) │      │  (ModelAdapter Bridge) │      │  (Class + Confidence)  │
    └────────────────────────┘      └────────────────────────┘      └────────────────────────┘      └────────────────────────┘
    ```
    """)

    col_btn1, col_btn2 = st.columns([0.3, 0.7])
    with col_btn1:
        if st.button("📥 Reload Synthetic Demo Trial", use_container_width=True):
            load_demo_trial(seed=42)
            st.success("Loaded synthetic EEG trial!")
            st.rerun()
    with col_btn2:
        st.caption("Loads a synthetic trial with shape [385, 56] adhering to the ML contract for immediate UI testing.")


# =============================================================================
# VIEW 2: EEG UPLOAD & DEMO
# =============================================================================
elif nav == "2. EEG Upload & Demo":
    st.subheader("EEG Trial Ingestion & Validation")
    st.markdown("Ingest a single EEG trial in CSV format or generate synthetic trials for interface testing.")

    up_col1, up_col2 = st.columns([0.65, 0.35])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload an EEG Trial File (CSV format)",
            type=["csv", "txt"],
            help="File must represent a 2D matrix of shape [385 time positions x 56 channels]."
        )

    with up_col2:
        st.markdown("**Synthetic Demo Trial Generators**")
        st.caption("Generate deterministic synthetic trials:")
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if st.button("Seed 42 (ADHD Profile)", use_container_width=True):
                load_demo_trial(seed=42)
                st.success("Loaded Demo Trial (Seed 42)")
                st.rerun()
        with b_c2:
            if st.button("Seed 100 (HC Profile)", use_container_width=True):
                load_demo_trial(seed=100)
                st.success("Loaded Demo Trial (Seed 100)")
                st.rerun()

    if uploaded_file is not None:
        try:
            df_up = pd.read_csv(uploaded_file)
            st.session_state["eeg_dataframe"] = df_up
            st.session_state["data_source"] = f"Uploaded File: {uploaded_file.name}"
            
            # Test validation
            res = ModelAdapter.predict_safe(df_up)
            st.session_state["prediction_result"] = res
            
            if res.get("status") == "error":
                st.session_state["input_validation_status"] = "Validation Failed"
                st.error(f"❌ Input validation failed: {res['error_message']}")
            else:
                st.session_state["input_validation_status"] = "Valid [385, 56] float32"
                st.success(f"✅ Successfully validated '{uploaded_file.name}' against contract shape [385, 56]!")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Display Metadata & Preview
    if st.session_state["eeg_dataframe"] is not None:
        df = st.session_state["eeg_dataframe"]
        non_time = [c for c in df.columns if c.lower() not in ['timestamp', 'time']]
        n_samples = len(df)
        n_chans = len(non_time)

        st.markdown("---")
        st.markdown("### 📋 Trial Specifications")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Source", st.session_state["data_source"])
        c2.metric("Time Positions (T)", f"{n_samples} (Expected: 385)")
        c3.metric("Channels (C)", f"{n_chans} (Expected: 56)")
        c4.metric("Validation Status", st.session_state["input_validation_status"])

        st.markdown("### 🔍 Trial Data Preview")
        st.dataframe(df.head(8), use_container_width=True)


# =============================================================================
# VIEW 3: SIGNAL INSPECTION
# =============================================================================
elif nav == "3. Signal Inspection":
    st.subheader("Signal Inspection & Waveform Analysis")
    st.markdown("Inspect synthetic EEG channels across the 385 time positions.")

    if st.session_state["eeg_dataframe"] is None:
        st.warning("Please load or generate an EEG trial in the **EEG Upload & Demo** tab first.")
    else:
        df = st.session_state["eeg_dataframe"]
        channel_cols = [c for c in df.columns if c.lower() not in ['timestamp', 'time']]
        time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        time_points = df[time_col].values

        ctrl_col1, ctrl_col2 = st.columns([0.5, 0.5])
        with ctrl_col1:
            sel_chan = st.selectbox("Select Electrode Channel to Plot:", channel_cols, index=0)
        with ctrl_col2:
            show_montage = st.checkbox("Display 56-Channel Stacked Montage Preview", value=False)

        if show_montage:
            fig_montage = plot_multi_channel_montage(df, max_display_chans=16)
            st.plotly_chart(fig_montage, use_container_width=True)
            st.caption("*(Showing first 16 channels vertically offset for layout review)*")
        else:
            sig = df[sel_chan].values
            fig_wave = plot_eeg_channel_waveform(time_points, sig, channel_name=sel_chan.upper())
            st.plotly_chart(fig_wave, use_container_width=True)

        st.markdown("""
        <div class="contract-box">
            <b>NOTE ON PREPROCESSING:</b> Preprocessing (filtering, ICA artifact rejection, and normalization) 
            is intentionally decoupled from this prototype branch. The prototype only expects the finalized 
            <b>[385, 56] float32</b> trial matrix from the ML pipeline.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 4: MODEL PREDICTION
# =============================================================================
elif nav == "4. Model Prediction":
    st.subheader("Model Prediction & Classification Output")
    st.markdown("Execute inference via the Model Interface bridge.")

    if st.session_state["eeg_dataframe"] is None:
        st.warning("Please load or generate an EEG trial in the **EEG Upload & Demo** tab first.")
    else:
        # Run inference trigger
        if st.button("🧠 Execute Model Prediction", type="primary", use_container_width=True):
            st.session_state["prediction_result"] = ModelAdapter.predict_safe(st.session_state["eeg_dataframe"])

        res = st.session_state.get("prediction_result")
        if res is None:
            res = ModelAdapter.predict_safe(st.session_state["eeg_dataframe"])
            st.session_state["prediction_result"] = res

        if res.get("status") == "error":
            st.error(f"Cannot perform prediction: {res.get('error_message')}")
        else:
            st.markdown("---")
            p_col1, p_col2 = st.columns([0.45, 0.55])

            with p_col1:
                st.markdown("### Classification Output")
                pred_class = res["class"]
                confidence_pct = res["confidence"] * 100

                color_map = {
                    "ADHD": "#e74c3c",
                    "HC": "#2ecc71",
                    "ADD": "#f39c12"
                }
                c_color = color_map.get(pred_class, "#333")

                st.markdown(f"""
                <div style="background-color: {c_color}15; border: 2px solid {c_color}; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #555; font-weight: 600;">Predicted Diagnostic Class</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: {c_color}; margin: 8px 0;">{pred_class}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #222;">Model Confidence: {confidence_pct:.1f}%</div>
                    <div style="font-size: 0.85rem; color: #666; margin-top: 4px;">Class Index: {res['class_index']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<span class="demo-badge">{res["status_label"]}</span>', unsafe_allow_html=True)
                st.caption(f"**Medical Notice:** {res['disclaimer']}")

            with p_col2:
                st.markdown("### Class Probability Distribution")
                fig_prob = plot_prediction_probabilities(res["probabilities"])
                st.plotly_chart(fig_prob, use_container_width=True)

            # Optional Explainability Display (Tolerates None/missing gracefully)
            st.markdown("---")
            st.markdown("### 🔍 Model Interpretability & Explanation (Optional)")
            
            if res.get("explanation") is not None:
                fig_exp = plot_optional_explanation(res["explanation"])
                if fig_exp is not None:
                    st.plotly_chart(fig_exp, use_container_width=True)
                    st.caption(f"**Note:** {res['explanation'].get('note', '')}")
            else:
                st.info("ℹ️ **Explanation not provided by model.** The ML contract treats interpretability payloads as optional. Missing explanation fields do not affect classification.")

            # Developer Integration Notice
            st.markdown("""
            <div class="contract-box">
                <b>🔌 HOW FUTURE REAL TRANSFORMER INTEGRATES:</b><br>
                This UI interacts strictly with <code>ModelAdapter.predict(eeg_trial)</code>. 
                When the ML engineer completes Transformer training on the ML branch:
                <ol style="margin-top:6px; margin-bottom:0;">
                    <li>Implement <code>load_trained_model()</code> and <code>predict_trained_model()</code> in <code>prototype/model/trained_model_placeholder.py</code>.</li>
                    <li>Set <code>USE_REAL_MODEL = True</code> in <code>prototype/model/model_interface.py</code>.</li>
                </ol>
                The UI will seamlessly transition from mock predictions to real Transformer inference with <b>zero UI modifications</b>.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# VIEW 5: MODEL INFORMATION
# =============================================================================
elif nav == "5. Model Information":
    st.subheader("EEG Transformer Architecture & Contract Overview")
    st.markdown("Structural overview of the planned sequence modeling architecture.")

    st.markdown("### 📐 Target Model Pipeline")
    st.markdown("""
    ```
          Raw EEG Trial Matrix [385 time positions × 56 channels] (float32)
                                         ↓
                     Preprocessing (Filtering, Artifact Removal)
                                         ↓
                       Temporal & Spatial Embedding Layer
                                         ↓
                         Learnable Positional Encoding
                                         ↓
                ┌─────────────────────────────────────────────────┐
                │            Transformer Encoder Stack            │
                │   - Multi-Head Self-Attention (MHSA)            │
                │   - Layer Normalization & Residual Additions    │
                │   - Position-Wise Feed Forward Network (FFN)    │
                │   - Layer Normalization & Residual Additions    │
                └─────────────────────────────────────────────────┘
                                         ↓
                     Global Max Pooling (GMP Peak Detection)
                                         ↓
                          Dense Classification Layer
                                         ↓
                      Softmax Output: [HC, ADD, ADHD]
    ```
    """)

    st.markdown("### ⚙️ Target Model Hyperparameters")
    st.caption("Hyperparameters are being finalized independently through empirical validation on the ML research branch.")

    params_df = pd.DataFrame([
        {"Hyperparameter": "Input Tensor Shape", "Specification": "[385, 56]", "Status": "Contract Established"},
        {"Hyperparameter": "Input Data Type", "Specification": "float32", "Status": "Contract Established"},
        {"Hyperparameter": "Output Classes", "Specification": "3 (HC: 0, ADD: 1, ADHD: 2)", "Status": "Contract Established"},
        {"Hyperparameter": "Transformer Encoder Layers", "Specification": "TBD", "Status": "Pending ML Experimentation"},
        {"Hyperparameter": "Multi-Head Attention Heads", "Specification": "TBD", "Status": "Pending ML Experimentation"},
        {"Hyperparameter": "Embedding Dimension (d_model)", "Specification": "TBD", "Status": "Pending ML Experimentation"},
        {"Hyperparameter": "Feed-Forward Hidden Dimension", "Specification": "TBD", "Status": "Pending ML Experimentation"},
        {"Hyperparameter": "Dropout Rate", "Specification": "TBD", "Status": "Pending Regularization Tuning"},
        {"Hyperparameter": "Pooling Mechanism", "Specification": "Global Max Pooling (GMP) / TBD", "Status": "Candidate Spec"}
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 6: ABOUT PROJECT
# =============================================================================
elif nav == "6. About Project":
    st.subheader("About the ADHD EEG Research Project")
    st.markdown("Context, methodology, separation of responsibilities, and research safeguards.")

    st.markdown("### 📌 Clinical Motivation")
    st.markdown("""
    **Attention Deficit / Hyperactivity Disorder (ADHD)** is a widespread neurodevelopmental condition. 
    Traditional diagnostic workflows rely on behavioral rating scales and clinical interviews (DSM-5 criteria), 
    which are time-consuming and subjective. Non-invasive Electroencephalography (EEG) provides rich 
    neurophysiological data that can be harnessed through modern deep learning sequence models.
    """)

    st.markdown("### 💡 Proposed Solution")
    st.markdown("""
    This project explores an **EEG Transformer** model using self-attention mechanisms to learn long-range temporal 
    dependencies and multi-channel spatial relationships across the scalp.
    """)

    st.markdown("### 👥 Separation of Responsibilities")
    st.markdown("""
    To ensure clean Git discipline and parallel engineering:
    * **Prototype / Application Layer (This Branch):** Streamlit interface, synthetic demo trial generator, 
      input validation, ModelAdapter bridge, mock prediction service, and contract documentation.
    * **ML / Research Pipeline Layer (Separate Branches):** Real EEG data ingestion, signal filtering, 
      ICA artifact removal, dataset curation, Transformer architecture training, and empirical evaluation.
    """)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-banner">
        <b>FORMAL ACADEMIC DISCLAIMER:</b><br>
        This prototype is developed exclusively for educational and academic research demonstration. 
        It has not been approved for clinical diagnostic use. Under no circumstances should outputs from this prototype 
        be used as clinical advice or medical diagnosis.
    </div>
    """, unsafe_allow_html=True)
