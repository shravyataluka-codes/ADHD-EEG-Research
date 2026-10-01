"""
================================================================================
PROTOTYPE VISUALIZATION UTILITIES
================================================================================
Interactive Plotly visualizers for:
- EEG channel waveform time-series
- 56-channel montage overview
- Model prediction probability distribution
- Optional model explanation / attention display
================================================================================
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

COLORS = {
    "primary": "#1f77b4",
    "signal": "#3498db",
    "adhd": "#e74c3c",   # Red / Alert
    "hc": "#2ecc71",     # Soft Green
    "add": "#f39c12",    # Amber Yellow
    "grid": "rgba(200, 200, 200, 0.2)"
}


def plot_eeg_channel_waveform(
    time_points: np.ndarray,
    signal: np.ndarray,
    channel_name: str = "Channel 1"
) -> go.Figure:
    """Renders interactive Plotly time-series for a single electrode channel."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=signal,
        mode='lines',
        name=channel_name,
        line=dict(color=COLORS["signal"], width=1.5),
        hovertemplate='Time: %{x:.3f} s<br>Amplitude: %{y:.3f} µV<extra></extra>'
    ))

    fig.update_layout(
        title=f"Synthetic EEG Waveform — {channel_name} (Contract: 385 timepoints)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (µV)",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig


def plot_multi_channel_montage(df: pd.DataFrame, max_display_chans: int = 16) -> go.Figure:
    """
    Renders a subset of channels vertically stacked to simulate EEG montage review.
    """
    non_time_cols = [c for c in df.columns if c.lower() not in ['timestamp', 'time']]
    chans_to_show = non_time_cols[:max_display_chans]
    time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    time_arr = df[time_col].values

    fig = go.Figure()
    offset_step = 1.0

    for idx, col in enumerate(chans_to_show):
        offset = (len(chans_to_show) - 1 - idx) * offset_step
        sig = df[col].values
        fig.add_trace(go.Scatter(
            x=time_arr,
            y=sig + offset,
            mode='lines',
            name=col.upper(),
            line=dict(width=1.0),
            hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"EEG Channel Montage View (First {len(chans_to_show)} of {len(non_time_cols)} Channels)",
        xaxis_title="Time (seconds)",
        yaxis_title="Stacked Amplitude (µV)",
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showticklabels=False, showgrid=False)
    )
    return fig


def plot_prediction_probabilities(probabilities: Dict[str, float]) -> go.Figure:
    """Renders a horizontal probability distribution bar gauge."""
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]

    color_map = {
        "ADHD": COLORS["adhd"],
        "HC": COLORS["hc"],
        "ADD": COLORS["add"]
    }
    bar_colors = [color_map.get(c, "#555") for c in classes]

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(color=bar_colors),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        hovertemplate='Class %{y}: <b>%{x:.1f}%</b><extra></extra>'
    ))

    fig.update_layout(
        title="Model Prediction Probabilities (%)",
        xaxis_title="Class Probability (%)",
        yaxis_title="Diagnostic Class",
        template="plotly_white",
        height=240,
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor=COLORS["grid"]),
        margin=dict(l=60, r=40, t=50, b=30)
    )
    return fig


def plot_optional_explanation(explanation: Optional[Dict[str, Any]]) -> Optional[go.Figure]:
    """
    Renders an optional channel importance / attention bar chart if provided.
    Returns None if explanation is not present.
    """
    if not explanation or "channel_importance" not in explanation:
        return None

    importances = explanation["channel_importance"]
    x_labels = [f"Ch {i+1}" for i in range(len(importances))]

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=importances,
        marker_color="#4facfe",
        hovertemplate='%{x}: Importance %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Optional Interpretability: Electrode Channel Importance",
        xaxis_title="EEG Channels (56)",
        yaxis_title="Relative Attention Weight",
        template="plotly_white",
        height=280,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickangle=90, dtick=2),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"])
    )
    return fig
