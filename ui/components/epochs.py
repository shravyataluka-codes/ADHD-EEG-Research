"""Epoch-level prediction and quality tables."""


def render_epoch_results(st, analysis, sampling_rate_hz):
    inference = analysis.inference
    explanation = analysis.explanation
    st.subheader("Epoch-level predictions")
    st.write(
        f"Accepted: **{explanation.accepted_epoch_count}** · "
        f"Rejected: **{explanation.rejected_epoch_count}**"
    )

    rows = []
    for prediction, attribution in zip(
        inference.epoch_predictions,
        explanation.epoch_explanations,
    ):
        rows.append(
            {
                "Epoch": prediction.epoch_index,
                "Start (s)": round(prediction.start_sample / sampling_rate_hz, 3),
                "Stop (s)": round(prediction.stop_sample / sampling_rate_hz, 3),
                "Predicted class": prediction.predicted_class,
                "Control model output": prediction.probability_control,
                "ADHD model output": prediction.probability_adhd,
                "ADHD − Control logit": attribution.logit_margin,
                "Completeness residual": attribution.completeness_residual,
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No accepted epochs were available.")

    if inference.rejected_epochs:
        st.markdown("**Rejected epochs**")
        st.dataframe(
            [
                {
                    "Epoch": rejected["epoch_index"],
                    "Start (s)": round(
                        rejected["start_sample"] / sampling_rate_hz, 3
                    ),
                    "Stop (s)": round(
                        rejected["stop_sample"] / sampling_rate_hz, 3
                    ),
                    "Reason": rejected["reason"],
                    "Channel slots": ", ".join(
                        str(slot) for slot in rejected.get("channel_slots", [])
                    ),
                }
                for rejected in inference.rejected_epochs
            ],
            hide_index=True,
            use_container_width=True,
        )


def render_explanation_summary(st, explanation):
    st.subheader("Explanation summary")
    if explanation.accepted_epoch_count == 0:
        st.info("No epoch explanations are available.")
        return

    residuals = [
        abs(item.completeness_residual)
        for item in explanation.epoch_explanations
    ]
    columns = st.columns(2)
    columns[0].metric(
        "Mean |completeness residual|",
        f"{sum(residuals) / len(residuals):.5g}",
    )
    columns[1].metric("Maximum |completeness residual|", f"{max(residuals):.5g}")
    with st.expander("Baseline and attribution notes"):
        st.write(explanation.baseline_description)
        st.caption(
            "A completeness residual is the difference between the target "
            "logit-margin change from baseline and the sum of attributions."
        )
