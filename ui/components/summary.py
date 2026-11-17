"""Recording-level prediction summary."""


def render_prediction_summary(st, analysis):
    inference = analysis.inference
    summary = inference.summary
    st.subheader("Recording-level prediction")
    if summary["valid_epoch_count"] == 0:
        st.info("No valid epochs were available for a prediction.")
        return

    prediction = summary["majority_vote_class"]
    prediction_text = prediction or "Tie (no majority)"
    control_probability = summary["mean_probabilities"]["Control"]
    adhd_probability = summary["mean_probabilities"]["ADHD"]
    columns = st.columns(3)
    columns[0].metric("Majority-vote class", prediction_text)
    columns[1].metric("Control model output", f"{control_probability:.1%}")
    columns[2].metric("ADHD model output", f"{adhd_probability:.1%}")
    st.caption(
        "Probabilities are the mean model outputs across accepted epochs; "
        "they are not calibrated confidence estimates."
    )
