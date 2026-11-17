"""Signed handcrafted-feature contribution visualization."""

import matplotlib.pyplot as plt
import numpy as np


def render_feature_attributions(st, explanation, top_k=10):
    st.subheader("Top handcrafted feature contributions")
    if explanation.accepted_epoch_count == 0:
        st.info("No feature attributions are available.")
        return

    strengths = np.asarray(
        explanation.mean_absolute_feature_attribution,
        dtype=np.float64,
    )
    signed = np.asarray(
        explanation.mean_signed_feature_attribution,
        dtype=np.float64,
    )
    indices = np.argsort(strengths)[::-1][:top_k]
    names = [explanation.feature_names[index] for index in indices]
    signed_values = signed[indices]
    absolute_values = strengths[indices]
    colors = ["#b33a3a" if value >= 0 else "#3569a8" for value in signed_values]

    figure, axis = plt.subplots(figsize=(10, max(4, 0.42 * len(indices))))
    positions = np.arange(len(indices))
    axis.barh(positions, signed_values, color=colors)
    axis.set_yticks(positions, labels=names)
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean signed contribution to ADHD − Control logit")
    axis.set_title("Ranked by mean absolute attribution")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    plt.close(figure)

    st.caption(
        "Positive contributions move the target margin toward ADHD; negative "
        "contributions move it toward Control. Ranking uses mean absolute "
        "attribution across accepted epochs."
    )
    st.dataframe(
        [
            {
                "Feature": name,
                "Mean signed contribution": float(signed_value),
                "Mean absolute contribution": float(absolute_value),
            }
            for name, signed_value, absolute_value in zip(
                names,
                signed_values,
                absolute_values,
            )
        ],
        hide_index=True,
        use_container_width=True,
    )
