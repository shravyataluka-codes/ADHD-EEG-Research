"""Raw EEG attribution charts using channel-slot labels only."""

import matplotlib.pyplot as plt
import numpy as np


def render_raw_attributions(st, explanation):
    st.subheader("Raw EEG attribution by channel slot")
    if explanation.accepted_epoch_count == 0:
        st.info("No raw EEG attributions are available.")
        return

    choices = ["Recording-level mean"] + [
        f"Epoch {item.epoch_index}"
        for item in explanation.epoch_explanations
    ]
    selected = st.selectbox(
        "Attribution view",
        choices,
        key="raw_attribution_view",
    )
    if selected == choices[0]:
        signed = explanation.mean_signed_raw_attribution
        absolute = explanation.mean_absolute_raw_attribution
        heading = "Mean across accepted epochs"
    else:
        epoch_number = int(selected.split()[-1])
        item = next(
            value
            for value in explanation.epoch_explanations
            if value.epoch_index == epoch_number
        )
        signed = item.raw_attributions
        absolute = item.raw_attribution_magnitudes
        heading = f"Epoch {epoch_number}"

    signed = np.asarray(signed, dtype=np.float64)
    absolute = np.asarray(absolute, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    sample_indices = np.arange(signed.shape[1])
    for slot in range(2):
        axes[slot, 0].plot(sample_indices, signed[slot], color="#3569a8")
        axes[slot, 0].axhline(0, color="black", linewidth=0.8)
        axes[slot, 0].set_ylabel(f"Channel slot {slot + 1}")
        axes[slot, 0].set_title("Signed attribution")
        axes[slot, 1].plot(sample_indices, absolute[slot], color="#b36b24")
        axes[slot, 1].set_title("Absolute attribution magnitude")
    axes[1, 0].set_xlabel("Sample within epoch")
    axes[1, 1].set_xlabel("Sample within epoch")
    figure.suptitle(heading)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    plt.close(figure)

    st.caption(
        "Slots are array positions only; electrode identities are unknown. "
        "Attributions are for preprocessed model inputs and do not explain "
        "the filtering operation."
    )
