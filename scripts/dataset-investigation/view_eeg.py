import h5py
import matplotlib.pyplot as plt

MAT_FILE = "dataset/d1.mat"

TRIAL_INDEX = 0

with h5py.File(MAT_FILE, "r") as f:
    d1 = f["d1"]

    print("Dataset shape:", d1.shape)
    print("Dataset dtype:", d1.dtype)

    # One complete trial:
    # 385 sample/time positions × 56 channels
    trial = d1[:, :, TRIAL_INDEX]

    print("Selected trial:", TRIAL_INDEX)
    print("Trial shape:", trial.shape)
    print("Minimum:", trial.min())
    print("Maximum:", trial.max())
    print("Mean:", trial.mean())
    print("Standard deviation:", trial.std())

plt.figure(figsize=(14, 7))

plt.imshow(
    trial.T,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Raw EEG amplitude")

plt.xlabel("Sample / time position")
plt.ylabel("Channel index")
plt.title(f"Raw EEG — Trial {TRIAL_INDEX} — All 56 Channels")

plt.tight_layout()
plt.show()