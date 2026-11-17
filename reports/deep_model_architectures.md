# Cell 1 Deep-Learning Architecture Design

## Scope and input contract

The library in `src/deep_models.py` provides compact research-inspired
architectures for the Cell 1 binary classification experiment. Each model
accepts raw EEG epochs with shape **`[batch, 2, 1024]`** and returns two
unnormalized class logits, ordered `Control = 0`, `ADHD = 1`. Inputs contain
two generic channel slots; neither slot is assigned an anatomical name.

The implementations expose `forward_features()` to provide a learned raw-EEG
representation for a future feature-fusion experiment. No fusion is
implemented. Handcrafted features from `src/features.py` are not substituted
for the raw EEG input.

Parameter counts below are trainable counts for the default hyperparameters in
`configs/deep_models_cell1.yaml`. They describe the implementations in this
repository, not parameter counts claimed for the cited papers.

## Architectures

### EEGNet-inspired model

- **Research reference:** Lawhern et al. (2018), *EEGNet: a compact
  convolutional neural network for EEG-based brain-computer interfaces*.
  [DOI: 10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- **Original concept:** Compact EEG decoding with temporal convolutions,
  depthwise spatial filtering, and separable convolutions.
- **Our adaptation:** One temporal convolution, depthwise spatial convolution
  spanning the two input slots, separable temporal/pointwise convolution,
  ELU, average pooling, dropout, and an adaptive pooled binary head.
- **Dimensions:** `[batch, 2, 1024]` to `[batch, 2]`; `forward_features()` has
  width 128.
- **Trainable parameters:** 1,370.
- **Why included:** It is a compact convolutional reference architecture
  designed for EEG-like signals and is suitable as a small-group baseline.
- **Limitations:** The two-slot spatial convolution can only learn a
  two-input relationship. It does not represent a full scalp montage, and the
  physical channel identities/order remain unverified.

### CNN-LSTM

- **Research reference:** Wang et al. (2022), CNN-LSTM model for EEG-based
  ADHD/ADD/healthy-child classification.
  [DOI: 10.1088/1741-2552/ac7f5d](https://doi.org/10.1088/1741-2552/ac7f5d)
- **Original concept:** CNN feature extraction followed by LSTM sequence
  modeling, applied in the cited work to pediatric ERP EEG and a three-class
  task.
- **Our adaptation:** Two compact temporal CNN stages reduce the sequence
  length before a single bidirectional LSTM layer; sequence-average pooling
  and a small projection feed the binary head. This is not an exact
  reproduction of the paper's layer dimensions, data, or task.
- **Dimensions:** `[batch, 2, 1024]` to `[batch, 2]`; two pooling stages reduce
  the temporal sequence to 256 steps before the LSTM; `forward_features()` has
  width 32.
- **Trainable parameters:** 15,650.
- **Why included:** It tests whether local temporal filters followed by
  recurrent sequence modeling offer a useful alternative to convolution-only
  and attention models.
- **Limitations:** The architecture is adapted to 2-channel, 4-second adult
  epochs, unlike the cited pediatric ERP setting. The two generic slots do not
  support claims about broader scalp or spatial patterns.

### EEG-Transformer-inspired model

- **Research reference:** He et al. (2023), EEG-Transformer.
  [DOI: 10.1088/1741-2552/acf7f5](https://doi.org/10.1088/1741-2552/acf7f5)
- **Original concept:** Temporal token embeddings, positional information,
  self-attention, and feed-forward Transformer blocks for EEG classification.
  The cited paper's configuration uses six Transformer blocks and six heads
  on a much larger 56-channel pediatric dataset.
- **Our adaptation:** A strided convolution creates temporal tokens; a
  learnable positional embedding and two pre-normalized Transformer blocks
  (four heads, width 32) process them. Global max pooling and a binary head
  produce logits. This is a compact adaptation, not an exact reproduction.
- **Dimensions:** `[batch, 2, 1024]` to `[batch, 2]`; convolution creates 127
  tokens; `forward_features()` has width 32.
- **Trainable parameters:** 22,402.
- **Why included:** It provides a compact attention-based comparison for
  learned temporal dependencies across an epoch.
- **Limitations:** The smaller dataset and two channel slots do not support
  direct claims of reproducing the cited 56-channel pediatric results or
  learning full spatial brain representations.

### EEG Conformer

- **Research reference:** Song et al. (2023), *EEG Conformer: Convolutional
  Transformer for EEG Decoding and Visualization*.
  [DOI: 10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
- **Original concept:** Convolutional feature extraction followed by
  Transformer-based processing to combine local EEG patterns and longer-range
  dependencies.
- **Our adaptation:** Temporal convolution and a channel-mixing convolution
  across the two input slots, followed by batch normalization, ELU, average
  pooling, temporal token projection, one compact self-attention block, and
  global average pooling. This is not an exact reproduction.
- **Dimensions:** `[batch, 2, 1024]` to `[batch, 2]`; pooling creates 128
  temporal tokens; `forward_features()` has width 32.
- **Trainable parameters:** 13,818.
- **Why included:** It combines convolutional local feature extraction with
  self-attention in a compact architecture for direct comparison with EEGNet
  and the Transformer model.
- **Limitations:** Channel mixing is limited to two generic slots. It must not
  be interpreted as full scalp topography or anatomical spatial filtering.

## Configuration and research limitations

`configs/deep_models_cell1.yaml` records the fixed seed, Cell 1 raw-epoch
input, generic channel count, model hyperparameters, dropout, weight decay,
batch size, learning rate, epoch cap, and early-stopping patience. Training
and evaluation are not implemented here.

The current 79 groups are provisional source-file/storage-slot identifiers,
not verified participant IDs. Consequently, these architectures and their
parameter counts do not establish subject-independent performance. The
channel order is also unverified. No model is clinically validated by being
implemented or passing synthetic tests.

## References

1. He et al. (2023). EEG-Transformer.
   [https://doi.org/10.1088/1741-2552/acf7f5](https://doi.org/10.1088/1741-2552/acf7f5)
2. Wang et al. (2022). CNN-LSTM for ADHD/ADD/healthy-child EEG classification.
   [https://doi.org/10.1088/1741-2552/ac7f5d](https://doi.org/10.1088/1741-2552/ac7f5d)
3. Song et al. (2023). EEG Conformer: Convolutional Transformer for EEG
   Decoding and Visualization.
   [https://doi.org/10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)
4. Lawhern et al. (2018). EEGNet: a compact convolutional neural network for
   EEG-based brain-computer interfaces.
   [https://doi.org/10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
