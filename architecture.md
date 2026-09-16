# Proposed Architecture — Multimodal ECG + PPG Cross-Attention Fusion for Arrhythmia Alarm Verification

> **Note on the numbers below:** concrete values (filter counts, kernel sizes, embedding dimensions) are reasonable starting configurations, not tuned results. Present them as the initial setup to be refined on validation folds. If asked how they were chosen, "starting point, tuned during cross-validation" is a defensible answer — a fabricated justification is not.

---

## 1. Input representation

Each training example is a single alarm event, represented as two synchronized channels covering the same time window around the alarm trigger:

- **ECG channel** — the electrical trace, carrying rhythm and morphology information (QRS complexes, intervals, regularity).
- **PPG channel** — the peripheral pulse waveform, carrying mechanical evidence that a heartbeat actually produced blood flow.

Both channels are resampled to a common rate, bandpass-filtered to their own useful frequency band (ECG and PPG have different informative ranges, so the filters differ), and z-normalized per window so the model cannot exploit absolute amplitude, which varies with sensor placement and monitor gain rather than physiology.

The essential property is **temporal alignment**: both channels must describe the same physical interval, sample for sample. If alignment is off, everything downstream compares the wrong moments and the attention mechanism has nothing meaningful to learn.

Network input is therefore a pair of 1D sequences, each of length T (samples in the window).

---

## 2. Modality-specific 1D-CNN encoders

Two convolutional encoders run in parallel — one per signal. They are **not weight-shared**, and this is a deliberate decision worth stating explicitly.

### Why separate encoders

ECG and PPG differ fundamentally in morphology, frequency content, and failure mode. ECG is sharp and spiky (the QRS complex is a high-frequency transient); PPG is smooth and slow (a rounded pressure wave). They also fail differently — ECG corrupts with muscle artifact and electrode motion, PPG with perfusion loss and sensor movement. Forcing both through one shared filter bank would make the early layers learn a compromise suiting neither signal well.

### Structure of each encoder

A stack of roughly three to four convolutional blocks, each consisting of:

- 1D convolution
- Batch normalization
- Non-linearity (ReLU)
- Stride or pooling that halves temporal resolution

Channel width grows as resolution shrinks — a typical progression is 32 → 64 → 128 filters. Kernel sizes run wide in early layers (roughly 7–15 samples, enough to span a full waveform feature) and narrow later. Dropout sits between blocks, with regularization set aggressively given the dataset size.

### Critical design point: the output is a sequence, not a vector

This is the single most important architectural constraint and the one most easily got wrong.

Most CNN classifiers end with global average pooling, collapsing the whole signal into one summary vector. **If either encoder did that here, the entire project would fail** — attention has nothing to align once time has been averaged away.

Instead, each encoder outputs a tensor of shape **(T′, d)**: a reduced-length sequence of d-dimensional feature vectors, where each position still corresponds to a specific moment in the original window.

### What each encoder learns

- **ECG branch** — local rhythm and morphology descriptors: beat presence, interval regularity, complex shape.
- **PPG branch** — pulse-level descriptors: whether a pulse occurred, its amplitude, rise time, and regularity.

Neither branch alone can determine whether an alarm is genuine. Each produces evidence that only becomes decisive when compared against the other.

---

## 3. Cross-attention fusion layer

This is the core contribution and the component the entire experiment is designed to evaluate.

### The mechanism

A transformer attention layer takes the two feature sequences and lets one query the other. The ECG feature sequence supplies the **queries**; the PPG feature sequence supplies the **keys and values** (direction is worth testing empirically — a bidirectional variant computing both directions and combining them is a reasonable extension).

Every ECG timestep computes a similarity score against every PPG timestep, producing a T′ × T′ attention matrix. Scores are softmax-normalized into weights, and each ECG position receives a weighted aggregate of PPG features.

Positional encodings are added to both sequences before attention, since attention is permutation-invariant by construction and would otherwise have no notion of temporal order — fatal for a task where timing is the entire signal.

### Why attention rather than simple concatenation

**1. It handles the pulse transit delay.**
An electrical event on the ECG does not produce a peripheral pulse at the same instant — the pressure wave takes time to reach the finger or ear where PPG is measured, and this delay varies between patients and with physiological state. A naive concatenation model, which lines up ECG sample *t* with PPG sample *t*, compares slightly mismatched moments. Attention learns the offset from data rather than requiring it to be hand-coded or assumed constant.

**2. It performs comparison rather than accumulation.**
Concatenation gives the classifier both feature sets and hopes the dense layers discover the relationship. Cross-attention builds the comparison directly into the architecture: for each region of the ECG, the model explicitly asks which region of the PPG is relevant and what it says. Since alarm verification is fundamentally a question of *agreement versus disagreement* between two views of the same event, an architecture that computes correspondence is better matched to the task than one that merely pools features.

**3. It produces an interpretable byproduct.**
The attention weight matrix can be extracted and visualized, showing which parts of each signal the model relied on. This directly supports the interpretability objective — checking whether the learned alignment corresponds to physiologically plausible ECG-to-pulse timing rather than to a dataset artifact.

### What the layer is expected to learn

- **True alarm** — the ECG abnormality is accompanied by a corresponding disturbance in the pulse: a missing pulse during asystole, an irregular pulse train during a genuine tachyarrhythmia.
- **False alarm** — the ECG shows an abnormal-looking pattern while the PPG continues its normal rhythm undisturbed.

The attention layer's task is to surface this presence or absence of corroboration.

---

## 4. Classification head

The fused representation is pooled across time and passed through a small dense head — typically one hidden layer with dropout, then a single output unit with sigmoid activation producing the true-versus-false alarm decision.

The head is kept deliberately shallow. With roughly 750 labeled records and severe imbalance across the five alarm types, a wide fully-connected head is the easiest place in the network to overfit, and any accuracy gain it produced would be indistinguishable from memorization of the training folds.

Class imbalance is handled with class weighting or a focal loss variant rather than resampling, since resampling risks duplicating patients across folds and undermining the record-wise split.

---

## 5. The baseline ladder — architectural discipline

The proposed model is only meaningful relative to its comparisons, so the ablations belong in the architecture description rather than as an afterthought.

| Variant | Encoders | Fusion | Purpose |
|---|---|---|---|
| ECG-only | ECG CNN | none | Establishes what a single-signal model achieves |
| PPG-only | PPG CNN | none | Tests whether pulse alone carries the answer |
| Concatenation fusion | Both CNNs | features concatenated, no attention | Isolates the value of *having* both signals |
| Cross-attention fusion | Both CNNs | transformer cross-attention | The proposed model |

**The rule governing this ladder:** all four variants are matched in approximate parameter count. Single-modality models are widened to compensate for their missing branch.

Without this constraint, a win by the fusion model would be attributable to extra capacity rather than to fusion, and the experiment would prove nothing. Explicitly matching parameter budgets is what converts the comparison from a demonstration into an actual test. The gap between the concatenation and attention variants is the specific quantity this project exists to measure.

---

## 6. Why this design and not a larger one

Anticipate this question, because it will come. The architecture is small: a few convolutional blocks, one attention layer, a thin head. Someone will ask why not use a large pretrained ECG foundation model or a deep transformer stack.

Model capacity has to be matched to available supervision. With roughly 750 labeled records and only free-tier Colab or Kaggle GPU time, a high-capacity model would overfit long before it generalized, and cross-validation variance would swamp any real effect.

More importantly, the project's contribution is a **controlled comparison**, not a leaderboard score. Scaling every variant up would raise all four numbers without clarifying the one question being asked. The architecture is sized to be exactly complex enough to detect whether cross-modal attention adds something that naive fusion does not — and no more.

---

## Suggested slide split

This runs long for a single slide. A workable division:

- **Architecture: encoders** — sections 1–2
- **Architecture: cross-attention fusion** — section 3 *(the key slide; give it the diagram and the most speaking time)*
- **Classification head and design rationale** — sections 4 and 6, condensed
- Section 5's table sits here or moves into the Methodology slide, depending on time allocation