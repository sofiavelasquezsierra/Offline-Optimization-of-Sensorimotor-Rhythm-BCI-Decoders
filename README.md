# Offline Optimization of Sensorimotor Rhythm BCI Decoders

**Optimizing EEG-based brain-computer interface decoding using Common Spatial Patterns, feature selection, and deep learning (EEGNet)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements and compares two offline decoding pipelines for a sensorimotor rhythm (SMR) brain-computer interface. Using 64-channel EEG data collected during motor imagery cursor control tasks, the pipelines classify left vs. right hand imagery and up vs. down (both hands vs. rest) from neural signals alone.

The goal: build an automated, data-driven decoder that matches or exceeds the performance of a manually calibrated online system.

### Key Results

| Pipeline | Task | Accuracy | Notes |
|----------|------|----------|-------|
| **CSP + LDA** | Left-Right | **82.2 – 90.8%** | Matched online baseline (~85–96%) within standard deviation |
| **EEGNet (CNN)** | Left-Right | **90.9%** | Matched highly-calibrated online performance using learned spatial-temporal features |
| Online Baseline | Left-Right | 85.5 – 96.3% | Fixed C3/C4 Laplacian + linear classifier |

<br>

<p align="center">
  <img src="assets/csp_feature_space.png" alt="CSP Feature Space Visualization" width="500"/>
  <br>
  <em>Left: Raw C3/C4 power shows heavy class overlap. Right: After CSP spatial filtering, left and right motor imagery classes separate into distinct clusters.</em>
</p>

---

## Pipelines

### Pipeline 1: CSP + LDA (Classical)

A lightweight, interpretable pipeline optimized for small-sample BCI datasets.

```
Raw EEG → Channel Selection → Bandpass Filter (8–13 Hz) → CAR → Downsample
    → Common Spatial Patterns (m=3 pairs) → Log-Variance Features → LDA → Classification
```

**Why CSP + LDA?** With ~135 trials per session, data-hungry models overfit. CSP learns spatial filters from covariance matrices with minimal data, and LDA provides a low-variance decision boundary. The pipeline is fully interpretable — spatial filter weights can be visualized on a scalp map to confirm physiological validity.

**Feature selection finding:** Reducing the electrode montage from 27 channels to 10 motor-cortex channels improved accuracy by ~1.5%. Peripheral electrodes (frontal, temporal) introduced artifact-driven variance that CSP mistakenly maximized, degrading generalization.

### Pipeline 2: EEGNet (Deep Learning)

A compact CNN architecture that learns spatial and temporal filters end-to-end.

```
Raw EEG → Channel Selection → Bandpass Filter (8–13 Hz) → CAR → Downsample
    → EEGNet (Temporal Conv → Depthwise Spatial Conv → Separable Conv) → Softmax
```

**Architecture:** Based on [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c), EEGNet uses depthwise separable convolutions to learn spatial filters (analogous to CSP) and temporal filters jointly, with far fewer parameters than standard CNNs. This makes it viable for small EEG datasets.

| Layer | Operation | Output Shape | Purpose |
|-------|-----------|--------------|---------|
| 1 | Temporal Conv2D (1→8, kernel 1×64) | (8, C, T) | Frequency filtering |
| 2 | Depthwise Conv2D (8→16, kernel C×1) | (16, 1, T/4) | Spatial filtering (learns CSP-like filters) |
| 3 | Separable Conv2D (16→16, kernel 1×16) | (16, 1, T/32) | Temporal pattern detection |
| 4 | Fully Connected | (2,) | Classification |

---

## Project Structure

```
.
├── CSP.py              # Common Spatial Patterns implementation (fit, transform, visualization)
├── LDA.py              # Full CSP+LDA pipeline with cross-validation
├── EEGNet.py           # EEGNet architecture (PyTorch)
├── runEEGNet.py        # Training loop, 5-fold CV, confusion matrix generation
├── assets/             # Figures and visualizations
│   ├── csp_feature_space.png
│   ├── electrode_montage.png
│   ├── confusion_matrix.png
│   └── performance_comparison.png
└── README.md
```

---

## Experimental Setup

**Data acquisition:** 64-channel EEG recorded at 1000 Hz using a Neuroscan system during a 1D cursor control BCI experiment. 27 electrodes were gelled over the sensorimotor cortex. Two sessions with different subjects, each performing 7 runs of Left-Right and 7 runs of Up-Down tasks (25 trials per run).

**Task paradigm:**
- **Left-Right:** Imagine left hand movement → cursor moves left; imagine right hand → cursor moves right
- **Up-Down:** Imagine both hands → cursor moves up; relax (rest) → cursor moves down
- **Trial structure:** Target presentation (2s) → Feedback period with cursor control (up to 6s) → Outcome (Hit / Miss / Abort)

**Preprocessing (both pipelines):**
1. **Channel selection** — 10 motor-cortex electrodes: FC3, FC4, C5, C3, C1, C2, C4, C6, CP3, CP4
2. **Common Average Reference (CAR)** — subtract mean across channels to remove global noise
3. **Bandpass filter** — 4th-order Butterworth, 8–13 Hz (mu band). Beta band excluded to avoid EMG contamination
4. **Downsampling** — 1000 Hz → 100 Hz (reduces computation, preserves mu-band information)
5. **Epoch extraction** — 0–4s window from feedback onset. Abort trials excluded

---

## Usage

### Requirements

```bash
pip install numpy scipy scikit-learn torch matplotlib
```

### Run the CSP + LDA pipeline

```bash
# Edit DATA_ROOT and TASK_TO_ANALYZE in LDA.py, then:
python LDA.py
```

Output:
```
Subject: session1 | Trials: 135 | Accuracy: 82.22% (+/- 6.37)
Subject: session2 | Trials: 148 | Accuracy: 90.81% (+/- 5.01)
```

### Run the EEGNet pipeline

```bash
# Edit DATA_ROOT and TASK_TO_ANALYZE in runEEGNet.py, then:
python runEEGNet.py
```

Output:
```
Fold 1: 88.57%
Fold 2: 91.43%
...
Saved as 'Figure4_ConfusionMatrix.png'
```

### Data format

The pipelines expect MATLAB `.mat` files with a `runData` struct containing:
- `allData` — EEG data matrix (channels × samples)
- `label` — channel labels
- `fs` — sampling rate
- `trialStart` — sample indices for trial onsets
- `target` — target class per trial (1 or 2)
- `outcome` — trial outcome (0 = abort, 1 = hit, 2 = miss)

---

## Results in Detail

### Electrode montage matters more than model complexity

| Montage | Channels | Session 1 (LR) | Session 2 (LR) |
|---------|----------|-----------------|-----------------|
| Broad | 17 | 80.7% | 90.3% |
| **Strict Motor** | **10** | **82.2%** | **90.8%** |

Removing frontal and temporal electrodes improved CSP performance by reducing artifact-driven covariance noise. This confirms that for small-sample BCI, **feature selection > model complexity**.

### CSP struggles with bilateral tasks

The Up-Down task (both hands vs. rest) showed a significant accuracy drop compared to the online baseline. CSP is designed to maximize *spatial* variance differences between classes. Left-Right imagery produces lateralized patterns (left hemisphere vs. right hemisphere) that CSP naturally separates. Both-hands vs. rest produces *bilateral symmetric* patterns where only amplitude changes — CSP has no spatial contrast to exploit.

| Task | Online PVC | Offline (CSP+LDA) | Why |
|------|-----------|-------------------|-----|
| Left-Right | 85–96% | 82–91% | ✅ Lateralized spatial patterns → CSP excels |
| Up-Down | 76–91% | 69–77% | ❌ Bilateral amplitude modulation → CSP fails |

### EEGNet compensates with learned temporal features

EEGNet achieved 90.9% on L-R tasks by jointly learning spatial and temporal filters, matching the performance ceiling of highly-calibrated online baselines. The depthwise convolution layer learns data-driven spatial filters analogous to CSP, while the temporal convolution captures frequency-specific dynamics that CSP+LDA ignores.

---

## CSP Implementation Details

The CSP algorithm finds spatial filters **W** by solving a generalized eigenvalue problem:

$$C_1 \mathbf{w} = \lambda (C_1 + C_2) \mathbf{w}$$

where $C_1$ and $C_2$ are the class covariance matrices estimated from concatenated trial data. Filters are ranked by $|\lambda - 0.5|$ — eigenvalues near 0 or 1 represent maximum class discrimination. The top $m$ pairs (default: 3) are selected, and the log-variance of the projected signal serves as the feature vector.

**Regularization:** A small ridge term ($\epsilon \cdot \text{trace}(C) / n \cdot I$) is added to covariance matrices to ensure positive definiteness, critical for the limited trial counts in BCI datasets.

---

## Future Directions

- **Riemannian geometry classifiers** — Replace CSP+LDA with Minimum Distance to Mean on the manifold of symmetric positive definite matrices. More robust to covariance non-stationarity across sessions
- **Transfer learning** — Pre-train EEGNet on large public MI-EEG datasets (e.g., BCI Competition IV 2a), then fine-tune on subject-specific data to address the small-sample problem
- **Hybrid features for Up-Down** — Combine CSP spatial features with spectral power density features to capture bilateral amplitude modulation that CSP alone misses
- **Artifact Subspace Reconstruction (ASR)** — Automated artifact rejection to replace manual channel selection, enabling use of all 27 electrodes without covariance contamination
- **Online adaptation** — Implement incremental CSP filter updates for real-time decoder recalibration during extended BCI sessions

---

## References

1. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Trans. Rehab. Eng.*, 8(4), 441–446.
2. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.*, 15(5), 056013.
3. Blankertz, B., et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56.
4. Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: A 10 year update. *J. Neural Eng.*, 15(3), 031005.
5. He, B. (Ed.). (2020). *Neural Engineering* (3rd ed.). Springer.

---

## Author

**Sofia Velasquez Sierra**
M.S. Biomedical Engineering, Carnegie Mellon University
B.Eng. Computer Engineering, McGill University

---

## License

This project is released under the [MIT License](LICENSE).
