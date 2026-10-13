# SemiVision AI
## Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift

**M.Tech Final-Year Research Project | Artificial Intelligence and Data Science**

**Research Status:** Phase 1 and Phase 2 implemented, evaluated, and integrated into the dashboard.

**Research Branch:** `research-improvements`

**Dashboard:** SemiVision AI — Semiconductor Defect Intelligence Platform

---

## 1. Project Overview

SemiVision AI is a deep-learning-based visual inspection and anomaly screening platform for semiconductor manufacturing.

The system analyzes Scanning Electron Microscopy (SEM) images to identify potential structural defects, calculate anomaly scores, generate defect-localization heatmaps, and support risk-aware screening decisions.

This research implements and comparatively evaluates two systems:

- **Phase 1 — Existing System:** A reference-trained Two-Dimensional Convolutional Autoencoder (2-D ConvAE) for reconstruction-based semiconductor anomaly detection.
- **Phase 2 — Proposed System:** Risk-Calibrated and Cost-Aware Convolutional Autoencoder (RCC-ConvAE), which extends the frozen Phase 1 model using robust score normalization, localized anomaly evidence, split-conformal calibration, and cost-sensitive decision selection.

Both phases use the Carinthia-S semiconductor SEM dataset.

The project additionally investigates defect localization, controlled distribution drift, calibration reliability, screening cost, and the limitations of anomaly detection under highly imbalanced data.

## 2. Research Problem

Semiconductor manufacturing requires reliable detection of surface abnormalities and microscopic pattern defects.

Conventional visual inspection and basic reconstruction-based anomaly detection face several challenges:

1. Small localized defects may be hidden when reconstruction errors are averaged across the entire image.
2. A high anomaly score does not directly indicate an appropriate screening decision.
3. Normal semiconductor structures can sometimes be classified as anomalous, increasing unnecessary review.
4. Distribution changes involving illumination, contrast, noise, and blur may affect anomaly scores.
5. Pixel-level defect localization may be inaccurate, particularly for very small defects.
6. Misclassification costs are asymmetric: missing an actual defect and unnecessarily flagging a reference image have different consequences.

This project investigates how a frozen reconstruction model can be extended with calibrated and cost-aware decision mechanisms while preserving reproducible evaluation.

### Research Question

**Can a reference-trained 2-D ConvAE be extended with calibrated anomaly scores and cost-sensitive screening decisions to improve semiconductor visual screening, while evaluating localization quality and reliability under distribution drift?**

## 3. Research Objectives

- Develop a 2-D ConvAE baseline using normal/reference SEM images.
- Detect anomalous semiconductor structures using reconstruction errors.
- Calculate global and localized anomaly evidence.
- Develop an RCC-ConvAE decision framework with robust normalization and conformal calibration.
- Select screening decisions based on asymmetric false-positive and false-negative costs.
- Generate pixel-level anomaly heatmaps and candidate defect masks.
- Evaluate Phase 1 and Phase 2 under the same recorded test split.
- Study defect-localization performance for tiny, small, medium, and large defects.
- Investigate sensitivity to synthetic distribution shifts.
- Build an interactive dashboard for single-image and batch semiconductor inspection.

## 4. Base Research Paper

**M. Gorman, X. Ding, L. Maguire, and D. Coyle**, "Anomaly Detection in Batch Manufacturing Processes Using Localized Reconstruction Errors From 1-D Convolutional AutoEncoders," *IEEE Transactions on Semiconductor Manufacturing*, vol. 36, no. 1, pp. 147–150, 2023.

DOI: https://doi.org/10.1109/TSM.2022.3216032

### Relationship to the Base Paper

The base paper investigates localized reconstruction errors using a **1-D convolutional autoencoder** for semiconductor manufacturing process data.

Its reported AUC of 1.00 on its LAM 9600 experiment is a result from that paper's own dataset and protocol; it is not equivalent to 100% classification accuracy.

Our research extends the underlying idea to **2-D semiconductor SEM images**, adds a calibrated decision framework, and evaluates both image-level anomaly screening and pixel-level defect localization.

The base paper and the current Carinthia-S experiments use different input modalities and datasets. Their performance scores are not directly comparable.

---

## 5. Dataset — Carinthia-S

The project uses the Carinthia-S dataset containing semiconductor SEM images and corresponding ground-truth segmentation masks.

### Verified Dataset Information

| Property | Verified value |
|---|---|
| Dataset | Carinthia-S |
| Total SEM images | 4,591 |
| Segmentation masks | 4,591 |
| Native image resolution | 480 × 480 |
| Image type | Grayscale |
| Images with empty masks | 226 |
| Images with nonempty masks | 4,365 |
| Supplied dataset categories | 6 |
| Ground-truth source | Segmentation masks |

### Ground-Truth Labeling

The project determines the binary screening label from the corresponding segmentation mask.

- **REFERENCE / NORMAL:** The ground-truth mask contains no defect pixels.
- **ANOMALY / DEFECT:** The ground-truth mask contains one or more annotated defect pixels.

The original category identifiers are retained for dataset analysis, but the implemented screening problem is binary anomaly detection rather than six-class defect classification.

### Dataset Partition

| Partition | Reference | Anomaly | Total |
|---|---:|---:|---:|
| Training | 115 | 0 | 115 |
| Validation | 34 | 873 | 907 |
| Calibration | 34 | 0 | 34 |
| Policy validation | 23 | 873 | 896 |
| Final test | 20 | 2,619 | 2,639 |
| **Total** | **226** | **4,365** | **4,591** |

The partitions have different purposes. Training uses reference images to learn normal structure, calibration supports the conformal reference-score distribution, policy validation is used to select operating parameters, and the test partition provides the recorded final evaluation.

### Dataset Integrity

The project includes:

- Dataset and segmentation-mask verification.
- Image and CSV consistency checks.
- Recorded partition manifests.
- Exact SHA-256 duplicate checking.
- Perceptual and difference-hash similarity analysis.
- Near-duplicate pixel-comparison analysis.

The latest audit found 4,591 unique exact image hashes and no exact cross-split duplicates. However, it also identified highly similar reference images across some partitions. Such similarity is not proof of data leakage, but it remains a possible independence limitation requiring acquisition-level verification.

**Dataset source:** https://doi.org/10.5281/zenodo.16895427

Raw SEM images and masks must be obtained separately and arranged according to `DATASET_SETUP.md`.

---

# 6. Phase 1 — Existing System: 2-D ConvAE

## 6.1 Objective

Phase 1 establishes a reconstruction-based visual anomaly detection baseline using a Two-Dimensional Convolutional Autoencoder.

The model learns normal semiconductor surface patterns from reference SEM images without using anomalous images to train the reconstruction backbone.

## 6.2 Phase 1 Architecture

**Input SEM Image → Preprocessing → 2-D ConvAE Encoder → Latent Representation → Decoder → Reconstructed Image → Reconstruction Error → Global Anomaly Score → Reference Threshold → Screening Decision**

### Module Description

**Input and Preprocessing**

A grayscale SEM image is loaded, resized to the configured model input dimensions, normalized, and converted into a tensor.

**Encoder**

Two-dimensional convolutional layers extract hierarchical spatial features representing semiconductor textures, patterns, and surface structures.

**Latent Representation**

The compressed representation captures features learned from normal reference images.

**Decoder**

The decoder reconstructs the input using the learned reference-pattern representation.

**Reconstruction Error**

The difference between the input and reconstructed image provides pixel-level anomaly evidence.

**Global Anomaly Score**

The model averages the pixelwise squared reconstruction errors to produce one image-level score.

**Decision Threshold**

A threshold derived from reference calibration data determines whether the image is classified as reference or anomalous.

## 6.3 Phase 1 Mathematical Formulation

For input image `x` and reconstructed image `x_hat`:

**Pixelwise reconstruction error:**

`E(i,j) = (x(i,j) - x_hat(i,j))²`

**Global anomaly score:**

`S_global = (1 / N) × Σ E(i,j)`

where `N` is the number of pixels.

The Phase 1 screening decision is obtained by comparing the global reconstruction score with the stored reference threshold.

## 6.4 Phase 1 Implementation

Phase 1 includes:

- Normal-reference image selection.
- SEM preprocessing.
- 2-D convolutional autoencoder training.
- Reconstruction-loss monitoring.
- Frozen model checkpoint.
- Image-level anomaly scoring.
- Reference-based threshold selection.
- Binary anomaly predictions.
- Residual heatmaps and exploratory localization.
- Evaluation against recorded ground-truth labels and masks.

The implemented ConvAE checkpoint contains **1,377,985 parameters**.

## 6.5 Phase 1 Limitations

The baseline has several limitations:

- Global averaging can dilute tiny localized defects.
- Reconstruction artifacts may trigger false alarms.
- A reconstruction score alone does not explain screening risk.
- Its threshold may become unreliable under acquisition changes.
- Pixel-level localization overlap can be limited even when image-level detection is strong.

These limitations motivated the Phase 2 decision framework.

---

# 7. Phase 2 — Proposed System: RCC-ConvAE

## 7.1 Objective

Phase 2 develops the **Risk-Calibrated and Cost-Aware Convolutional Autoencoder** framework.

Instead of retraining the reconstruction model, Phase 2 reuses the frozen Phase 1 ConvAE and adds decision-processing and evaluation modules.

The main objective is to convert reconstruction evidence into more interpretable, reference-calibrated and cost-aware screening decisions.

## 7.2 Phase 2 Architecture

**SEM Image → Preprocessing → Frozen 2-D ConvAE → Reconstruction Error Map → Global and Local Evidence → Robust Normalization → Score Fusion → Conformal Calibration → Cost-Aware Decision → Screening Report**

A separate localization branch processes the reconstruction error map to produce a heatmap and candidate defect mask.

## 7.3 Proposed System Modules

### Module 1 — Frozen ConvAE Backbone

The pretrained Phase 1 ConvAE is reused without modifying its learned weights.

This enables controlled comparison between the baseline reconstruction model and the proposed decision framework.

### Module 2 — Global Anomaly Evidence

The mean squared reconstruction error is computed over the complete SEM image.

`S_global = mean(E)`

It measures overall image-level reconstruction deviation.

### Module 3 — Localized Top-k Evidence

The local score focuses on the largest reconstruction-error pixels rather than averaging the entire image.

`S_local = mean(TopK(E, k))`

For the saved experiment:

`k = 0.001`

This provides additional information about concentrated reconstruction errors.

### Module 4 — Robust Score Normalization

Global and local scores may have different numerical ranges.

Reference-fitted robust normalization uses the median and median absolute deviation (MAD):

`Z = (S - median_reference) / (1.4826 × MAD_reference + ε)`

where `ε` provides numerical stability.

### Module 5 — Anomaly Score Fusion

The global and local normalized scores are combined using:

`S_fused = Z_global + λ × Z_local`

The fusion weight is selected using development data.

**Important implemented result:** The current saved policy selected `λ = 0.0`.

Therefore, the local score is calculated and displayed as supplementary evidence, but the deployed fused decision score effectively depends on normalized global evidence.

The current results do not establish an incremental image-detection improvement from local fusion.

### Module 6 — Split-Conformal Calibration

The framework compares the new fused score against calibration scores from 34 normal reference images.

The conformal anomaly p-value is:

`p(x) = (1 + number of calibration scores >= S_fused(x)) / (n + 1)`

where `n = 34`.

The smallest attainable p-value is:

`1 / 35 = 0.0285714`

A conformal p-value is **not** the probability that an image has a defect.

Its statistical interpretation depends on the new reference image and calibration reference images satisfying the required exchangeability conditions.

### Module 7 — Cost-Aware Decision Selection

Different prediction errors are assigned different experimental costs:

`Cost = (10 × FN) + (1 × FP)`

where:

- `FN` = actual anomalies incorrectly classified as reference.
- `FP` = reference images incorrectly classified as anomaly.

The cost weights are experimental assumptions rather than measured industrial financial costs.

The decision threshold alpha is selected using policy-validation data.

**Saved policy:**

`alpha = 0.05`

**Final decision rule:**

- If `p(x) <= 0.05`, classify as **ANOMALY**.
- Otherwise, classify as **REFERENCE**.

### Module 8 — Pixel-Level Defect Localization

The original reconstruction-error map is used to generate spatial evidence.

A pixel is marked anomalous when:

`E(i,j) >= localization_threshold`

The interface displays the error heatmap and thresholded candidate mask.

Localization is evaluated against the provided expert segmentation masks using Dice, IoU, pixel AUROC, and pixel average precision.

### Module 9 — Distribution-Drift Evaluation

The frozen detection system is investigated under controlled image transformations.

Studied conditions include:

- Clean image.
- Brightness +0.10.
- Brightness −0.10.
- Contrast 0.70.
- Gamma 1.50.
- Gaussian noise 0.02.
- Gaussian noise 0.05.
- Gaussian blur 1.0.
- Gaussian blur 2.0.

The experiments compare frozen and reference-only recalibrated policies. A separate development experiment investigates whether disagreements between those policies should trigger additional review.

### Module 10 — Screening Dashboard

All scoring and reporting components are integrated into the SemiVision AI interface.

The dashboard provides screening decisions together with anomaly evidence, reference-calibrated p-values, thresholds, heatmaps and research metrics.

---

# 8. Phase 1 vs Phase 2 — Verified Comparative Results

The following table reports the project's **current recorded held-out evaluation**, performed using the same 2,639-image test partition for both systems.

| Evaluation Metric | Phase 1: ConvAE | Phase 2: RCC-ConvAE |
|---|---:|---:|
| ROC-AUC | 0.997747 | 0.997747 |
| PR-AUC | 0.999982 | 0.999982 |
| Accuracy | 0.998863 | 0.999242 |
| Balanced accuracy | 0.9250 | 0.9500 |
| Precision | 0.998856 | 0.999237 |
| Recall | 1.0000 | 1.0000 |
| F1-score | 0.999428 | 0.999618 |
| Specificity | 0.8500 | 0.9000 |
| False-positive rate | 0.1500 | 0.1000 |
| False-negative rate | 0.0000 | 0.0000 |
| True negatives | 17 | 18 |
| False positives | 3 | 2 |
| False negatives | 0 | 0 |
| True positives | 2,619 | 2,619 |
| Cost per image | 0.001137 | 0.000758 |
| Localization Dice | 0.239342 | 0.239342 |
| Localization IoU | 0.143173 | 0.143173 |
| Pixel AUROC | 0.900238 | 0.900238 |
| Pixel average precision | 0.477758 | 0.477758 |

### Interpretation

Phase 2 achieves a slightly better recorded image-level operating point, reducing false positives from 3 to 2 while preserving zero observed false negatives on the test partition.

The corresponding experimental cost per image decreases by approximately 33.3%.

However:

- Both systems have identical image ROC-AUC and PR-AUC.
- Both use the same frozen reconstruction backbone.
- The selected fusion weight is zero.
- Original held-out localization performance is identical.
- The test contains only 20 normal reference images.

Therefore, the principal observed improvement is attributable to the **screening decision policy**, not to an independently demonstrated improvement in reconstruction, image-level ranking, or original test localization.

These findings should not be interpreted as proof of superior real-factory performance.

---

# 9. Additional Completed Research Experiments

## 9.1 Fusion-Weight Ablation

The framework evaluated several candidate fusion weights:

`λ = {0, 0.25, 0.5, 1.0, 1.5, 2.0}`

The development diagnostic used 896 policy-validation images and 34 calibration reference images.

| Result | Finding |
|---|---|
| Global/local Pearson correlation | 0.773706 |
| Global/local Spearman correlation | 0.831745 |
| Diagnostic ROC-AUC | 1.000 for
