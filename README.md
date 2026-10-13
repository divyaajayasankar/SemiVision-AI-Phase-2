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
| Diagnostic ROC-AUC | 1.000 for all tested weights |
| Diagnostic PR-AUC | 1.000 for all tested weights |
| False positives | 2 for all tested weights |
| False negatives | 0 for all tested weights |
| Selected fusion weight | 0.0 |

The evaluation demonstrated that including additional local evidence did not change the tested binary decisions.

The saved RCC-ConvAE implementation therefore retains `λ = 0.0`, rather than claiming a fusion improvement unsupported by the experimental results.

## 9.2 Spatial Smoothing Evaluation

Three localization methods were evaluated over 907 development-validation images.

| Localization Method | Mean Dice | Mean IoU |
|---|---:|---:|
| Raw reconstruction error | 0.225609 | 0.133694 |
| 3×3 smoothing | 0.161689 | 0.092901 |
| 5×5 smoothing | 0.117140 | 0.065684 |

The raw reconstruction-error map performed better than both smoothing alternatives.

Therefore, spatial smoothing was not selected as the improved localization method.

## 9.3 Defect-Size Localization Analysis

Raw-error localization performance was analyzed according to annotated defect-mask coverage.

| Defect Size | Images | Mean Dice |
|---|---:|---:|
| Tiny — below 0.5% | 72 | 0.0511 |
| Small — 0.5% to 2% | 444 | 0.1828 |
| Medium — 2% to 10% | 332 | 0.3133 |
| Large — at least 10% | 25 | 0.6305 |

The results demonstrate that tiny semiconductor defects are considerably harder to localize than large defect regions.

This finding motivated a reference-calibrated pixel-threshold sensitivity experiment.

## 9.4 Improved Localization Threshold — Development Candidate

Six reference-based threshold quantiles were evaluated:

`0.970, 0.980, 0.990, 0.995, 0.997, 0.999`

The strongest Dice result among the tested candidates was obtained at the 0.999 quantile.

| Metric | Original 0.995 | Candidate 0.999 |
|---|---:|---:|
| Mean anomaly Dice | 0.234396 | 0.305197 |
| Mean anomaly IoU | 0.138901 | 0.187597 |
| Tiny-defect Dice | 0.051079 | 0.091668 |
| Mean defect-pixel recall | 0.826392 | 0.786948 |
| Mean reference pixel fraction falsely highlighted | 0.008958 | 0.002974 |

The candidate improves mean anomaly Dice by approximately 30.2% relative to the original quantile, but decreases pixel-level recall.

**Research status:** The candidate is saved as an experimental, development-selected localization configuration.

It has **not** replaced the original deployed Phase 2 threshold and has not been independently confirmed on a fresh untouched test partition.

## 9.5 Distribution-Drift and Review-Policy Analysis

Synthetic distribution shifts were used to investigate the stability of calibrated screening decisions.

The development-only disagreement-review experiment produced the following important findings:

| Condition | Recalibrated Automatic FN | Images Referred for Review | Review Rate |
|---|---:|---:|---:|
| Clean | 0 | 0 | 0% |
| Brightness +0.10 | 651 | 674 | 75.22% |
| Gamma 1.50 | 106 | 129 | 14.40% |
| Blur 2.0 | 229 | 252 | 28.13% |

Recalibration was not consistently beneficial. Under some synthetic shifts, it reduced reference false alarms but introduced substantial missed-defect risk.

Disagreement-triggered review could prevent certain automatic decisions from being treated as final, but potentially requires reviewing a large proportion of images.

The experimental review-cost calculation assumes successful human review and is not an observed industrial deployment result.

**Research finding:** Reference-only recalibration and human-review policies must be evaluated for both screening errors and operational review burden. Neither approach should be assumed to improve robustness automatically.

---

# 10. Research Contributions

The main contributions of this project are:

### Contribution 1 — Reference-Trained Visual Screening

An implemented 2-D convolutional autoencoder trained on normal semiconductor SEM structures, providing an internal anomaly-screening baseline.

### Contribution 2 — Risk-Calibrated Decision Framework

An RCC-ConvAE extension that transforms reconstruction scores into rank-based conformal p-values using separately maintained reference-calibration data.

### Contribution 3 — Cost-Aware Screening

An explicit empirical screening policy selected under asymmetric false-negative and false-positive costs.

### Contribution 4 — Localized Anomaly Evidence

Global and top-k local reconstruction-error evidence, supported by interpretable SEM residual heatmaps and candidate masks.

### Contribution 5 — Defect-Size and Threshold Analysis

An investigation of localization performance across different defect sizes, including a development-validated pixel-threshold candidate.

### Contribution 6 — Distribution-Drift and Review Analysis

Controlled evaluation of acquisition-like image corruptions, reference recalibration, missed-defect risk, and disagreement-based review burden.

### Contribution 7 — Reproducible Research Pipeline

Recorded model checkpoints, split manifests, experimental results, metrics, ablations, dataset audits, reproducible scripts, and application-level tests.

These contributions describe an integrated research framework. They do not constitute a newly invented convolutional architecture or a new mathematical conformal guarantee.

---

# 11. SemiVision AI Dashboard

The Streamlit-based Semiconductor Defect Intelligence Platform integrates the implemented screening pipeline with an interactive interface.

### Dashboard Features

**Project Overview**

Introduces the semiconductor screening problem, architecture, datasets, research objectives, and key findings.

**Single-Image Screening**

Allows an individual SEM image to be analyzed using the frozen RCC-ConvAE model and saved decision policy.

**Batch Screening**

Provides screening for multiple SEM images with consolidated results and CSV export support.

**Image Reconstruction and Localization**

Displays reconstruction-based error heatmaps and candidate defect masks.

**Anomaly Evidence**

Displays:

- Global anomaly score.
- Local top-k anomaly evidence.
- Fused anomaly score.
- Conformal p-value.
- Decision threshold alpha.
- Final REFERENCE / ANOMALY classification.

**Research Metrics**

Presents saved Phase 1 and Phase 2 comparison results.

**Distribution-Drift Analysis**

Displays saved experimental findings for image corruption and calibration policies.

**Ablation Analysis**

Presents fusion, threshold, localization, and selected diagnostic comparisons.

**Methodology and Integrity**

Documents training/calibration separation, experiment settings, limitations, and result provenance.

### Example Live Screening Output

| Output | Example value |
|---|---:|
| Global anomaly score | 0.000328536 |
| Local anomaly evidence | 0.0619177 |
| Fused anomaly score | 74.2722 |
| Conformal p-value | 0.0285714 |
| Decision alpha | 0.05 |
| Final screening decision | ANOMALY |

The decision follows:

`0.0285714 <= 0.05`

Therefore, the system classifies the image as **ANOMALY**.

The fused score and alpha use different scales and must not be directly compared with one another.

---

# 12. Project Structure

The repository is organized around the following components:

| Directory / File | Purpose |
|---|---|
| `src/` | Core dataset handling, ConvAE, scoring, calibration, localization, and policy algorithms |
| `experiments/` | Training, evaluation, ablation, calibration, drift, and localization experiments |
| `models/` | Locally generated model checkpoints and saved screening policies |
| `results/phase1/` | Recorded Phase 1 evaluation results |
| `results/phase2/` | Recorded Phase 2 evaluation results |
| `results/drift/` | Distribution-drift experiment outputs |
| `results/dataset/` | Dataset split and integrity manifests |
| `reports/` | Diagnostic results, comparative tables, and localization analyses |
| `app/` | Streamlit screening dashboard |
| `tests/` | Automated tests and integrity checks |
| `scripts/` | Project utilities and packaging helpers |
| `paper/` | Research manuscript and supporting paper materials, where included |
| `requirements.txt` | Python dependencies |
| `run_all.py` | Main research experiment runner |

### Selected Research Artifacts

| Artifact | Description |
|---|---|
| `models/convae_best.pt` | Frozen 2-D ConvAE checkpoint, generated locally |
| `models/phase1_policy.json` | Phase 1 screening settings |
| `models/phase2/rcc_policy.json` | Phase 2 calibrated screening policy |
| `results/phase1/metrics.json` | Saved Phase 1 evaluation |
| `results/phase2/metrics.json` | Saved Phase 2 evaluation |
| `reports/fusion_diagnostics_summary.json` | Fusion diagnostic |
| `reports/drift_review_development/policy_comparison.csv` | Drift review-policy comparison |
| `reports/localization_development/method_comparison.csv` | Raw versus smoothed localization |
| `reports/localization_threshold_development/candidate_policy_0999.json` | Frozen experimental localization candidate |
| `reports/phase1_vs_phase2_verified.csv` | Verified comparative-results table |

The large dataset files and model binaries are not necessarily distributed through GitHub. Their presence depends on the local setup or a separately provided project package.

---

# 13. Installation and Execution

## Requirements

- Windows or another supported Python development environment.
- Python and pip.
- PyTorch and project dependencies.
- Carinthia-S dataset.
- Sufficient CPU/GPU resources and storage for training and evaluation.
- Streamlit for the interface.

## Step 1 — Clone the Repository

```powershell
git clone https://github.com/divyaajayasankar/SemiVision-AI-Phase-2.git
cd SemiVision-AI-Phase-2
git checkout research-improvements
```

## Step 2 — Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Step 3 — Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Step 4 — Prepare the Dataset

Follow `DATASET_SETUP.md` and place the SEM images, segmentation masks and accompanying CSV into the expected dataset folders.

Typical local dataset structure:

```text
data/
  raw/
    carinthia_s/
      data/
        images/
        masks/
        carinthia-s.csv
```

The source split manifest may contain paths from the original development machine. Some later experimental scripts resolve dataset paths using local image filenames. Do not overwrite the official split assignments when adapting file paths.

## Step 5 — Verify the Dataset

```powershell
python experiments\01_verify_dataset.py
```

Verify the dataset counts and file structure before training or evaluation.

## Step 6 — Run the Research Pipeline

```powershell
python run_all.py
```

**Important:** The main research pipeline may train models or regenerate saved outputs, depending on its configured execution stages.

The existing verified result files should be backed up before rerunning experiments. To review completed research, use the saved reports rather than unnecessarily retraining the model.

## Step 7 — Run Automated Tests

```powershell
python -m pytest -q
```

The latest verified repository run reported:

**7 tests passed.**

## Step 8 — Launch SemiVision AI

```powershell
python -m streamlit run app\streamlit_app.py
```

Open the local URL displayed by Streamlit, commonly:

`http://localhost:8501`

Live screening requires the locally available trained checkpoint and saved screening-policy files.

---

# 14. Evaluation Metrics

### Image-Level Detection

- ROC-AUC.
- PR-AUC.
- Accuracy.
- Balanced accuracy.
- Precision.
- Recall.
- Specificity.
- F1-score.
- False-positive rate.
- False-negative rate.
- Confusion matrix: TN, FP, FN, TP.

### Screening Cost

`Cost per image = (10 × FN + FP) / Number of evaluated images`

### Localization

- Dice coefficient.
- Intersection over Union.
- Sampled pixel ROC-AUC.
- Pixel average precision.
- Defect-size-stratified Dice and IoU.
- Reference-image false-highlight fraction.

### Distribution Drift

- Frozen-policy FPR and FNR.
- Recalibrated-policy FPR and FNR.
- Balanced accuracy.
- Cost per image.
- Review rate.
- Automatic errors remaining after deferral.

All reported metrics should identify whether they originate from the original test partition, a development-validation experiment, or a historical reference run.

---

# 15. Historical Reference Run — Separate Experimental Configuration

Earlier project materials included a historical reference run with different model settings, partitions and results.

These records are retained for provenance but must **not** be combined with the current 2,639-image test protocol.

| Metric | Historical Phase 1 | Historical Phase 2 |
|---|---:|---:|
| Image ROC-AUC | 0.9971 | 0.9971 |
| Image PR-AUC | 0.9998 | 0.9998 |
| F1-score | 0.9989 | 0.9989 |
| Balanced accuracy | 0.9886 | 0.9886 |
| FPR | 0.0217 | 0.0217 |
| FNR | 0.0011 | 0.0011 |
| Localization Dice | 0.4142 | 0.5381 |
| Localization IoU | 0.2704 | 0.3795 |
| Pixel AUROC | 0.9450 | 0.9860 |

The historical RCC configuration also reported:

- `top-k = 0.001`
- `lambda = 1.0`
- `alpha = 0.075`

These are **not the parameters of the latest verified RCC-ConvAE run**.

The latest verified configuration is:

- `top-k = 0.001`
- `lambda = 0.0`
- `alpha = 0.05`

Historical figures are provided only to document earlier experimental development. The primary results of this README are the latest recorded Phase 1/Phase 2 comparison.

---

# 16. Research Limitations

The project is an academic research prototype with the following limitations.

**Small Reference-Test Population**

Only 20 reference images appear in the recorded test partition. One additional false positive changes FPR by five percentage points.

**Strong Class Imbalance**

The test partition contains considerably more anomalous than reference images. Accuracy and anomaly-positive PR-AUC should be interpreted alongside balanced accuracy, specificity and the full confusion matrix.

**Limited Conformal Resolution**

Only 34 reference calibration examples are used, leading to a minimum conformal p-value of approximately 0.028571.

**Potential Split Dependence**

Near-identical SEM reference images may occur in different partitions. Acquisition, wafer and process-lot identifiers would strengthen split validation.

**No Demonstrated Incremental Fusion Benefit**

The final selected fusion coefficient is zero. Additional local evidence is available for interpretation but has not independently improved current image-ranking performance.

**Localization Challenges**

The original system has limited overlap with very small annotated defect regions.

**Unconfirmed Improved Threshold**

The 0.999 localization candidate was selected on development data and requires a new independent confirmation dataset.

**Synthetic Rather Than Factory Drift**

Brightness, noise, contrast, gamma and blur transformations cannot represent every manufacturing or SEM acquisition change.

**Experimental Cost Weights**

The 10:1 FN/FP ratio and human-review cost assumptions are not directly measured financial outcomes.

**Deployment Scope**

The dashboard demonstrates image screening and decision support. It is not a production-qualified semiconductor fab inspection controller.

---

# 17. Conclusion

This project completes the development of a **two-stage semiconductor visual anomaly screening system**.

Phase 1 establishes a normal-reference-trained 2-D convolutional autoencoder for SEM anomaly detection.

Phase 2 introduces RCC-ConvAE, integrating robust anomaly-score normalization, localized evidence, conformal calibration and cost-sensitive decision selection on the frozen reconstruction backbone.

The current verified evaluation demonstrates a reduction from three to two false positives, with zero observed false negatives, while maintaining identical image-level ROC-AUC between both phases.

Additional experiments reveal meaningful limitations and research opportunities in small-defect localization, conformal calibration, distribution drift and review-policy design.

An experimental pixel-threshold adjustment improves validation anomaly Dice from 0.2344 to 0.3052, subject to reduced recall and the requirement for independent confirmation.

The final deliverable is an integrated research and dashboard framework designed to support reproducible SEM anomaly screening, transparent decisions, and further evaluation on independently collected semiconductor data.

## 18. Future Work

Future development can include:

- Independent SEM image datasets and acquisition-disjoint evaluation.
- Same-split comparisons against PatchCore, PaDiM and other published visual anomaly detectors.
- Improved defect localization for extremely small structures.
- Larger and more representative reference calibration sets.
- Drift-detection and carefully controlled recalibration.
- Human-review experiments with measured time and accuracy.
- Real industrial cost estimates.
- Semiconductor equipment and manufacturing-system integration.
- Expanded model-uncertainty analysis and confidence intervals.

---

## 19. Key References

1. M. Gorman, X. Ding, L. Maguire, and D. Coyle, "Anomaly Detection in Batch Manufacturing Processes Using Localized Reconstruction Errors From 1-D Convolutional AutoEncoders," *IEEE Transactions on Semiconductor Manufacturing*, 2023. https://doi.org/10.1109/TSM.2022.3216032

2. P. Bergmann et al., "MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," *CVPR*, 2019.

3. T. Defard et al., "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization," *ICPR Workshops*, 2021.

4. K. Roth et al., "Towards Total Recall in Industrial Anomaly Detection," *CVPR*, 2022.

5. S. Bates et al., "Testing for Outliers with Conformal p-Values," *Annals of Statistics*, 2023.

6. C. Elkan, "The Foundations of Cost-Sensitive Learning," *IJCAI*, 2001.

7. Carinthia-S Dataset, Zenodo. https://doi.org/10.5281/zenodo.16895427

---

**Project:** SemiVision AI — Semiconductor Defect Intelligence Platform

**Research:** Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift

**Implementation:** Python, PyTorch, Scikit-learn, NumPy, pandas, and Streamlit

**Status:** Phase 1 and Phase 2 completed; development experiments, dashboard, and automated tests verified.

**Purpose:** Academic research, semiconductor anomaly-screening experimentation, and reproducible methodological evaluation.