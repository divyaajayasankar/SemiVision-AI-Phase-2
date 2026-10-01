# Risk-Calibrated and Cost-Aware Anomaly Detection Visual Screening for Semiconductor Manufacturing Under Distribution Drift

**[Author Name]**, Department of [Department], [Institution], [City, Country], [Email]

## Abstract
Scanning electron microscope (SEM) inspection is an important stage in semiconductor manufacturing because visually subtle structural defects can be associated with yield loss, reliability concerns, and additional engineering review. However, supervised defect classifiers are constrained by the rarity, imbalance, and continual emergence of failure modes, while a single anomaly score does not by itself provide calibrated risk, spatial evidence, or an operating policy that reflects asymmetric manufacturing costs. This work presents a two-phase visual screening framework for semiconductor SEM images. Phase 1 establishes a reference-only two-dimensional convolutional autoencoder (2-D ConvAE) baseline in which defect-free structures are learned through reconstruction and global reconstruction error is used for anomaly screening. Phase 2 introduces RCC-ConvAE, a risk-calibrated and cost-aware decision framework built on the frozen Phase-1 autoencoder. The proposed framework combines global reconstruction error with localized top-k reconstruction evidence, robust score normalization, split-conformal calibration, cost-aware operating-point selection, pixel-level localization, and distribution-drift stress testing. Evaluation is performed on Carinthia-S, which contains 4,591 SEM images paired with segmentation masks. In the archived reference run, the Phase-1 baseline achieved ROC-AUC 0.9971, PR-AUC 0.9998, F1 0.9989, and balanced accuracy 0.9886. Phase 2 preserved the strong image-level performance while increasing localization Dice from 0.4142 to 0.5381, IoU from 0.2704 to 0.3795, and pixel AUROC from 0.9450 to 0.9860. Under controlled distribution shifts, reference-only recalibration increased mean balanced accuracy from 0.6805 to 0.9331 and reduced mean false-positive rate from 0.6332 to 0.1033. The results indicate that the principal value of the proposed framework lies in reliable localization, calibrated decision support, and operating-point robustness rather than an artificial claim of improvement over an already saturated image-level ranking metric.

**Index Terms—** semiconductor manufacturing, SEM inspection, anomaly detection, convolutional autoencoder, conformal prediction, cost-sensitive learning, defect localization, distribution drift.

## I. INTRODUCTION AND LITERATURE REVIEW

Semiconductor manufacturing involves a sequence of tightly controlled physical and chemical processes in which deviations at a single stage can propagate into functional or yield-related failures. Automated optical and electron-microscopy inspection therefore plays a central role in screening structures that are too numerous for exhaustive manual evaluation. Scanning electron microscopy is particularly useful for resolving fine structural and surface details, but high-resolution inspection also creates an analysis problem: defects may be visually small, heterogeneous, rare, and distributed unevenly across production conditions. A screening algorithm must therefore do more than assign a class label. It should detect samples that deviate from known-good structure, identify where the deviation is located, and provide decision evidence that can be interpreted under operational risk.

Conventional supervised image classification is attractive when abundant and stable labels are available, yet semiconductor defect taxonomies frequently violate this assumption. New or uncommon anomalies may have few training examples, class frequencies may be severely imbalanced, and the operational concept of “abnormal” may extend beyond the set of defect categories that was available during model development. Reference-only anomaly detection addresses this limitation by learning the appearance of acceptable material and treating departures from the learned reference manifold as potentially anomalous. In this setting, autoencoders are useful because they compress an input into a latent representation and reconstruct it from the learned representation. When training is restricted to reference samples, the reconstruction error becomes a nonconformity signal: structures consistent with the learned reference distribution should generally reconstruct more accurately than structures containing previously unseen local defects.

Gorman et al. [1] demonstrated the value of this principle for semiconductor batch-manufacturing time-series data. Their method employed a one-dimensional convolutional autoencoder and localized reconstruction error (LRE) to improve anomaly detection and provide interpretable information about the variables contributing to an anomalous batch. The reported LAM 9600 experiment achieved an AUC of 1.00 [1]. Importantly, that result is an area-under-the-curve value and must not be interpreted as “100% accuracy.” It was also obtained on a different sensing modality and dataset; consequently, it is used in this work as a methodological foundation rather than as a directly comparable benchmark target.

The broader visual-anomaly literature supports the use of reference-only learning for industrial inspection. Bergmann et al. [4] introduced MVTec AD as a real-world benchmark in which defect-free images are used for training and pixel-accurate anomaly annotations are provided for evaluation. Their work formalized the importance of evaluating not only sample-level discrimination but also localization quality in industrial anomaly detection. Ruff et al. [5] developed Deep SVDD as an alternative one-class objective, further illustrating that anomaly detection can be formulated without requiring balanced examples of all abnormal classes. These approaches motivate a semiconductor inspection design in which abnormality is defined relative to the learned reference distribution instead of only through a finite defect-class vocabulary.

A second concern is uncertainty and calibration. A raw neural-network or reconstruction score has no universal probabilistic interpretation. Guo et al. [7] showed that predictive scores produced by modern neural networks can be poorly calibrated, while Angelopoulos and Bates [6] presented conformal prediction as a model-agnostic framework for distribution-free uncertainty quantification under suitable exchangeability conditions. Although conformal prediction is often discussed for prediction sets and intervals, the same ranking principle can be applied to anomaly nonconformity scores by comparing a new score with scores observed on reference calibration samples. This produces an empirical p-value that has a clearer operational interpretation than an arbitrary reconstruction threshold.

Deployment reliability becomes more difficult when the input distribution changes. Ovadia et al. [8] demonstrated that predictive uncertainty and calibration can deteriorate under dataset shift. For SEM images, shifts may be produced by changes in brightness, contrast, imaging conditions, noise, focus, or preprocessing. A detector that retains a strong ranking metric but uses a frozen operating threshold can therefore generate an unacceptable false-alarm rate under distribution drift. This observation motivates explicit stress testing of decision thresholds and recalibration procedures, rather than reporting only clean-test ROC-AUC.

The dataset used in the present work is Carinthia-S. The dataset provides 4,591 SEM images paired with expert-validated segmentation masks and is unevenly distributed across six defect classes [2], [3]. The segmentation masks are central to the proposed methodology because they permit a ground-truth definition that is independent of an arbitrary class identifier: an empty mask is treated as a reference/normal candidate, while a non-empty mask indicates a defect. The same masks provide pixel-level ground truth for evaluating whether reconstruction evidence actually points to the spatial location of the defect.

The research gap addressed by this work is therefore not the absence of autoencoders. Reconstruction-based anomaly detectors are well established. The gap is the absence, in a single semiconductor SEM screening workflow, of a controlled combination of (i) a reference-only 2-D reconstruction baseline, (ii) localized reconstruction evidence, (iii) calibrated nonconformity, (iv) asymmetric cost-aware operating-point selection, (v) pixel-level localization validation, and (vi) explicit distribution-drift and recalibration analysis. The proposed RCC-ConvAE framework is designed to fill this gap without incorrectly representing the decision layer as a completely new neural-network architecture.

The contributions of this work are fourfold. First, a reproducible 2-D ConvAE baseline is established for reference-only anomaly screening of Carinthia-S SEM images. Second, the frozen reconstruction model is extended with a global-local evidence fusion mechanism and reference-only conformal calibration. Third, a cost-aware policy is introduced to select the anomaly operating point according to asymmetric false-negative and false-positive costs. Fourth, the framework is evaluated at both image and pixel levels and is stress-tested under controlled distribution shifts with and without reference-only recalibration.

## II. PROPOSAL AND EXISTING SYSTEM ANALYSIS

### A. Existing System: Environment and Setup

The principal existing system used as methodological context is the framework of Gorman et al. [1]. That work considered multivariate semiconductor batch-manufacturing time series rather than SEM imagery. A one-dimensional convolutional autoencoder was trained in an unsupervised setting and reconstruction error was used to identify anomalous behavior. The method introduced localized reconstruction error to associate anomalous reconstruction behavior with individual process variables or local temporal regions. This improved interpretability relative to a single global reconstruction score.

The environmental assumptions of that system differ materially from the present work. Batch process data are structured as multichannel time-series measurements with semantically distinct process variables. SEM images, by contrast, are two-dimensional spatial intensity fields. A localized image defect may occupy only a small fraction of pixels, and averaging reconstruction error over the full image can dilute a high-intensity local anomaly. Consequently, a direct reuse of a 1-D architecture would not preserve the relevant spatial inductive bias. A 2-D convolutional representation is therefore adopted for the image domain.

### B. Existing Method

The existing reconstruction approach can be summarized as follows. A convolutional autoencoder is trained to reconstruct nominal process patterns. For an input x and reconstruction x_hat, a global reconstruction error is calculated over all dimensions. If the reconstruction error exceeds a threshold, the sample is considered anomalous. Localized reconstruction error is then examined to identify the variables or regions that contribute most strongly to the global anomaly score. In the semiconductor batch setting studied by Gorman et al., localized error improved the interpretability of an otherwise black-box reconstruction model [1].

The conceptual strength of this method is that labels for every possible failure mode are unnecessary. Instead, the autoencoder models the reference behavior, and deviations are detected through nonconformity. This is particularly relevant to manufacturing domains in which defects are diverse and rare. The method also establishes an important methodological precedent for the present project: global reconstruction error is useful for sample-level screening, whereas localized error provides evidence about the origin of the anomaly.

### C. Existing Result Abstract

Gorman et al. evaluated the method on the Tennessee Eastman Process and LAM 9600 Metal Etcher datasets and reported that localized reconstruction error improved anomaly-detection performance and explainability for similar model architectures. The LAM 9600 experiment reported AUC = 1.00 [1]. Because that value was obtained from batch process time series and not from Carinthia-S SEM images, it is not used as a direct numerical baseline for claiming superiority. The controlled baseline for the present work is the Phase-1 2-D ConvAE evaluated on the same Carinthia-S split as the proposed Phase-2 framework.

### D. Limitations and Drawbacks of the Existing Approach

Four limitations motivate the proposed system. First, a one-dimensional convolutional architecture is not directly suited to spatial defect morphology in SEM images. Spatial adjacency, texture, edge discontinuity, and local morphology are naturally represented by 2-D kernels. Second, a single reconstruction threshold does not provide calibrated risk. Two scores with the same numerical difference may not have the same rarity relative to the reference population, particularly after the image distribution changes. Third, standard anomaly thresholds generally do not incorporate the asymmetric manufacturing cost of a false negative. A missed defect can be substantially more expensive than an unnecessary engineering review, so an operating point should be selected under an explicit cost model. Fourth, the existing process-data formulation does not directly evaluate pixel-level defect localization against segmentation ground truth.

These limitations lead to a proposal in which the neural reconstruction model and the decision policy are treated as separate layers. Phase 1 supplies a frozen 2-D reconstruction model and a transparent global-error baseline. Phase 2 does not retrain a second independent network. Instead, it extracts additional local evidence from the same reconstruction-error map, robustly normalizes the global and local score distributions, fuses those scores, calibrates the fused nonconformity against held-out reference samples, and then chooses an operating alpha according to a validation-only cost function. This separation is important because it makes the contribution attributable to the decision framework rather than to uncontrolled changes in the neural backbone.

## III. SYSTEM DESIGN AND ARCHITECTURE

The proposed architecture follows a strict input-processing-output pipeline. The input consists of Carinthia-S grayscale SEM images and their associated segmentation masks. Masks are used for dataset verification, ground-truth definition, and evaluation; they are not provided to the autoencoder as training targets. The image pipeline begins with dataset verification, path resolution, image-mask dimension checks, duplicate detection, and a deterministic leakage-free split. The reference training partition contains only samples whose masks are empty. Separate validation, reference-calibration, policy-validation, and final-test partitions are maintained so that the final test data do not influence model or operating-point selection.

In Phase 1, each grayscale SEM image is normalized to the [0,1] range and supplied to a symmetric 2-D convolutional autoencoder. The encoder progressively reduces spatial resolution while increasing feature depth, producing a latent representation of the reference morphology. The decoder reconstructs the image from that latent representation. Mean squared reconstruction error over all pixels is used as the global anomaly score. A high quantile of reconstruction scores obtained from reference calibration images defines the Phase-1 operating threshold. This policy is then frozen before the final test split is evaluated.

Phase 2 starts from exactly the same frozen autoencoder. The pixel-wise squared reconstruction error map is retained instead of being immediately reduced to a single mean. Two complementary summary statistics are computed. The global score is the mean error over all pixels, and the local score is the mean of the highest-error k fraction of pixels. The local score is designed to preserve evidence from small spatial defects that may be diluted by global averaging.

Because the global and local scores have different numerical scales, robust normalization is fitted on reference calibration data using the median and median absolute deviation. A fused nonconformity score is then formed from the normalized global and local scores. The top-k fraction and the fusion coefficient lambda are selected using validation data only. Once selected, the fused score is calibrated against reference calibration samples using a split-conformal ranking rule. The resulting p-value measures how extreme a new fused score is relative to the held-out reference population.

A separate policy-validation partition is used to select the conformal significance threshold alpha. The false-negative cost is set higher than the false-positive cost, reflecting the manufacturing preference for avoiding missed defects. Once alpha has been selected, all Phase-2 settings are frozen and the final test set is evaluated only once under the complete decision policy.

The same reconstruction-error map is also used for localization. A pixel threshold is obtained from the upper quantile of error values on reference calibration images. Pixels above this threshold form the predicted anomaly mask. This predicted mask is compared with the expert segmentation mask using Dice coefficient, intersection over union (IoU), and pixel AUROC. The design therefore produces both an image-level screening decision and a spatial explanation that identifies suspicious regions.

Finally, the frozen detector is evaluated under controlled distribution shifts. Brightness, contrast, gamma, additive noise, and Gaussian blur perturbations are applied to both reference calibration data and the held-out test data. Each condition is evaluated first with the original frozen policy and then with reference-only recalibration. Test labels are never used for recalibration. The output of the system is therefore a set of screening decisions, anomaly scores, conformal p-values, predicted masks, image-level metrics, localization metrics, ablation results, and drift-robustness summaries.

## IV. ALGORITHM DESIGN AND MATHEMATICAL PROGRAMMING

### A. Phase-1 Reconstruction Model

Let x in [0,1]^(H x W) denote a grayscale SEM image. The autoencoder is composed of an encoder f_theta and decoder g_phi. The reconstruction is

x_hat = g_phi(f_theta(x)).                                                     (1)

For reference-only training, parameters theta and phi are optimized by minimizing mean squared reconstruction loss over the reference training set R_train:

L_rec = (1/|R_train|) sum_{x in R_train} [ (1/HW) sum_i (x_i - x_hat_i)^2 ].    (2)

The image-level global anomaly score is

S_g(x) = (1/HW) sum_i (x_i - x_hat_i)^2.                                      (3)

A Phase-1 threshold tau_g is selected as a high quantile q of the scores from the held-out reference-calibration set R_cal:

tau_g = Q_q({S_g(x): x in R_cal}).                                            (4)

The Phase-1 prediction is anomalous when S_g(x) > tau_g.

### B. Localized Top-k Evidence

The pixel-wise reconstruction-error map is

E_i(x) = (x_i - x_hat_i)^2.                                                    (5)

Let T_k(E) denote the set containing the largest k fraction of pixel errors. The local score is

S_l(x;k) = (1/|T_k|) sum_{i in T_k(E)} E_i(x).                                (6)

Candidate k values are evaluated only on the validation split. In the archived reference run, k = 0.001 was selected.

### C. Robust Normalization and Fusion

For score variable S, the calibration-set median m and median absolute deviation d are estimated. The robust normalized score is

Z(S) = (S - m)/(1.4826 d + epsilon).                                          (7)

The fused score is

S_f(x) = Z(S_g(x)) + lambda Z(S_l(x;k)).                                      (8)

Lambda is selected on the validation set by ranking performance and then frozen. In the archived reference run, lambda = 1.0.

### D. Conformal Risk Calibration

Let a_1,...,a_n be fused nonconformity scores from reference calibration images and let a_new be the fused score of a new image. The empirical conformal p-value is

p(x) = [1 + #{i: a_i >= a_new}] / (n + 1).                                   (9)

A small p-value indicates that the image is unusual relative to reference calibration data. This calibration step does not require a parametric score distribution.

### E. Cost-Aware Alpha Selection

For candidate significance level alpha, the image is predicted anomalous when p(x) <= alpha. On the policy-validation set, the cost is

C(alpha) = C_FN FN(alpha) + C_FP FP(alpha),                                   (10)

with C_FN = 10 and C_FP = 1 in the default experiment. The selected operating point is

alpha* = arg min_alpha C(alpha).                                              (11)

The archived reference run selected alpha* = 0.075. The final test set is not evaluated during this optimization.

### F. Pixel Localization

A localization threshold tau_p is fitted from the high quantile of pixel errors observed on reference calibration images. The predicted mask is

M_hat_i = 1[E_i >= tau_p].                                                     (12)

Given ground-truth mask M, Dice and IoU are

Dice = 2|M intersect M_hat| / (|M| + |M_hat|),                                (13)

IoU = |M intersect M_hat| / |M union M_hat|.                                  (14)

### G. Pseudo-code

**Algorithm 1: RCC-ConvAE Training, Calibration, and Evaluation**

```
Input:
    Carinthia-S SEM images X and masks M
    candidate top-k set K
    candidate fusion weights L
    candidate conformal levels A
    false-negative cost C_FN
    false-positive cost C_FP

1: Verify image-mask pairs and derive reference/anomaly ground truth from mask content.
2: Create disjoint TRAIN, VALIDATION, CALIBRATION, POLICY_VALIDATION, and TEST splits.
3: Train 2-D ConvAE only on reference TRAIN images.
4: Select checkpoint using reference VALIDATION reconstruction loss.
5: Freeze the ConvAE.
6: Compute reference CALIBRATION global reconstruction scores.
7: Set Phase-1 threshold from a reference-calibration quantile.
8: For each k in K:
       compute localized top-k scores on VALIDATION;
       evaluate ranking metrics.
9: Freeze k* using VALIDATION only.
10: Fit robust global/local normalization on reference CALIBRATION scores.
11: For each lambda in L:
        fuse global/local VALIDATION scores;
        evaluate ranking metrics.
12: Freeze lambda* using VALIDATION only.
13: Compute fused reference CALIBRATION scores.
14: Convert POLICY_VALIDATION fused scores to conformal p-values.
15: For each alpha in A:
        predict anomaly if p <= alpha;
        cost = C_FN * FN + C_FP * FP.
16: Freeze alpha* with minimum policy-validation cost.
17: Fit pixel localization threshold using reference CALIBRATION error maps.
18: Evaluate frozen Phase 1 and Phase 2 once on TEST.
19: Compare predicted masks with ground-truth masks using Dice, IoU, pixel AUROC.
20: Repeat stress tests under controlled drift with frozen and reference-only recalibrated policies.
Output:
    decisions, scores, p-values, masks, metrics, ablations, drift summaries.
```

## V. IMPLEMENTATION

The reproducible implementation is developed in Python with PyTorch for the autoencoder, NumPy and pandas for numerical and tabular processing, scikit-learn for evaluation metrics, Pillow and SciPy-compatible image operations for preprocessing, Matplotlib for research figures, and Streamlit/Plotly for the interactive dashboard. The project is organized as independent modules for data verification, training, scoring, conformal calibration, cost policy, localization, drift generation, evaluation, inference, and reporting. Configuration is stored in a YAML file so that paths and hyperparameters are not embedded into source code.

The packaged implementation is designed for Windows PowerShell and CPU-capable environments. The training pipeline uses deterministic seed 42, an Adam optimizer, MSE reconstruction loss, and early stopping based exclusively on reference-validation loss. The autoencoder accepts one-channel grayscale input. Carinthia-S source images are 480 x 480 pixels in the verified release [2], [3], and the default configuration preserves a 480 x 480 input size. A lighter resolution can be configured for constrained hardware, but any such change should be treated as a new experimental run rather than mixed with the archived results.

Dataset integrity is checked before training. The verifier confirms that each CSV row resolves to an image and mask, verifies equal image-mask dimensions, counts empty and non-empty masks, calculates mask coverage, inspects duplicate filenames, and records class distribution. A split manifest and leakage report are written to disk. The final-test split is prohibited from all tuning steps. This rule is enforced at the experimental-design level: Phase-1 training uses reference TRAIN, early stopping uses reference VALIDATION, threshold/calibration statistics use reference CALIBRATION, and cost-aware alpha is selected using POLICY_VALIDATION.

The application layer is implemented as a Streamlit dashboard named “SemiVision AI - Semiconductor Defect Intelligence Platform.” The interface provides project overview, single-image screening, batch screening with CSV export, research metrics, distribution-drift analysis, ablation results, model/policy metadata, and methodology notes. Live screening loads the frozen checkpoint and policy artifacts. The dashboard does not display an invented confidence percentage; instead, Phase 2 exposes the global score, local score, fused score, conformal p-value, selected alpha, reconstruction, error heatmap, and predicted anomaly mask.

The project package also separates archived reference metrics from freshly recomputed results. Files under `results/reference_run` contain metrics obtained from the previously verified full project execution. Fresh experiments write to `results/phase1`, `results/phase2`, `results/ablation`, and `results/drift`. This prevents a new environment from silently treating historical metrics as if they had just been regenerated without access to the original raw dataset and checkpoint.

## VI. RESULTS AND DISCUSSION

### A. Image-Level Screening

The archived final Phase-1 run produced ROC-AUC 0.9971 and PR-AUC 0.9998, indicating very strong separation of reference and anomalous SEM samples. At the frozen operating threshold, F1 was 0.9989 and balanced accuracy was 0.9886, with false-positive rate 0.0217 and false-negative rate 0.0011. These results demonstrate that a relatively compact reference-only 2-D ConvAE can model the normal SEM structure sufficiently well for high-quality image-level screening on the selected split.

The Phase-2 RCC-ConvAE framework retained essentially the same image-level performance in the archived run: ROC-AUC 0.9971, PR-AUC 0.9998, F1 0.9989, balanced accuracy 0.9886, FPR 0.0217, and FNR 0.0011. Cost per image was approximately 0.0119 under the default C_FN:C_FP ratio of 10:1. These results are important because they prevent an incorrect research narrative. Phase 2 should not be described as producing a large image-level accuracy gain. The Phase-1 baseline was already close to saturation; therefore the meaningful contribution is the quality and reliability of the evidence and decision process layered on top of the same reconstruction model.

### B. Localization Performance

The largest controlled improvement was observed in localization. Phase-1 exploratory localization achieved Dice 0.4142, IoU 0.2704, and pixel AUROC 0.9450. Phase 2 achieved Dice 0.5381, IoU 0.3795, and pixel AUROC 0.9860. The absolute Dice gain was 0.1239, corresponding to approximately 29.9% relative improvement. IoU increased by 0.1091, or approximately 40.3% relative to the Phase-1 value. Pixel AUROC increased by 0.0410.

This pattern is consistent with the motivation for localized top-k evidence. A global mean is effective for deciding whether an image is anomalous but can suppress information about a small defect because most pixels may still be reconstructed accurately. Retaining the high-error tail emphasizes spatially concentrated departures from the learned reference structure. The pixel-level result therefore provides evidence that Phase 2 is not simply reparameterizing the same image score; it makes the reconstruction error more useful for defect localization.

### C. Ablation Analysis

The archived ablation experiment compared global reconstruction, global/local fusion, fixed-alpha conformal calibration, cost-aware alpha, and local-only evidence. The Phase-1 global baseline produced ROC-AUC 0.997093, PR-AUC 0.999836, F1 0.998857, FPR 0.021739, FNR 0.001143, and cost/image 0.011944. The global/local fusion variant produced essentially the same values at the evaluated operating point.

When a fixed conformal alpha = 0.05 was imposed, ranking metrics remained unchanged but F1 fell to 0.930403 and FNR increased to 0.129143; cost/image rose to 1.228013. This observation illustrates why a calibrated score alone is insufficient. Calibration must be paired with an operating policy appropriate for the asymmetric manufacturing loss. When the cost-aware alpha* = 0.075 was selected on the dedicated policy-validation partition, F1 returned to 0.998857, FNR returned to 0.001143, and cost/image returned to 0.011944.

The local-only variant achieved ROC-AUC 0.999801 and PR-AUC 0.999990, but its FPR was 0.108696 and cost/image was 0.016287 at the evaluated threshold. Thus, a higher ranking metric did not automatically translate into a preferable operational decision point. This result supports a central argument of the work: ranking quality, calibration, and decision cost should be evaluated separately.

### D. Distribution-Drift Stress Testing

Distribution-shift experiments revealed a failure mode that is not visible from clean-test AUC alone. Under the frozen policy, the archived average balanced accuracy across the evaluated conditions was 0.680495 and the worst balanced accuracy was 0.500000. Mean FPR increased to 0.633152 even though mean ROC-AUC remained 0.97027. The contrast between relatively strong ranking and poor thresholded performance indicates that score distributions can move substantially under image perturbations without destroying the pairwise ranking of anomalies and references.

Reference-only recalibration materially improved thresholded reliability. Mean balanced accuracy increased to 0.933084, an absolute gain of 0.252589, while mean FPR decreased from 0.633152 to 0.103261. The worst balanced accuracy improved from 0.500000 to 0.878211. Brightness +0.10, brightness -0.10, gamma 1.50, and additive-noise conditions were particularly illustrative: frozen-policy balanced accuracy fell to 0.5 in the archived run, while recalibration restored values between 0.878 and 0.989 depending on the perturbation.

The archived drift table showed the same aggregate drift values for Phase 1 and Phase 2 under their respective frozen policies and the same improvement after recalibration. Consequently, no claim is made that the RCC fusion alone solved distribution shift. Instead, the evidence supports the broader framework requirement that deployment should include explicit monitoring and recalibration of the operating distribution when acquisition conditions change.

### E. Discussion

Three conclusions follow from the results. First, a reference-only 2-D ConvAE is a strong baseline for the studied Carinthia-S screening task. This makes methodological discipline essential: introducing additional decision machinery must not be justified by an artificial claim of improved AUC when the baseline already achieves 0.9971. Second, localization is a more discriminating evaluation dimension. The improvement in Dice, IoU, and pixel AUROC demonstrates that localized reconstruction evidence adds practical value even when image-level ranking is nearly unchanged. Third, the drift results show that calibration and operating-point management are deployment issues rather than optional presentation features. A detector can retain high ranking capability while a frozen threshold produces unacceptable false-positive behavior.

The results also emphasize limitations. Carinthia-S is based on SEM imagery from a particular manufacturing context, and the reference/anomaly split is derived from mask content. Generalization to different semiconductor layers, tools, magnifications, or factories requires additional external validation. Conformal p-values rely on the calibration sample being representative under the assumptions of the chosen procedure; distribution drift can violate that condition, which is why separate recalibration experiments are reported. Finally, reconstruction error is not a causal diagnosis. The heatmap localizes model mismatch, but process-engineering root cause requires integration with process metadata or downstream expert analysis.

## VII. COMPARATIVE ANALYSIS

A rigorous comparison must distinguish cross-dataset context from controlled same-dataset evaluation. The Gorman et al. system [1] reported AUC = 1.00 on a LAM 9600 batch-manufacturing experiment using 1-D temporal/process signals. The proposed project operates on Carinthia-S 2-D SEM imagery. Since the modality, population, split, and anomaly distribution are different, 1.00 and 0.9971 are not treated as a direct performance contest. The contribution of [1] is instead used to justify the reconstruction/LRE methodological lineage.

The controlled quantitative comparison is Phase 1 versus Phase 2 on Carinthia-S. At image level, both phases produced ROC-AUC 0.9971, PR-AUC 0.9998, F1 0.9989, and balanced accuracy 0.9886 in the archived run. Thus, Phase 2 preserved the strong baseline rather than increasing an already saturated ranking metric. At pixel level, however, Phase 2 increased Dice from 0.4142 to 0.5381, IoU from 0.2704 to 0.3795, and pixel AUROC from 0.9450 to 0.9860. These differences directly support the localized-evidence component because all comparisons are performed within the same project and data domain.

The ablation study further separates score ranking from decision behavior. Local-only evidence reached ROC-AUC 0.999801 but had FPR 0.108696, while the cost-aware full policy retained FPR 0.021739 with lower cost/image. Similarly, fixed conformal alpha = 0.05 increased FNR to 0.129143, whereas validation-selected alpha* = 0.075 restored the low FNR of 0.001143. These comparisons are more informative than a checklist of added features because they quantify how individual policy choices alter operational error.

Finally, the distribution-drift experiment demonstrates a comparison between frozen and recalibrated operating policies. Mean balanced accuracy increased from 0.680495 to 0.933084 after reference-only recalibration, and mean FPR decreased from 0.633152 to 0.103261. This result does not imply that recalibration repairs every form of domain shift; blur conditions, for example, did not universally improve. It does demonstrate that decision thresholds should be treated as environment-dependent policy parameters rather than immutable constants.

Overall, the comparative evidence supports a precise claim: the proposed RCC-ConvAE framework does not replace the strong ConvAE representation with a deeper model solely to raise ROC-AUC. Instead, it preserves image-level discrimination while improving localization and providing a reproducible mechanism for calibrated, cost-sensitive, and drift-aware decision support.

## VIII. CONCLUSION AND FUTURE WORK

A two-phase semiconductor SEM visual screening framework was presented. Phase 1 established a reference-only 2-D convolutional autoencoder baseline, while Phase 2 introduced RCC-ConvAE as a risk-calibrated and cost-aware decision framework built on the frozen reconstruction model. The design integrates localized top-k reconstruction evidence, robust global-local fusion, split-conformal calibration, asymmetric cost-aware alpha selection, pixel-level defect localization, and distribution-drift evaluation.

The archived Carinthia-S experiment showed that the Phase-1 baseline already achieved very strong image-level discrimination. Phase 2 therefore did not require an exaggerated accuracy narrative. Its principal empirical contribution was improved localization, with Dice increasing from 0.4142 to 0.5381 and IoU from 0.2704 to 0.3795, while pixel AUROC increased from 0.9450 to 0.9860. Distribution-shift analysis additionally showed that recalibration can substantially improve thresholded performance under acquisition perturbations even when ranking metrics remain comparatively stable.

Future work should validate the framework across additional SEM acquisition settings, wafer layers, tools, and manufacturing sites. Multi-scale reconstruction or pretrained industrial visual features could be incorporated while preserving the same leakage-free calibration protocol. A selective-review state could also be evaluated for samples near the operating boundary. Finally, integration with process metadata would enable the localized visual evidence to support downstream root-cause analysis rather than only anomaly localization.

## REFERENCES

[1] M. Gorman, X. Ding, L. Maguire, and D. Coyle, “Anomaly Detection in Batch Manufacturing Processes Using Localized Reconstruction Errors From 1-D Convolutional AutoEncoders,” IEEE Transactions on Semiconductor Manufacturing, vol. 36, no. 1, pp. 147-150, 2023, doi: 10.1109/TSM.2022.3216032.

[2] V. Hasić, C. Kofler, and S. Krivić, “Towards Explaining SEM Defect Image Classification,” Frontiers in Artificial Intelligence and Applications, vol. 413, pp. 5463-5470, ECAI 2025, doi: 10.3233/FAIA251487.

[3] C. Kofler and V. Hasić, “Carinthia-S dataset,” Zenodo, version 1, Aug. 2025, doi: 10.5281/zenodo.16895427.

[4] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9592-9600, doi: 10.1109/CVPR.2019.00982.

[5] L. Ruff et al., “Deep One-Class Classification,” in Proc. 35th International Conference on Machine Learning (ICML), PMLR, vol. 80, 2018, pp. 4393-4402.

[6] A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494-591, 2023, doi: 10.1561/2200000101.

[7] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. 34th International Conference on Machine Learning (ICML), PMLR, vol. 70, 2017, pp. 1321-1330.

[8] Y. Ovadia et al., “Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019, pp. 13991-14002.
