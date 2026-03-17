# Literature Review: Strategies to Improve Segmentation with Limited, Noisy Labels and Abundant Unlabeled Data

**Context:** Binary segmentation of protein aggregates in fluorescence microscopy images. 19 annotated images, noisy boundaries, millions of unlabeled raw images available. Current baseline: UNet with Dice+BCE loss, 0.81 Dice.

---

## 1. Self-Supervised / Unsupervised Pretraining

The core idea: pretrain the encoder (or full UNet) on the millions of unlabeled images using a pretext task, then fine-tune on the 19 labeled images. This gives the network strong feature representations before it ever sees a label.

### 1.1 Autoencoders and Variational Autoencoders (AE/VAE)

**How it works:** Train the UNet (or its encoder-decoder) to reconstruct input images from a bottleneck representation. The encoder learns meaningful features of aggregate morphology, cell texture, and background without any labels. For a VAE, the bottleneck is regularized to follow a Gaussian distribution, which can improve generalization.

**Key references:**
- Baur et al., "Autoencoders for unsupervised anomaly segmentation in brain MRI" (2021) -- AE/VAE pretraining for medical segmentation
- Yu et al., "An Auto-Encoder Strategy for Adaptive Image Segmentation" (MIDL 2020) -- VAE subnetwork provides regularization when training data is limited

**Typical improvement:** 3-8% Dice improvement when fine-tuning from AE-pretrained weights vs. random init, especially with <50 labeled images.

**Microscopy validation:** Yes. CellSeg3D (eLife, 2024) uses self-supervised pretraining for 3D cell segmentation in fluorescence microscopy, achieving performance on par with top supervised methods.

**Practical complexity:** LOW. The UNet architecture is already an autoencoder. Pretraining = train UNet to reconstruct images with MSE/L1 loss, then swap the output head for segmentation. Can be done in a few hours on the millions of unlabeled images.

**Addresses the scenario:** DIRECTLY. This is the most natural way to leverage millions of unlabeled images for a UNet-based pipeline.

### 1.2 Masked Image Modeling (MAE / BEiT)

**How it works:** Randomly mask out patches of the input image and train the network to reconstruct the missing content. Forces the model to learn contextual understanding of image structure.

**Key references:**
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- "Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology" (2024) -- ViT-based MAE on up to 93M microscopy images, achieving 11.5% relative improvement over weakly supervised baselines
- "Masked pretraining of U-Net for ultrasound image segmentation" (Scientific Reports, 2025) -- **6-20 percentage points Dice improvement** on small datasets by masking pixels in UNet input and pretraining to reconstruct

**Typical improvement:** 6-20% Dice improvement on small labeled datasets. The UNet-specific masked pretraining paper is particularly relevant -- it randomly masks pixels in the input, pretrains UNet to predict masked content, then fine-tunes for segmentation.

**Microscopy validation:** YES. The 2024 MAE-for-microscopy paper demonstrates strong scaling on cellular biology datasets (tested on up to 93M images). Performance consistently improves with more unlabeled data.

**Practical complexity:** MODERATE. For UNet: mask random pixels, train to reconstruct, fine-tune. For ViT-based MAE: requires switching to a Vision Transformer backbone. The UNet-pixel-masking approach is simpler and directly applicable.

**Addresses the scenario:** DIRECTLY. Designed for leveraging massive unlabeled datasets.

### 1.3 Contrastive Learning (SimCLR, MoCo, BYOL)

**How it works:** Train the encoder to produce similar embeddings for augmented views of the same image and dissimilar embeddings for different images. No labels needed.

**Key references:**
- Chen et al., "SimCLR: A Simple Framework for Contrastive Learning" (ICML 2020)
- He et al., "MoCo: Momentum Contrast for Unsupervised Visual Representation Learning" (CVPR 2020)
- Grill et al., "BYOL: Bootstrap Your Own Latent" (NeurIPS 2020)
- "Self-supervised pre-training with contrastive and masked autoencoder methods for dealing with small datasets in deep learning for medical imaging" (Scientific Reports, 2023) -- comparative study
- "A knowledge-based learning framework (TOWER) for self-supervised pre-training towards enhanced recognition of biomedical microscopy images" (2023) -- synergizes contrastive and generative learning for microscopy

**Typical improvement:** 3-10% accuracy improvement on downstream tasks with few labels. BYOL tends to outperform SimCLR and MoCo on medical imaging tasks. MoCo and BYOL require smaller batch sizes than SimCLR, making them more practical.

**Microscopy validation:** YES. Multiple studies validate contrastive pretraining on microscopy data. The TOWER framework specifically targets biomedical microscopy.

**Practical complexity:** MODERATE-HIGH. Requires careful augmentation design, projection heads, and either large batch sizes (SimCLR) or momentum encoders (MoCo/BYOL). Works on the encoder only -- need to attach a decoder afterward.

**Addresses the scenario:** YES, but less naturally than AE/MAE approaches for dense prediction (segmentation). Contrastive learning excels for classification; for segmentation, pixel-level pretraining (AE/MAE) tends to be more effective.

### 1.4 Recommendation for Self-Supervised Pretraining

**Priority order for this project:**
1. **Masked pixel reconstruction on UNet** (simplest, directly proven for UNet + small datasets, 6-20% improvement)
2. **Standard autoencoder pretraining** (simplest to implement, 3-8% improvement)
3. **BYOL contrastive pretraining** (if switching to a pretrained encoder backbone)

---

## 2. Semi-Supervised Learning

Use both the 19 labeled images AND a large pool of unlabeled images during training simultaneously.

### 2.1 Mean Teacher

**How it works:** Maintain two copies of the network: a student (trained with gradient descent) and a teacher (exponential moving average of student weights). Enforce consistency between their predictions on unlabeled data.

**Key references:**
- Tarvainen & Valpola, "Mean teachers are better role models" (NeurIPS 2017)
- "Uncertainty Aware Mean Teacher" -- adds uncertainty weighting to consistency loss
- SSL4MIS repository (github.com/HiLab-git/SSL4MIS) -- benchmark implementation of 12+ semi-supervised methods for medical image segmentation

**Typical improvement:** 5-15% Dice improvement over supervised-only when using 5-20 labeled images + hundreds/thousands unlabeled. One study achieved 0.878 Dice with only 10% labeled data using Mean Teacher on cardiac segmentation.

**Microscopy validation:** Yes, validated in multiple medical imaging contexts. The SSL4MIS benchmark supports UNet backbone.

**Practical complexity:** LOW-MODERATE. Add an EMA copy of the network, add consistency loss on unlabeled data. Well-documented implementations available.

### 2.2 Cross Pseudo Supervision (CPS)

**How it works:** Train two networks with different initializations. Each network generates pseudo-labels for the other on unlabeled data. The disagreement between networks acts as implicit regularization.

**Key references:**
- Chen et al., "Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision" (CVPR 2021)
- "Cross Teaching between CNN and Transformer" -- uses architecturally different networks (CNN + Transformer) for higher diversity

**Typical improvement:** Outperforms Mean Teacher by 2-4% Dice. With <10% labeled data, improvements of >4% Dice and 6mm reduction in Hausdorff distance reported.

**Practical complexity:** MODERATE. Need to train two networks simultaneously. Higher GPU memory requirement.

### 2.3 FixMatch / Consistency Regularization

**How it works:** Generate pseudo-labels from weakly augmented unlabeled images. Train on strongly augmented versions of the same images using only high-confidence pseudo-labels (above a threshold).

**Key references:**
- Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning" (NeurIPS 2020)
- "UniMatch" (2023) -- extends FixMatch with three-branch consistency for segmentation
- "SegMatch" (Scientific Reports, 2025) -- adapts FixMatch specifically for segmentation with pseudo-label generation

**Typical improvement:** 5-12% Dice improvement in low-label regimes. Adaptive thresholding variants improve stability.

**Practical complexity:** LOW-MODERATE. Single network, straightforward implementation. The key challenge is defining appropriate weak vs. strong augmentations for microscopy.

### 2.4 Self-Training / Pseudo-Labeling

**How it works:** Train on labeled data, generate predictions on unlabeled data, filter high-confidence predictions as pseudo-labels, retrain. Iterate.

**Key references:**
- Lee, "Pseudo-label: The simple and efficient semi-supervised learning method" (2013)
- CellSeg3D (eLife, 2024) -- uses self-supervised learning to generate pseudo-labels for 3D cell segmentation, then inspects/corrects them, achieving performance on par with top supervised methods

**Typical improvement:** 3-8% Dice improvement. Quality depends heavily on initial model quality and confidence thresholding.

**Practical complexity:** LOW. Train model, predict on unlabeled data, threshold, retrain. Can be done iteratively.

### 2.5 Recommendation for Semi-Supervised Learning

**Priority order:**
1. **Mean Teacher** (simplest, well-proven, good implementations available in SSL4MIS)
2. **Self-training with pseudo-labels** (very simple iterative approach)
3. **Cross Pseudo Supervision** (higher gains but more complex)

**Important note:** With only 19 labeled images, semi-supervised methods can be unstable. Combining with self-supervised pretraining (Section 1) is strongly recommended -- pretrain first, then apply semi-supervised fine-tuning.

---

## 3. Noise-Robust Training

Address the noisy boundary annotations and potentially missed objects in the 19 labeled images.

### 3.1 Label Smoothing and Spatially Varying Label Smoothing (SVLS)

**How it works:** Standard label smoothing replaces hard 0/1 labels with soft targets (e.g., 0.05/0.95). SVLS varies the smoothing spatially -- more smoothing near boundaries (where uncertainty is high), less in confident regions.

**Key references:**
- Islam & Glocker, "Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations" (IPMI 2021) -- achieves "superior boundary prediction with improved uncertainty and model calibration"
- "Clinical Expert Uncertainty Guided Generalized Label Smoothing" (2025) -- adapts smoothing based on clinical knowledge

**Typical improvement:** 1-3% Dice improvement, with much better calibration and boundary quality. The improvement is specifically at boundaries -- exactly where noise exists in this project.

**Practical complexity:** VERY LOW. Replace hard labels with soft labels. For SVLS: apply distance transform to boundaries, use distance-based smoothing. ~10 lines of code.

**Directly addresses:** Boundary uncertainty in annotations. Highly recommended.

### 3.2 Symmetric Cross-Entropy (SCE) and Robust Loss Functions

**How it works:** SCE combines standard cross-entropy with a "reverse cross-entropy" term that is inherently noise-robust. Addresses both overfitting to noisy labels on easy regions and underfitting on hard regions.

**Key references:**
- Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
- Ma et al., "Normalized Loss Functions for Deep Learning with Noisy Labels" (ICML 2020) -- generalized framework for noise-robust losses
- "Loss Functions in the Era of Semantic Segmentation: A Survey" (2023) -- comprehensive survey noting region-level losses (Dice) are inherently more robust to outliers than pixel-level losses

**Typical improvement:** 2-5% accuracy improvement under 20-40% label noise. The current Dice+BCE combination already provides some robustness (Dice is region-level). Adding SCE or replacing BCE with SCE could help.

**Practical complexity:** VERY LOW. Drop-in replacement for BCE loss. `SCE = CE + alpha * RCE`.

### 3.3 Co-Teaching

**How it works:** Train two networks simultaneously. Each network selects small-loss samples for the other to train on, filtering out likely noisy samples.

**Key references:**
- Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels" (NeurIPS 2018)
- "CoDC: Accurate Learning with Noisy Labels via Disagreement and Consistency" (2024) -- combines disagreement and consistency

**Typical improvement:** 5-10% accuracy improvement under heavy noise (>40% noise rate). Less beneficial for moderate noise.

**Practical complexity:** MODERATE. Train two networks, implement sample selection logic. Higher compute cost.

### 3.4 Mean-Teacher-Assisted Confident Learning (MTCL)

**How it works:** Combines Mean Teacher with confident learning for label denoising. The teacher provides stable predictions used to identify and correct noisy labels, while consistency regularization extracts knowledge from imperfect data.

**Key references:**
- "Anti-Interference From Noisy Labels: Mean-Teacher-Assisted Confident Learning for Medical Image Segmentation" (IEEE TMI, 2022)

**Typical improvement:** Demonstrated "superior segmentation performance" on cardiac, hepatic vessel, and retinal vessel segmentation with real-world noisy annotations.

**Practical complexity:** MODERATE. Combines two techniques but both are well-documented.

### 3.5 Confident Learning (Cleanlab)

**How it works:** Automatically identifies mislabeled examples by analyzing model predictions vs. given labels. For segmentation: identifies pixels/regions where annotations are likely wrong.

**Key references:**
- Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels" (JAIR, 2021)
- Cleanlab library (github.com/cleanlab/cleanlab)

**Typical improvement:** Improves dataset quality rather than model directly. Removing/correcting 5-15% of noisy labels can yield 2-5% model improvement.

**Practical complexity:** LOW for classification, MODERATE for segmentation (need pixel-level analysis). The cleanlab library handles classification well; for segmentation, custom implementation is needed.

### 3.6 Recommendation for Noise-Robust Training

**Priority order:**
1. **Spatially Varying Label Smoothing** (trivial to implement, directly addresses boundary noise)
2. **Symmetric Cross-Entropy loss** (drop-in replacement for BCE, addresses label noise)
3. **Confident Learning** (identify and re-annotate the worst labels in the 19 images)

---

## 4. Data Augmentation Beyond Basics

### 4.1 CutMix and MixUp for Segmentation

**How it works:**
- **MixUp:** Linearly blend two images and their masks: `img = lambda*img1 + (1-lambda)*img2`
- **CutMix:** Cut a rectangular patch from one image and paste it onto another, blending masks accordingly

**Key references:**
- Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
- Zhang et al., "MixUp: Beyond Empirical Risk Minimization" (ICLR 2018)

**Typical improvement:** 1-3% Dice improvement. CutMix generally outperforms MixUp for segmentation because it preserves local structure.

**Practical complexity:** VERY LOW. ~5-10 lines of code in the data loader.

**Note:** With only 19 images, these become more valuable as they create novel training compositions. CutMix effectively multiplies the apparent dataset size.

### 4.2 Generative Augmentation (GAN/Diffusion)

**How it works:** Train a generative model (GAN or diffusion model) on the labeled images to synthesize new image-mask pairs.

**Key references:**
- "Test-Time Generative Augmentation for Medical Image Segmentation" (Medical Image Analysis, 2025) -- 0.1-2.3% Dice improvement
- Various conditional diffusion models for medical image synthesis (2023-2024)

**Typical improvement:** 1-5% Dice improvement, but highly dependent on quality of generated images.

**Practical complexity:** HIGH. Training a good generative model on only 19 images is extremely difficult. The generator will likely produce poor-quality or overfitted images. NOT recommended with so few labeled images.

**Exception:** If a generative model is trained on the millions of UNLABELED images (unconditional generation) and then used for style transfer / domain adaptation, this becomes more viable but still complex.

### 4.3 Test-Time Augmentation (TTA)

**How it works:** At inference, apply multiple augmentations (flips, rotations, scale) to the input, predict on each, and average the predictions.

**Key references:**
- "S3-TTA: Scale-Style Selection for Test-Time Augmentation in Biomedical Image Segmentation" (2023) -- 1.3-3.4% improvement on cell and lung segmentation
- "Test-time augmentation for deep learning-based cell segmentation on microscopy images" (Scientific Reports, 2020)

**Typical improvement:** 1-3% Dice improvement at inference. Effect is larger with small training sets and simpler models.

**Practical complexity:** VERY LOW. Apply flips + 90-degree rotations at inference, average predictions. No retraining needed. Free improvement.

### 4.4 Copy-Paste Augmentation

**How it works:** Extract segmented objects (aggregates) from one image and paste them onto another image's background. Creates novel spatial configurations.

**Key references:**
- Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" (CVPR 2021)

**Typical improvement:** 2-5% for instance/binary segmentation. Particularly effective when objects are sparse (as protein aggregates often are).

**Practical complexity:** LOW. Extract aggregate masks, paste onto random backgrounds. Works well for binary segmentation.

### 4.5 Recommendation for Augmentation

**Priority order:**
1. **Test-Time Augmentation** (free improvement, no retraining)
2. **Copy-Paste augmentation** (effective for sparse objects like aggregates)
3. **CutMix** (simple regularization benefit)
4. **Generative augmentation** (only if trained on unlabeled data, not on the 19 labeled images)

---

## 5. Active Learning

Select which of the millions of unlabeled images to annotate next for maximum benefit.

### 5.1 Uncertainty-Based Selection

**How it works:** Run the current model on unlabeled images. Select images where the model is most uncertain (highest entropy, lowest confidence, or highest disagreement between augmented predictions) for annotation.

**Key references:**
- "A comprehensive survey on deep active learning in medical image analysis" (Medical Image Analysis, 2024)
- "Breaking the Barrier: Selective Uncertainty-based Active Learning for Medical Image Segmentation" (2024) -- prioritizes uncertain pixels near decision boundaries
- "Active Learning in Brain Tumor Segmentation with Uncertainty Sampling and Annotation Redundancy Restriction" (2024) -- reduces needed training data by 20-30%

**Typical improvement:** Achieves equivalent performance to full dataset training with 30-50% less labeled data. With Bayesian dropout uncertainty, target performance can be reached with ~70% fewer annotations than random selection.

**Practical complexity:** LOW-MODERATE. Run MC Dropout (multiple forward passes with dropout enabled), compute prediction variance, rank images by uncertainty.

**Addresses the scenario:** DIRECTLY. With millions of unlabeled images, intelligently selecting the next 10-20 images to annotate could be more valuable than any algorithmic improvement.

### 5.2 Diversity-Based Selection

**How it works:** Select images that are diverse (different from already-labeled images and from each other), ensuring coverage of the data distribution.

**Typical improvement:** Combined uncertainty + diversity outperforms either alone by 2-5%.

**Practical complexity:** MODERATE. Requires embedding the images (e.g., from the pretrained encoder) and performing clustering or coreset selection.

### 5.3 Recommendation for Active Learning

**Use uncertainty-based selection** (MC Dropout) to pick the next batch of images to annotate. Even annotating 10-20 more well-selected images could push Dice from 0.81 to 0.86+. This is the single highest-impact strategy if annotation budget exists.

---

## 6. Transfer Learning and Foundation Models

### 6.1 ImageNet Pretrained Encoders

**How it works:** Use a ResNet/EfficientNet encoder pretrained on ImageNet as the UNet encoder, instead of random initialization.

**Key references:**
- "Enhancing pretraining efficiency for medical image segmentation via transferability metrics" (2024)
- Multiple studies showing 6-20 percentage point Dice improvement on small medical datasets with pretrained encoders

**Typical improvement:** With ~19 labeled images, pretrained encoders typically improve Dice by 5-15%. The benefit diminishes significantly above ~150 labeled images.

**Microscopy-specific:** Pretraining on microscopy datasets (e.g., MicroNet) gives 52.5% accuracy vs. ImageNet's 47.7% vs. no pretraining at 34.0% for downstream microscopy tasks.

**Practical complexity:** VERY LOW. Use `torchvision.models.resnet34(pretrained=True)` as encoder. Requires adjusting input channels (1-channel fluorescence vs. 3-channel RGB). Common approach: repeat the single channel 3x, or average the pretrained first-conv weights.

### 6.2 Foundation Models: CellSAM

**How it works:** CellSAM combines a cell detector (CellFinder) with SAM's mask decoder. Achieves human-level cell segmentation across mammalian cells, yeast, and bacteria.

**Key references:**
- Israel et al., "CellSAM: A Foundation Model for Cell Segmentation" (Nature Methods, 2025)

**Performance:** No statistical difference from human annotators (p=0.18 to p=0.90 across cell types). Strong zero-shot performance, improvable with ~10 example fields of view.

**Applicability to protein aggregates:** LIMITED. CellSAM is designed for whole-cell segmentation, not subcellular structures like protein aggregates. The aggregates are smaller, denser, and have different morphology than cells.

### 6.3 Foundation Models: micro-SAM

**How it works:** Fine-tunes SAM specifically for microscopy. Adds automatic instance segmentation via distance maps and foreground probability prediction.

**Key references:**
- "Segment Anything for Microscopy" (2024) -- achieves performance on par or better than CellPose for automatic segmentation

**Performance:** Clear improvement over default SAM across all microscopy settings. Generalist light microscopy model transfers to unseen imaging conditions.

**Applicability:** MODERATE. Could be fine-tuned for aggregate detection using the napari plugin interface with the 19 labeled images. However, switching to SAM architecture means abandoning the current UNet pipeline.

**Practical complexity:** MODERATE. Install micro-SAM, fine-tune on labeled data through napari GUI. But integrating into the existing pipeline requires significant refactoring.

### 6.4 Recommendation for Transfer Learning

**Priority order:**
1. **ImageNet pretrained encoder** (trivial, immediate improvement, keep UNet architecture)
2. **Microscopy-pretrained encoder** (if available, slightly better than ImageNet)
3. **micro-SAM** (only if willing to change architecture entirely)
4. **CellSAM** (not directly applicable to aggregate segmentation)

---

## 7. Combined Strategy Recommendation

Based on this review, here is the recommended priority-ordered implementation plan, sorted by expected impact and ease of implementation:

### Tier 1: Quick Wins (1-2 days each, 3-8% improvement each)

| Strategy | Expected Gain | Effort |
|----------|--------------|--------|
| ImageNet pretrained encoder | +5-10% Dice | Very low |
| Test-time augmentation | +1-3% Dice | Very low |
| Spatially varying label smoothing | +1-3% Dice (boundary) | Very low |
| Symmetric cross-entropy loss | +1-3% Dice | Very low |
| Copy-paste augmentation | +2-5% Dice | Low |

### Tier 2: High Impact (1-2 weeks, 5-15% improvement)

| Strategy | Expected Gain | Effort |
|----------|--------------|--------|
| Autoencoder pretraining on unlabeled data | +3-8% Dice | Low-moderate |
| Masked pixel pretraining (UNet) | +6-20% Dice | Moderate |
| Mean Teacher semi-supervised | +5-15% Dice | Moderate |
| Active learning (annotate 20 more images) | +5-10% Dice | Moderate (annotation time) |

### Tier 3: Advanced (2-4 weeks, potentially large improvement)

| Strategy | Expected Gain | Effort |
|----------|--------------|--------|
| BYOL contrastive pretraining | +3-10% Dice | Moderate-high |
| Cross Pseudo Supervision | +5-15% Dice | Moderate |
| Co-teaching for noise robustness | +2-5% Dice | Moderate |
| micro-SAM fine-tuning | Unknown | High (architecture change) |

### Recommended Combined Pipeline

```
Phase 1: Self-supervised pretraining
  - Pretrain UNet on millions of unlabeled images (AE reconstruction or masked pixel prediction)

Phase 2: Noise-robust supervised fine-tuning
  - Fine-tune on 19 labeled images with:
    - Pretrained encoder weights
    - Dice + Symmetric Cross-Entropy loss
    - Spatially varying label smoothing at boundaries
    - Copy-paste + CutMix augmentation

Phase 3: Semi-supervised training
  - Apply Mean Teacher with the pretrained model
  - Use consistency regularization on unlabeled batch alongside supervised loss on labeled batch

Phase 4: Active learning loop
  - Use MC Dropout uncertainty to select 20 most informative images from unlabeled pool
  - Annotate and add to training set
  - Repeat Phases 2-3

Phase 5: Inference
  - Test-time augmentation (flips + rotations, average predictions)
```

**Estimated total improvement:** From 0.81 Dice to 0.88-0.93 Dice, depending on which strategies are combined and how well they interact.

---

## Key References

### Self-Supervised Learning
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- Chen et al., "SimCLR" (ICML 2020); Grill et al., "BYOL" (NeurIPS 2020)
- "Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology" (2024)
- "Masked pretraining of U-Net for ultrasound image segmentation" (Scientific Reports, 2025)
- "Self-supervised pre-training with contrastive and masked autoencoder methods for small datasets" (Scientific Reports, 2023)

### Semi-Supervised Learning
- Tarvainen & Valpola, "Mean teachers are better role models" (NeurIPS 2017)
- Chen et al., "Cross Pseudo Supervision" (CVPR 2021)
- Sohn et al., "FixMatch" (NeurIPS 2020)
- SSL4MIS benchmark: github.com/HiLab-git/SSL4MIS

### Noise-Robust Training
- Islam & Glocker, "Spatially Varying Label Smoothing" (IPMI 2021)
- Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
- Han et al., "Co-teaching" (NeurIPS 2018)
- "Mean-Teacher-Assisted Confident Learning for Medical Image Segmentation" (IEEE TMI, 2022)

### Foundation Models
- Israel et al., "CellSAM" (Nature Methods, 2025)
- "Segment Anything for Microscopy (micro-SAM)" (2024)

### Active Learning
- "A comprehensive survey on deep active learning in medical image analysis" (Medical Image Analysis, 2024)
- "Selective Uncertainty-based Active Learning for Medical Image Segmentation" (2024)

### Data Augmentation
- Ghiasi et al., "Simple Copy-Paste" (CVPR 2021)
- "S3-TTA: Scale-Style Selection for Test-Time Augmentation in Biomedical Image Segmentation" (2023)
- "Test-time augmentation for deep learning-based cell segmentation on microscopy images" (Scientific Reports, 2020)
