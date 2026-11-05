# Attention-Augmented EfficientNet Transfer Learning for Four-Class Brain Tumour MRI Classification

**MATLAB Research Implementation | Deep Learning | Transfer Learning | Attention Mechanisms**

## 1. Overview

This repository contains the MATLAB source code developed for a research study on four-class brain tumour classification using Magnetic Resonance Imaging (MRI).

The research investigates the application of transfer learning and attention mechanisms to improve the classification of brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary Tumour
- No Tumour

The primary focus is on evaluating attention-enhanced deep learning architectures, particularly EfficientNet-B0 integrated with attention mechanisms. The study also investigates the individual contributions of channel attention and spatial attention through attention ablation experiments.

The research examines multiple CNN architectures under the experimental configurations documented in the associated research work.

### Research Manuscript Title

**Attention-Augmented EfficientNet Transfer Learning for Four-Class Brain Tumour MRI Classification: Systematic Benchmarking, Attention Ablation, and Overlap-Audited External Evaluation**

---

## 2. Research Objectives

The primary objectives of this research are:

1. Investigate transfer-learning-based CNN architectures for four-class brain tumour MRI classification.
2. Implement attention mechanisms to investigate their contribution to feature representation and classification.
3. Integrate channel and spatial attention mechanisms with EfficientNet-B0.
4. Conduct attention ablation experiments to examine the individual and combined contributions of attention components.
5. Evaluate classification performance using appropriate quantitative metrics.
6. Investigate external dataset evaluation and potential image-level overlap between datasets.
7. Examine model interpretability using explainability techniques.

---

## 3. Models and Architectures

The research implementation includes the following deep learning architectures and attention configurations:

| Architecture | Description |
|---|---|
| EfficientNet-B0 | Transfer-learning-based CNN architecture |
| EfficientNet-B0 + Channel Attention | EfficientNet-B0 integrated with channel attention |
| EfficientNet-B0 + Spatial Attention | EfficientNet-B0 integrated with spatial attention |
| EfficientNet-B0 + CBAM | EfficientNet-B0 integrated with a Convolutional Block Attention Module |
| ResNet50 | Transfer-learning-based CNN |
| GoogLeNet | Transfer-learning-based CNN |
| VGG16 | Transfer-learning-based CNN |

The attention mechanisms investigated include:

- **Channel Attention (CA):** Focuses on recalibrating feature channels according to their importance.
- **Spatial Attention (SA):** Focuses on identifying informative spatial regions within feature maps.
- **Convolutional Block Attention Module (CBAM):** Combines channel and spatial attention mechanisms.
- **Adaptive Attention:** An additional attention implementation developed to investigate adaptive feature refinement.

The presence of these implementations in the repository does not imply that every architecture was trained under identical hyperparameters or experimental conditions. Refer to the individual MATLAB scripts and research manuscript for the relevant configurations.

---

## 4. Repository Structure

The repository is organized into the following directories:

```text
BrainTumorMRI-AttentionBenchmark/
│
├── Dataset_Structure/
│   └── README.md
│
├── Manuscript/
│   └── README.md
│
├── Scripts/
│   │
│   ├── Attention_Layers_Scripts/
│   │   ├── AdaptiveAttentionLayer.m
│   │   ├── ChannelAttentionLayer.m
│   │   └── SpatialAttentionLayer.m
│   │
│   ├── Train_Scripts/
│   │   ├── train_efficientnetb0.m
│   │   ├── train_efficientnet_cbam.m
│   │   ├── train_resnet50.m
│   │   ├── train_googlenet.m
│   │   └── train_vgg16.m
│   │
│   ├── Evaluation_Scripts/
│   │   └── Evaluation and analysis scripts
│   │
│   └── Utilities_Scripts/
│       └── Supporting MATLAB utility functions
│    
│├── TrainedModels/
│   └── Trained MATLAB model files (.mat)
│
├── LICENSE
└── README.md
```

*Note: The directory tree above illustrates the main organization and representative scripts. Refer to the actual repository directories for the complete list of MATLAB files.*

### Directory Description

| Directory | Description |
|---|---|
| `Attention_Layers_Scripts/` | Custom MATLAB implementations of channel, spatial, and adaptive attention layers. |
| `Train_Scripts/` | Scripts for constructing and training the investigated deep learning architectures. |
| `Evaluation_Scripts/` | Scripts for model evaluation, performance analysis, and experimental assessment. |
| `Utilities_Scripts/` | Supporting functions used across training and evaluation workflows. |
| `Dataset_Structure/` | Dataset information, directory structures, class distributions, and data-integrity documentation. |
| `Manuscript/` | Publication information and the eventual manuscript reference. |

---

## 5. Datasets

Three brain MRI datasets are associated with this research.

### 5.1 Bhuvaji Brain Tumour MRI Dataset

The Bhuvaji dataset is used for the primary model development and internal evaluation.

- Total images: 3,264
- Glioma: 926
- Meningioma: 937
- Pituitary: 901
- No Tumour: 500

The research uses an image-level partition:

| Partition | Number of Images |
|---|---:|
| Training | 2,285 |
| Validation | 490 |
| Testing | 489 |
| **Total** | **3,264** |

**Dataset source:**  
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

### 5.2 BRISC 2025

BRISC is an expert-annotated T1-weighted MRI dataset containing:

- Total images: 6,000
- Training images: 5,000
- Testing images: 1,000
- Four classification categories
- Axial, coronal, and sagittal anatomical planes
- Additional segmentation annotations

The 1,000-image testing partition is used for external evaluation in this research.

**Dataset reference:** Fateh et al. (2025), BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet.

https://arxiv.org/abs/2506.14318

### 5.3 BDNeuro-MRI

BDNeuro-MRI is a Bangladeshi clinical brain tumour MRI dataset.

- Total images: 5,941
- Training images: 4,160
- Validation images: 892
- Testing images: 889
- Four classification categories

The 889-image testing partition is used for external evaluation in this research.

**Dataset source:**  
https://doi.org/10.17632/zwr4ntf94j.9

### Dataset Availability

The MRI datasets are **not included** in this repository.

Users must obtain the datasets independently from their respective sources and comply with the original providers' licensing, attribution, and access requirements.

For detailed information about dataset organization, class distributions, and data-integrity considerations, refer to:

[`Dataset_Structure/README.md`](Dataset_Structure/README.md)

---

## 6. Software and Requirements

The project was developed using:

- MATLAB R2026a
- Deep Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox

Additional functionality or support packages may be required depending on the selected architecture and MATLAB script.

The scripts may require modifications to local dataset paths and configuration parameters before execution.

---

## 7. Getting Started

### Step 1: Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/Sci-Mrityunjay-Tiwari/BrainTumorMRI-AttentionBenchmark.git
```

### Step 2: Obtain the Datasets

Download the required datasets from their respective original sources.

The datasets must be organized locally according to the directory structures described in:

`Dataset_Structure/README.md`

The dataset files are not downloaded automatically by the MATLAB scripts.

### Step 3: Open MATLAB

Open MATLAB and navigate to the cloned repository directory.

### Step 4: Configure the MATLAB Path

Add the required script directories to the MATLAB search path.

For example:

```matlab
addpath(genpath(pwd));
```

### Step 5: Configure Dataset Locations

Review the selected training or evaluation script and update the dataset paths according to your local directory organization.

### Step 6: Execute the Required Script

Select the relevant MATLAB script from the appropriate directory:

- `Scripts/Attention_Layers_Scripts/`
- `Scripts/Train_Scripts/`
- `Scripts/Evaluation_Scripts/`
- `Scripts/Utilities_Scripts/`

Review the script configuration and dependencies before execution.

**Important:** These scripts constitute research implementations and may require local configuration. The repository does not guarantee that every script will execute without modification across different MATLAB installations.

---

## 8. Evaluation and Explainability

The research incorporates quantitative evaluation and analysis of classification performance.

The evaluation workflow includes relevant classification metrics and confusion-matrix-based assessment.

The research also investigates explainability techniques, including Grad-CAM-based visualization, to examine model attention and feature localization.

### External Dataset Integrity Assessment

An image-level overlap audit was conducted between the Bhuvaji dataset and the external datasets.

The audit incorporated:

1. SHA-256 file-byte hashing for exact duplicate detection.
2. 64-bit DCT-based perceptual hashing.
3. Hamming-distance-based screening to identify potentially similar images.

The perceptual-hash threshold used for candidate screening was a Hamming distance of ≤5.

**Interpretation limitations:**

- Exact hash matches indicate byte-identical files.
- Perceptual-hash matches are similarity candidates, not automatically confirmed duplicates.
- Similarity does not establish that images belong to the same patient.
- The external datasets were compared against the complete Bhuvaji collection.
- The audit does not independently establish whether a matching Bhuvaji image was present in the training, validation, or testing partition.
- Patient identifiers were unavailable; therefore, patient-level independence cannot be established.

These limitations are important when interpreting external evaluation results.

---

## 9. Reproducibility and Research Limitations

The following considerations should be taken into account when interpreting or reproducing the research:

1. **Image-level partitioning:** The Bhuvaji dataset was partitioned at image level because patient identifiers were unavailable.
2. **Patient-level independence:** Patient-independent separation cannot be established.
3. **Cross-dataset overlap:** Exact and perceptual similarity audits provide image-level evidence but do not establish patient-level independence.
4. **Training configurations:** Different architecture scripts may use different training configurations. Identical training conditions should not be assumed without verifying the relevant scripts and experimental records.
5. **Model provenance:** Experimental predictions and trained-model provenance should be verified against the corresponding training records before being used for further comparisons.
6. **Explainability:** Grad-CAM visualizations provide qualitative interpretability and should not independently be treated as proof of clinical localization or model reliability.

For complete experimental methodology, verified results, and detailed discussion, refer to the associated research manuscript when available.

---

## 10. Repository Contents and Data Availability

This repository provides the MATLAB source code, trained model files, and supporting documentation associated with the research.

The repository includes:

- MATLAB implementations of attention mechanisms.
- Training scripts for the investigated deep learning architectures.
- Evaluation and utility scripts.
- Trained MATLAB model files (`.mat`) in the `TrainedModels/` directory.
- Dataset descriptions and directory structures.
- Research manuscript information.

**Model Availability:** Trained model files are provided for the implemented architectures, except VGG16, whose trained weights are not included.

**Dataset Availability:** Original MRI datasets are not distributed through this repository. Users must obtain them independently from their respective sources..

---

## 11. Manuscript and Publication

The associated research manuscript is currently being prepared for publication.

The manuscript citation, journal information, DOI, and publication link will be added to the `Manuscript/` directory once the publication details are confirmed.

No publication DOI or acceptance status is claimed at this stage.

---

## 12. Author

**Mrityunjay Tiwari**

Research Interests:
- Robotic Systems-Morphology and Locomotion/Autonomy
- Deep Learning and Medical Image Analysis
- Attention Mechanisms
- Computer Vision

GitHub: https://github.com/Sci-Mrityunjay-Tiwari

LinkedIn: https://www.linkedin.com/in/mrityunjay-tiwari-bb3767229/

---

## License

The source code is distributed under the terms specified in the repository's `LICENSE` file.

The datasets referenced in this project are subject to their respective providers' licensing and usage conditions. The repository's source-code licence does not override or replace dataset-specific terms.