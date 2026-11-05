# Dataset Sources and Directory Structures

This document describes the datasets used in the brain tumour MRI classification research, including their directory structures, class distributions, sources, and relevant data-integrity considerations.

**Important:** Dataset files are not included in this GitHub repository. Users must obtain the datasets independently from their original sources and comply with the applicable licensing, attribution, and access requirements.

---

## 1. Bhuvaji Brain Tumour MRI Dataset

**Source:** Kaggle – Brain Tumor Classification (MRI)  
**Dataset URL:** https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

### Dataset Overview

- **Total images:** 3,264
- **Number of classes:** 4
- **Image format:** MRI images
- **Image dimensions used in the research:** 224 × 224 pixels
- **Input channels:** RGB (grayscale images are replicated across three channels)

### Class Distribution

| Class | Number of Images |
|---|---:|
| Glioma | 926 |
| Meningioma | 937 |
| Pituitary | 901 |
| No Tumour | 500 |
| **Total** | **3,264** |

### Research Dataset Split

The original dataset was partitioned at image level using a fixed random seed (`rng(42)`).

| Class | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Glioma | 648 | 139 | 139 | 926 |
| Meningioma | 656 | 141 | 140 | 937 |
| Pituitary | 631 | 135 | 135 | 901 |
| No Tumour | 350 | 75 | 75 | 500 |
| **Total** | **2,285** | **490** | **489** | **3,264** |

### Original Directory Structure

The original Kaggle dataset is commonly distributed using the following directory organization:

```text
Dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/
```

The research workflow uses its own training, validation, and testing partition. Therefore, the original directory structure should not be confused with the final experimental split.

### Important Note

The partitioning was performed at image level because patient identifiers were unavailable. Consequently, patient-independent separation cannot be established.

---

## 2. BRISC 2025 – Brain Tumour MRI Dataset

**Dataset:** BRISC (BRain tumor Image Segmentation & Classification)  
**Reference:** Fateh et al., *BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet*.  
**Preprint:** arXiv:2506.14318  
**URL:** https://arxiv.org/abs/2506.14318

### Dataset Overview

BRISC is a curated, expert-annotated T1-weighted MRI dataset designed for brain tumour classification and segmentation.

- **Total images:** 6,000
- **Training images:** 5,000
- **Testing images:** 1,000
- **Number of classes:** 4
- **Imaging modality:** T1-weighted MRI
- **Anatomical planes:** Axial, Coronal, and Sagittal
- **Segmentation annotations:** Pixel-wise masks reviewed by medical experts

### Classification Categories

1. Glioma
2. Meningioma
3. Pituitary
4. No Tumour

### Directory Structure

The BRISC dataset contains separate directories for classification and segmentation tasks.

```text
BRISC2025/
├── classification_task/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── segmentation_task/
│   ├── images/
│   └── masks/
│
├── manifest.json
├── manifest.csv
├── manifest.json.sha256
├── manifest.csv.sha256
└── README.md
```

### Directory Description

| Directory | Description |
|---|---|
| `classification_task/` | Contains class-labelled MRI images for image classification. |
| `segmentation_task/images/` | Contains MRI images used for segmentation. |
| `segmentation_task/masks/` | Contains corresponding binary segmentation masks. |
| `manifest.json` | Dataset manifest in JSON format. |
| `manifest.csv` | Dataset manifest in CSV format. |

### File Naming Convention

BRISC uses the following filename structure:

```text
brisc2025_<split>_<index>_<tumor>_<view>_<sequence>.<ext>
```

The filename components represent:

| Component | Description |
|---|---|
| `brisc2025` | Dataset identifier |
| `split` | Training or testing partition |
| `index` | Zero-padded image index |
| `tumor` | Tumour class abbreviation |
| `view` | Anatomical plane |
| `sequence` | MRI sequence |

Class abbreviations:

- `gl` – Glioma
- `me` – Meningioma
- `pi` – Pituitary
- `nt` – No Tumour

Anatomical plane abbreviations:

- `ax` – Axial
- `co` – Coronal
- `sa` – Sagittal

Example:

```text
brisc2025_test_00010_gl_ax_t1.jpg
```

The corresponding segmentation mask has the same basename but uses the `.png` extension.

### External Evaluation

The research uses the **1,000-image BRISC testing partition** for external evaluation.

The classification task is used for four-class image classification. The segmentation masks are separate annotations and are not used as classification labels.

### Citation

Fateh, A., Rezvani, Y., Moayedi, S., Rezvani, S., Fateh, F., Fateh, M., and Abolghasemi, V. (2025). *BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet*. arXiv preprint arXiv:2506.14318.

---

## 3. BDNeuro-MRI Dataset

**Dataset:** BDNeuro-MRI – A Bangladeshi Clinical Brain Tumour MRI Dataset  
**Dataset record:** Mendeley Data  
**DOI:** 10.17632/zwr4ntf94j.9  
**URL:** https://doi.org/10.17632/zwr4ntf94j.9

### Dataset Overview

BDNeuro-MRI is a clinical brain tumour MRI dataset containing four classification categories.

- **Total images:** 5,941
- **Training images:** 4,160
- **Validation images:** 892
- **Testing images:** 889
- **Number of classes:** 4
- **Imaging modality:** T1-weighted contrast-enhanced MRI

### Class Distribution

| Class | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| Glioma | 1,534 | 329 | 328 | 2,191 |
| Meningioma | 1,035 | 222 | 221 | 1,478 |
| No Tumour | 551 | 118 | 118 | 787 |
| Pituitary | 1,040 | 223 | 222 | 1,485 |
| **Total** | **4,160** | **892** | **889** | **5,941** |

### Directory Structure

```text
Brain_Tumor_MRI_Dataset_Final/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

### Preprocessing and Dataset Integrity

According to the supplied dataset documentation, the dataset preparation involved:

1. Combining the original training and testing images into a single image pool.
2. Removing exact byte-identical duplicate files using MD5 hashing.
3. Removing near-duplicate images using perceptual hashing (`imagehash.phash`) with a Hamming distance threshold of ≤5.
4. Creating training, validation, and testing partitions using stratified sampling with a fixed random seed of 42.
5. Checking for image-content overlap between the three partitions.
6. Renaming files using anonymized, sequential filenames.

The dataset documentation reports zero image-content overlap between its own training, validation, and testing partitions.

**Important limitation:** Image-level duplicate removal does not establish patient-level independence. Different images from the same patient or study may remain across partitions if they are not visually near-identical.

### External Evaluation

The research uses the **889-image BDNeuro-MRI testing partition** for external evaluation.

### Citation and Licensing

Dataset DOI: `10.17632/zwr4ntf94j.9`

The supplied dataset README does not provide a completed licence field or a finalized preferred citation statement. Users should consult the original Mendeley Data record for the current citation and applicable reuse conditions before using or redistributing the dataset.

---

## 4. Cross-Dataset Overlap Audit

An image-level overlap audit was conducted to examine potential similarities between the external datasets (BRISC and BDNeuro-MRI) and the original Bhuvaji dataset.

Two methods were used:

### 4.1 Exact Duplicate Detection

SHA-256 file-byte hashing was used to identify byte-identical image files.

### 4.2 Perceptual Duplicate Screening

A 64-bit Discrete Cosine Transform (DCT)-based perceptual hashing method was used to identify visually similar images.

A Hamming distance threshold of ≤5 was used to flag potential perceptual matches.

### Interpretation and Limitations

The following distinctions are important when interpreting the audit:

- Exact hash matches indicate byte-identical files.
- Perceptual-hash matches indicate similarity candidates and do not automatically establish confirmed duplicates.
- Similarity does not establish that images belong to the same patient.
- The external images were compared against the complete Bhuvaji collection.
- The audit does not independently establish whether a matching Bhuvaji image belonged to the training, validation, or testing partition.
- Patient identifiers were unavailable; therefore, patient-level independence cannot be established.

The overlap audit should be interpreted as an image-level integrity assessment, not as proof of patient-level separation or training-set exposure.

---

## 5. General Data-Handling Guidelines

1. Dataset files are not distributed through this GitHub repository.
2. Download each dataset from its original source.
3. Keep downloaded datasets outside the Git repository.
4. Do not upload MRI images, segmentation masks, or other restricted dataset files.
5. Follow the original dataset's licensing, attribution, and access requirements.
6. Update local MATLAB dataset paths according to the actual directory locations on your system.
7. Do not assume that different datasets are independent merely because their original providers use separate train/test partitions.

---

**Note:** This repository contains the MATLAB source code and documentation associated with the research. Dataset files, trained model weights, generated experimental results, and manuscript figures are maintained separately.