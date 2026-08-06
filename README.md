::: {align="center"}
# Attention-Augmented EfficientNet Transfer Learning for Four-Class Brain Tumour MRI Classification: Systematic Benchmarking and Attention Ablation Analysis

### A reproducible MATLAB framework for benchmarking attention-augmented transfer learning models for four-class brain tumour MRI classification.

![MATLAB](https://img.shields.io/badge/MATLAB-R2026a-orange)
![License](https://img.shields.io/badge/License-MIT-green) ![Deep
Learning](https://img.shields.io/badge/Deep-Learning-blue) ![Medical
Imaging](https://img.shields.io/badge/Medical-Imaging-red)
![Status](https://img.shields.io/badge/Status-Manuscript%20in%20Preparation-yellow)
:::

------------------------------------------------------------------------

# Table of Contents

-   Project Overview
-   Repository Highlights
-   Repository Contents
-   Repository Structure
-   Dataset
-   Software Requirements
-   Installation
-   Training
-   Evaluation
-   Results Summary
-   Repository Notes
-   Future Work
-   Contact
-   License

------------------------------------------------------------------------

# Project Overview

This repository contains the complete implementation, trained models,
evaluation pipeline and experimental results for the research
manuscript:

> **Attention-Augmented EfficientNet Transfer Learning for Four-Class
> Brain Tumour MRI Classification: Systematic Benchmarking and Attention
> Ablation Analysis**

The objective of this work is to investigate whether lightweight
attention mechanisms can improve transfer-learning performance for
automated brain tumour MRI classification while maintaining
computational efficiency and reproducibility.

Seven transfer learning architectures are benchmarked under identical
experimental conditions. In addition to comparing different CNN
backbones, the work performs a systematic ablation study that
independently evaluates Spatial Attention (SA), Channel Attention (CA),
and the complete Convolutional Block Attention Module (CBAM).

The repository is intended to serve as a reproducible research resource
by providing MATLAB implementations, trained models (where repository
limits permit), raw experimental results, performance metrics, and
publication assets.

------------------------------------------------------------------------

# Repository Highlights

-   Systematic benchmarking of seven deep learning architectures.
-   Independent evaluation of Spatial Attention, Channel Attention and
    CBAM.
-   Five independent training runs for every architecture.
-   Mean ± standard deviation reported for all evaluation metrics.
-   Publication-quality experimental results.
-   MATLAB implementation designed for reproducibility.

------------------------------------------------------------------------

# Repository Contents

  Resource                           Available
  -------------------------- --------------------------
  MATLAB Source Code                     ✅
  Trained Models                         ✅
  Raw Experimental Results               ✅
  Performance Metrics                    ✅
  Publication Figures                    ✅
  Manuscript Resources                   ✅
  Public Dataset              ❌ (Download separately)

------------------------------------------------------------------------

# Repository Structure

``` text
BrainTumorMRI-AttentionBenchmark
│
├── Dataset/
├── Images/
├── Paper/
├── Results/
│   ├── Figures/
│   ├── Metrics/
│   └── Raw Results/
├── Scripts/
│   ├── AttentionLayers/
│   ├── Evaluation/
│   ├── Train/
│   └── Utilities/
├── TrainedModels/
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

  -----------------------------------------------------------------------
  Folder                        Description
  ----------------------------- -----------------------------------------
  Scripts                       MATLAB implementation for training,
                                evaluation and utilities

  Results                       Raw results, performance tables and
                                figures

  TrainedModels                 Saved model checkpoints (except VGG16)

  Paper                         Manuscript resources

  Images                        Repository assets
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Dataset

Experiments were conducted using the **Brain Tumour MRI Dataset**
released by **Bhuvaji et al.(2020)**.

**Dataset statistics**

-   Total MRI images: **3,264**
-   Glioma: **926**
-   Meningioma: **937**
-   Pituitary Tumour: **901**
-   No Tumour: **500**

The dataset contains both T1-weighted and T2-weighted MRI scans and
provides a realistic multi-class classification benchmark.

**Dataset is not redistributed through this repository.**

Please download it directly from:

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

------------------------------------------------------------------------

# Software Requirements

-   MATLAB R2026a
-   Deep Learning Toolbox
-   Image Processing Toolbox
-   Computer Vision Toolbox

------------------------------------------------------------------------

# Installation

1.  Clone this repository.
2.  Download the dataset from Kaggle.
3.  Organize the dataset as expected by the training scripts.
4.  Add the repository to the MATLAB path.
5.  Run the desired training or evaluation script.

------------------------------------------------------------------------

# Training

Training scripts are available in:

``` text
Scripts/Train/
```

All models were trained using identical experimental settings to ensure
fair benchmarking.

------------------------------------------------------------------------

# Evaluation

Evaluation scripts are available in:

``` text
Scripts/Evaluation/
```

The repository includes both the complete raw experimental results and
the final aggregated performance metrics reported in the manuscript.

------------------------------------------------------------------------

# Results Summary

    Rank Model                 Accuracy (%)
  ------ ------------------- --------------
       1 EfficientNet-SA       96.40 ± 0.79
       2 EfficientNet-CA       96.07 ± 0.88
       3 EfficientNet-B0       96.02 ± 0.50
       4 EfficientNet-CBAM     95.78 ± 0.82
       5 ResNet50              90.89 ± 0.85
       6 GoogLeNet             89.64 ± 2.34
       7 VGG16                 71.78 ± 1.39

All reported metrics represent the mean and standard deviation obtained
from five independent experimental runs.

------------------------------------------------------------------------

# Repository Notes

> **Important**
>
> The VGG16 checkpoint is intentionally omitted because it exceeds
> GitHub's maximum file size limit (100 MB). All remaining trained
> models are included.

------------------------------------------------------------------------

# Future Work

Future work will focus on validating the proposed framework using larger
multi-center clinical datasets to improve generalizability across
diverse imaging protocols. Planned extensions include MRI tumour
grading, automatic tumour segmentation, lightweight model compression
for edge deployment, and comprehensive external validation to support
reliable clinical translation.

------------------------------------------------------------------------

# Contact

**Mrityunjay Tiwari**

-   GitHub: https://github.com/Sci-Mrityunjay-Tiwari
-   Repository:
    https://github.com/Sci-Mrityunjay-Tiwari/BrainTumorMRI-EfficientNet-SA
-   LinkedIn: https://www.linkedin.com/in/mrityunjay-tiwari-bb3767229/
-   Email: mrityunjay.tiwari.sci@gmail.com

------------------------------------------------------------------------

# License

This repository is released under the MIT License.

See the `LICENSE` file for complete licensing information.

------------------------------------------------------------------------

::: {align="center"}
**If you find this repository useful for your research, please consider
starring the repository.**
:::
