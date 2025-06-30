# Style Transfer HiP-CT

Project to apply Style Transfer between HiP-CT and Clinical CT using CycleGANs + Preprocessing.

## Overview

This repository contains code and resources for performing style transfer between Hierachical Phase Contrast Tomography (HiP-CT) and Clinical CT images. The approach leverages CycleGANs for unpaired image-to-image translation, along with preprocessing routines to prepare the datasets. This model was trained and developed to work on Lung Datasets

## Features

- End-to-end preprocessing pipeline for HiP-CT and Clinical CT
- CycleGAN implementation for style transfer
- Utilities for data loading, augmentation, and result visualization
- Evaluation metrics and scripts

## Requirements


You can install the dependencies with:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:
    ```bash
    git clone https://github.com/LJMedPhys/Style_Transfer_HiP-CT.git
    cd Style_Transfer_HiP-CT
    ```

2. Segment your datasets:
    - Before you can start the training you need to segment you data set
    - For HiP-CT data you can the segmentation tool: https://github.com/LJMedPhys/HiP-CT-segmentator
    - For Clinical CT data the TotalSegmenator works: https://github.com/wasserth/TotalSegmentator

3. Preprocessing:

In the preprocessing folder open the config.yaml file and complete the entries for you dataset. HiP-CT and clinical CT each follow a seperate Preprocessing pipeline.

Pipeline Clinical CT:

![Preprocessing_clinical drawio(1)](https://github.com/user-attachments/assets/6a81bdb1-d8cb-4a78-b342-6390e1698469)


Pipeline HiP-CT:

![Reworked_Preprocessing_HiP-CT drawio(1)](https://github.com/user-attachments/assets/8357759f-c051-4e1e-aacd-1b40bb8ba671)




    ```bash
    python preprocess.py
    ```

4. Train the CycleGAN:
    ```bash
    python train.py --config configs/config.yaml
    ```

5. Generate style-transferred images:
    ```bash
    python test.py --config configs/config.yaml --weights checkpoints/latest.pth
    ```

## Repository Structure

```
.
├── data/                # Datasets and data preparation scripts
├── models/              # CycleGAN and related model code
├── utils/               # Utility scripts (metrics, visualization, etc.)
├── configs/             # Configuration files
├── preprocess.py        # Preprocessing script
├── train.py             # Training script
├── test.py              # Inference/testing script
├── requirements.txt     # Python dependencies
└── README.md
```

## References

- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
