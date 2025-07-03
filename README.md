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

![Pipelines_preproc_together drawio](https://github.com/user-attachments/assets/58e71d04-451a-4423-852a-f8827dfb1e7b)

Starting with HiP-CT pipeline, the following parameters need to be filled out

| Parameter    | Type     | Description                                                        | 
| ------------ | -------- | ------------------------------------------------------------------ | 
| `enable`  | `bool`  | set to true if you want to apply hip_ct preprocessing | 
| `input_dir`  | `string` | Path to the input directory containing the HiP-CT scan                | 
| `mask_dir` | `string` | Path to the input directory containing the masks for the scans from the previous step| 
| `output_dir` | `string`  |Path to the output directory where the preprocessed scans will be stored | 
| `output_mask_dir`    | `string`   | Path to the output directory where the masks correspond to the preprocessed scans will be stored|
| `spacing`  | `list floats` | pixel sizes in mm for the [x,y]-Dimension that the input will be scaled|
| `window size`  | `int`  | The input images are sliced into tiles using a sliding window, this determines the size of the window in pixels, must be smaller or equal to the input image| 
| `stride`  | `int`  | Distance the window travels in pixels | 
| `threshold`  | `float`  | After slicing the input, empty slices are removed. This value determines which value counts as empty | 
| `percentile`  | `float`  | between [0,1), describes the maximum percentage of pixels in a slice that are allowed to be empty | 
| `clip_values`  | `list of floats`  | after standardscaling the pixel distribution of the input scan can be clipped within this range. After standard scaling, the unit becomes standard deviations| 
| `standard_scaling`  | `bool`  | activates if the scan is standard scaled (x-mean)/standard_deviation | 
| `segmentation`  | `bool`  | activates the application of the mask | 
| `min_padding`  | `bool`  | if activated the empty pixels are replaced with the minimum of the distribution | 
| `adjust spacing`  | `bool`  | activates adjustment of pixel size | 
| `clipping`  | `bool`  | activates clipping of the ends of the distribution| 
| `min_max_scaling`  | `bool`  | activates min_max scaling after standard scaling | 
| `slicing`  | `bool`  | activates sclicing into tiles | 


For Clinical CT the parameters are similar with some differences due to the modality:

| Parameter    | Type     | Description                                                        | 
| ------------ | -------- | ------------------------------------------------------------------ | 
| `enable`  | `bool`  | set to true if you want to apply clinical ct preprocessing | 
| `input_dir`  | `string` | Path to the input directory containing the clinical CT dicoms| 
| `mask_dir` | `string` | Path to the input directory containing the masks for the scans from the previous step| 
| `output_dir` | `string`  |Path to the output directory where the preprocessed scans will be stored | 
| `output_mask_dir`    | `string`   | Path to the output directory where the masks correspond to the preprocessed scans will be stored|
| `spacing`  | `list floats` | pixel sizes in mm for the [x,y]-Dimension that the input will be scaled|
| `window size`  | `int`  | The input images are sliced into tiles using a sliding window, this determines the size of the window in pixels, must be smaller or equal to the input image| 
| `stride`  | `int`  | Distance the window travels in pixels | 
| `threshold`  | `float`  | After slicing the input, empty slices are removed. This value determines which value counts as empty | 
| `padding`  | `float`  | Value the empty pixels are replaced with (in HU) | 
| `cutoff`  | `float`  | limit of the upper end of the distribution that will be clipped | 
| `percentile`  | `float`  | between [0,1), describes the maximum percentage of pixels in a slice that are allowed to be empty | 
| `HU_scaling`  | `bool`  | activates if the scan is standard scaled to Houndsfield units using the meta data of the DICOM files | 
| `min_max_scaling`  | `bool`  | activates min_max scaling after HU scaling | 
| `adjust spacing`  | `bool`  | activates adjustment of pixel size | 
| `segmentation`  | `bool`  | activates the application of the mask | 
| `clipping`  | `bool`  | activates clipping of the upper end of the distribution| 
| `slicing`  | `bool`  | activates sclicing into tiles | 


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
