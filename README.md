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

After setting all the parameters the preprocessing can be started with:
```bash
python run_preprocessing.py --config /path/to/config.yaml
```

4. Train the CycleGAN

To configure the training process the parameters need to configured in the config file provded in the configs_json folder:


| Parameter    | Type     | Description                                                        | 
| ------------ | -------- | ------------------------------------------------------------------ | 
| `name_run`  | `string`  | if weights and biases is installed this is the name of the run | 
| `device`  | `string` | Graphics card used for training | 
| `skip` | `boolean` | if True activates skip connections between encoder and decoder, not recommended to activate | 
| `batch_size` | `int`  | size of the batch | 
| `image_size` | `int`  | Path to the output directory where the preprocessed scans will be stored | 
| `test_split` | `float`  | percentage used for plotting during training (ignore) | 
| `learning_rate_G` | `float`  | Learning rate for the generators | 
| `learning_rate_D` | `float`  | Learning rate for the discriminators | 
| `n_residual_blocks` | `int`  | Number of residual blocks between encoding and decoding part | 
| `beta1` | `float` | decay parameter for first moment of adam optimizer | 
| `beta2` | `float` | decay parameter for second moment of adam optimizer |
| `epochs` | `200` | Epochs to train the network for |
| `lambda_A` | `float` | weighting parameter for the cycle loss of HiP-CT|
| `lambda_B` | `float` | weighting parameter for the cycle loss of clinical CT|
| `lambda_identity` | `float` | weighting parameter for the identity loss|
|`paths_A_trian`|`list of strings`| array of paths to the preprocessed hip-CT lungs in .h5 format|
|`paths_B_trian`|`list of strings`| array of paths to the preprocessed clincal CT lungs in .h5 format|

To start the training run the following command: 

```bash
    python train.py --config configs/config.yaml --load_checkpoint --log_wandb 
```

The last two flags are optional. load_checkpoint loads the last saved checkpoint from the checkpoint directory in the config files. log_wandb endables tracking of the training if weights and biases is installed: https://wandb.ai

5. Postprocessing

To translate the dataset and reapply the original segmentation removing any additional artifact the postprocessing.py can be applied: 

```bash
    python postprocessing.py --epoch 199 --output_path ./output.h5 --config ./configs_json/config_training.json --mask_in_vivo_test path/to/clinical/masks.h5 --mask_HiP_test path/to/HiP-CT/masks.h5
 ```

The epoch to load, the output path of the h5 file, the config file used for training and the paths to the masks of the test data set for the clinical CT and HiP-CT lung need to specified. 

The generated H5 file containes the input patch, the recoverd patches, the identity patches and the translated (fake) batches both raw and resegmented. 


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
