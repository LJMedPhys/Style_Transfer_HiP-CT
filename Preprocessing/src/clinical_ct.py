import os
from src.tools import (
    load_dicom_scan_to_np,
    dicom_2_HU,
    align_nifti_seg,
    apply_segment_dicom_np,
    resample_np_2d_dicom,
    split_image_into_tiles_and_write,
    np_to_h5
)
import nibabel as nib

def run(input_dir, mask_dir, output_dir, output_mask_dir, spacing, window_size, stride, padding, threshold, percentile, cut_off, steps):
    """
    Processes a set of DICOM files and their corresponding segmentation masks through a configurable preprocessing pipeline.
    Args:
        input_dir (str): Directory containing input DICOM folders.
        mask_dir (str): Directory containing segmentation masks in NIfTI format (.nii).
        output_dir (str): Directory to save processed image data in HDF5 format.
        output_mask_dir (str): Directory to save processed mask data in HDF5 format.
        spacing (tuple or list): Target spacing for resampling (e.g., (1.0, 1.0)).
        window_size (int or tuple): Size of the window for image tiling/slicing.
        stride (int or tuple): Stride for sliding window during tiling/slicing.
        padding (float): Padding value used for normalization and scaling.
        threshold (float): Value which is checked for removing empty tiles
        percentile (float): Maximum allow percentage of pixels with threshold value per tile, tiles below get deleted.
        cut_off (float): Maximum value for intensity clipping.
        steps (dict): Dictionary specifying which preprocessing steps to apply. Keys may include:
            - "HU_scaling" (bool): Whether to convert DICOM to Hounsfield Units.
            - "segmentation" (bool): Whether to apply segmentation mask.
            - "clipping" (bool): Whether to clip intensities at cut_off.
            - "min_max_scaling" (bool): Whether to apply min-max normalization.
            - "adjust_spacing" (bool): Whether to resample to target spacing.
            - "slicing" (bool): Whether to split images into tiles.
    Returns:
        None
    Side Effects:
        Saves processed image and mask data as HDF5 files in the specified output directories.
        Prints progress and status messages to the console.
    """
    
    paths_dicoms = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]

    for k in range(len(paths_dicoms)):

        name_dicom = paths_dicoms[k]

        print(f"Processing dicom file {name_dicom}")

        path_dicom = os.path.join(input_dir, name_dicom)
        path_seg = os.path.join(mask_dir, name_dicom+'.nii')

        print("Loading Dicom into numpy array")

        data_array = load_dicom_scan_to_np(path_dicom)

        if steps.get("HU_scaling", True):
            print("Scaling enabled")
            data_array = dicom_2_HU(path_dicom)
        else:
            print("Skipping scaling")

        if steps.get("segmentation", True):
            print("Segmentation enabled")
            img_seg = nib.load(path_seg)
            seg_array = img_seg.get_fdata()                    
            seg_array = align_nifti_seg(seg_array)
            data_array = apply_segment_dicom_np(data_array, path_seg, padding)
        else:
            print("Skipping Segmentation")

        if steps.get("clipping", True):
            print("Clipping enabled")
            data_array[data_array>cut_off] = cut_off
        else:
            print("Skipping clipping")
    
        if steps.get("min_max_scaling", True):
            print("min_max_scaling enabled")
            data_array = (data_array-(padding))/(cut_off-(padding))
            threshold = (threshold-(padding))/(cut_off-(padding))
        else:
            print("Skipping min_max_scaling")

        if steps.get("adjust_spacing", True):
            data_array = resample_np_2d_dicom(data_array, path_dicom, spacing)
            if steps.get("segmentation", True):
                seg_array = resample_np_2d_dicom(seg_array, path_dicom, spacing)
        else:
            print("Skipping spacing adjustment")

        if steps.get("slicing", True):
            print("Slicing enabled")
            if steps.get("segmentation", True):
                data_array, seg_array = split_image_into_tiles_and_write(data_array, seg_array, window_size, stride, threshold, percentile)
            else:
                data_array, _ = split_image_into_tiles_and_write(data_array, None, window_size, stride, threshold, percentile)
        else:
            print("Skipping slicing")

        print("Saving to h5")
    
        np_to_h5(data_array, path_dicom, output_dir)

        if steps.get("segmentation", True):
            np_to_h5(seg_array, path_dicom, output_mask_dir)
        
        print(f'Completed Processing {name_dicom}')




