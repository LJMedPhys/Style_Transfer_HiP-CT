import os
import h5py
from src.tools import (
    h5_to_np,
    load_jp2_scan_to_np,
    contrast_stretch_with_clipping,
    resample_np_2d_jp2,
    split_image_into_tiles_and_write,
    np_to_h5
)


def run(input_dir, mask_dir, output_dir, output_mask_dir, spacing, window_size, stride, threshold, percentile, clip_values, steps):
    """
    Processes CT images and their segmentation masks with optional standard scaling, clipping, resampling, and slicing.

    Parameters:
        input_dir (str): Path to the input CT image file or directory.
        mask_dir (str): Path to the segmentation mask file.
        output_dir (str): Directory to save the processed CT images.
        output_mask_dir (str): Directory to save the processed segmentation masks.
        spacing (float or tuple): Desired pixel spacing for resampling.
        window_size (int or tuple): Size of the window for slicing the image.
        stride (int or tuple): Stride for window slicing.
        threshold (float): Threshold value for slicing or mask processing.
        percentile (float): Percentile value for intensity normalization or slicing.
        clip_values (tuple): Tuple of (min, max) values for intensity clipping.
        steps (dict): Dictionary specifying which processing steps to apply.

    Returns:
        None
    """

    # Validate clip_values
    if not (isinstance(clip_values, (tuple, list)) and len(clip_values) == 2):
        raise ValueError("clip_values must be a tuple or list with exactly two elements.")

    # Loading dataset

    print(f'Loading dataset {os.path.basename(input_dir)}')

    if os.path.basename(input_dir).endswith('.h5'):
        jp2_array = h5_to_np(input_dir)
    else:
        jp2_array = load_jp2_scan_to_np(input_dir)

    name_ds = os.path.basename(input_dir)

    # Segmentation mask loading and standard scaling

    if steps.get("segmentation", True) and steps.get("standard_scaling", True):
        print("Loading segmentation mask and applying standard scaling")
        with h5py.File(mask_dir, 'r') as h5_file:
            seg_array = h5_file[name_ds][:]
        
        foreground = jp2_array[seg_array == 1]
        mean = np.mean(jp2_array[seg_array==1])
        std = np.std(jp2_array[seg_array==1])

        print('mean ',mean, 'std ', std)

        if std == 0:
            print("Warning: Standard deviation is zero. Skipping standardization.")
            jp2_array = jp2_array - mean
        else:
            jp2_array = (jp2_array-mean)/std

        jp2_array[seg_array==0] = np.min(jp2_array)
        jp2_array = (jp2_array-mean)/std

        jp2_array[seg_array==0] = np.min(jp2_array)

    elif steps.get("segmentation", True) and steps.get("standard_scaling", False):
        print("Loading segmentation mask without standard scaling")

        with h5py.File(mask_dir, 'r') as h5_file:
            seg_array = h5_file[name_ds][:]
        
        jp2_array[seg_array==0] = np.min(jp2_array)
        mean = np.mean(jp2_array)
        std = np.std(jp2_array)

        print('mean ',mean, 'std ', std)

        if std == 0:
            print("Warning: Standard deviation is zero. Skipping standardization.")
        else:
            jp2_array = (jp2_array-mean)/std
        print('mean ',mean, 'std ', std)

        jp2_array = (jp2_array-mean)/std
    else:
        print("Skipping segmentation and standard scaling")

        seg_array = None
    
    # Apply clipping if specified
    if steps.get("clipping", True):
        print(f"Clipping values to {clip_values}")
        jp2_array = contrast_stretch_with_clipping(jp2_array, clip_values[0], clip_values[1])
    else:
        print("Skipping clipping")  

    # Apply resampling if specified
    if steps.get("adjust_spacing", True): 
        print(f"Adjusting pixel spacing to {spacing}")
        jp2_array = resample_np_2d_jp2(jp2_array, input_dir, spacing)
        if seg_array is not None:
            seg_array = resample_np_2d_jp2(seg_array, input_dir, spacing)

    #slicing and windowing
    if steps.get("slicing", True):
        print(f"Slicing with window size {window_size} and stride {stride}")
        jp2_array, seg_array = split_image_into_tiles_and_write(jp2_array, seg_array, window_size, stride, threshold, percentile)
    else:
        print("Skipping slicing")

    # Save the processed image and mask 

    print('Saving to h5')

    np_to_h5(jp2_array, input_dir, output_dir)

    np_to_h5(seg_array, input_dir, output_mask_dir)

    print(f'Finished processing {name_ds}')
