import os
import pydicom
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
import h5py
import glymur
from PIL import Image


def get_dicom_attribute(dicom_file, attribute):
    """
    Retrieve a specified attribute from a DICOM file object.

    Parameters:
        dicom_file: The DICOM file object from which to retrieve the attribute.
        attribute (str): The name of the attribute to retrieve.

    Returns:
        The value of the specified attribute if it exists in the DICOM file object; 
        otherwise, returns "Unknown".
    """
    return getattr(dicom_file, attribute, "Unknown")

def determine_orientation_from_cosines(orientation):
    """
    Determines the anatomical orientation (Axial, Coronal, Sagittal, Oblique, or Unknown) from a list of direction cosines.
    Args:
        orientation (list or str): A list of six floats representing the direction cosines, or the string "Unknown".
    Returns:
        str: The determined orientation as one of "Axial", "Coronal", "Sagittal", "Oblique", or "Unknown".
    Notes:
        The function expects the orientation to be a list of six elements corresponding to the direction cosines
        of a medical image. If the input is "Unknown", it returns "Unknown" directly.
    """
    if orientation == "Unknown":
        return "Unknown"
    
    # 1.0\-0.0\0.0\-0.0\1.0\0.0 axial
    # 1.0\-0.0\0.0\-0.0\-0.0\-1.0 coronal
    # 0.0\1.0\0.0\-0.0\-0.0\-1.0  sagital
    
    if (abs(orientation[0]) == 1) and (abs(orientation[4]) == 1):
        return "Axial"
    elif (abs(orientation[0])== 1) and (abs(orientation[5]) == 1):
        return "Coronal"
    elif (abs(orientation[1]) == 1) and (abs(orientation[5]) == 1):
        return "Sagittal"
    else:
        return "Oblique"
    
def get_metadata(dicom_file, num_slices):
    """
    Extracts and returns relevant DICOM metadata from a given DICOM file object.
    Parameters:
        dicom_file: pydicom.dataset.FileDataset
            The DICOM file object from which to extract metadata.
        num_slices: int
            The number of slices in the DICOM series, used as a fallback for resolution_z.
    Returns:
        dict:
            A dictionary containing the following metadata fields:
                - pixel_spacing: The physical distance between the centers of each pixel (mm).
                - resolution_x: The number of rows (image height) or "Unknown" if unavailable.
                - resolution_y: The number of columns (image width) or "Unknown" if unavailable.
                - resolution_z: The number of frames (slices) or num_slices if unavailable.
                - slice_thickness: The thickness of each slice (mm).
                - spacing_between_slices: The spacing between adjacent slices (mm).
                - slope: The rescale slope for pixel value transformation.
                - intercept: The rescale intercept for pixel value transformation.
                - RescaleType: The type of rescale operation applied to the pixel data.
    """
    pixel_spacing = get_dicom_attribute(dicom_file, 'PixelSpacing')
    slice_thickness = get_dicom_attribute(dicom_file, 'SliceThickness')
    spacing_between_slices = get_dicom_attribute(dicom_file, 'SpacingBetweenSlices')
    slope = get_dicom_attribute(dicom_file, 'RescaleSlope')
    intercept = get_dicom_attribute(dicom_file, 'RescaleIntercept')
    rescale_type = get_dicom_attribute(dicom_file, 'RescaleType')
    
    
    # Extract resolution in x, y, and z directions if available
    resolution_x = dicom_file.Rows if hasattr(dicom_file, 'Rows') else "Unknown"
    resolution_y = dicom_file.Columns if hasattr(dicom_file, 'Columns') else "Unknown"
    resolution_z = dicom_file.NumberOfFrames if hasattr(dicom_file, 'NumberOfFrames') else num_slices
    
    return {
        "pixel_spacing": pixel_spacing,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "resolution_z": resolution_z,
        "slice_thickness": slice_thickness,
        "spacing_between_slices": spacing_between_slices,
        "slope": slope,
        "intercept": intercept,
        "RescaleType":rescale_type
    }


def collect_meta_data_dicom(source_directory):
    """
    Collects and summarizes metadata from DICOM files in subfolders of the specified source directory.
    This function iterates through each subfolder in the given source directory, reads the first DICOM file
    in each subfolder, extracts relevant metadata, and writes a summary table to a text file named
    'summary_table.txt' in the source directory. The summary includes information such as pixel spacing,
    resolution, slice thickness, spacing between slices, rescale slope/intercept, and rescale type.
    Args:
        source_directory (str): Path to the directory containing subfolders with DICOM files.
    Side Effects:
        - Creates or overwrites 'summary_table.txt' in the source directory with the collected metadata.
        - Prints progress and metadata information to the console.
    Note:
        This function assumes that each subfolder contains at least one DICOM file and that the helper
        functions `get_dicom_attribute` and `get_metadata` are defined elsewhere in the codebase.
    """
    
    # Create summary table
    
    summary_file_path = os.path.join(source_directory, f"summary_table.txt")
    with open(summary_file_path, "w") as summary_file:
            summary_file.write("Folder\tPixel Spacing\tResolution (X)\tResolution (Y)\tResolution (Z)\tLayer Thickness\tLayer Distance\tSlope\tIntercept\tRescale Type\n")
    
    
    print(f"Created summary table at '{summary_file_path}'")
    
    foldernames = [f for f in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, f))]
    
    for foldername in foldernames:
        orientation = "Unknown"
        num_slices = 0
        first_dicom_file = None
        folder_path = os.path.join(source_directory ,foldername)
        
        files_list = os.listdir(folder_path)
        num_slices = len(files_list)
        filepath = os.path.join(folder_path, files_list[0])
        dicom_file = pydicom.dcmread(filepath)
        
        first_dicom_file = dicom_file
        orientation = get_dicom_attribute(dicom_file, 'ImageOrientationPatient')
        # orientation_name = determine_orientation_from_cosines(orientation)
        metadata = get_metadata(first_dicom_file, num_slices)
        metadata["folder"] = foldername
        print(metadata)
            
        with open(summary_file_path, "a") as summary_file:
            summary_file.write(
                f"{metadata['folder']}\t{metadata['pixel_spacing']}\t{metadata['resolution_x']}\t{metadata['resolution_y']}\t{metadata['resolution_z']}\t{metadata['slice_thickness']}\t{metadata['spacing_between_slices']}\t{metadata['slope']}\t{metadata['intercept']}\t{metadata['RescaleType']}\n"
            )
        print(f"Created summary for {foldername}")
        
def extract_number(filename):
    
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0
        
        
def load_dicom_scan_to_np(dicom_path_folder):
    """
    Loads a folder containing DICOM images into a 3D NumPy array.
    Args:
        dicom_path_folder (str or Path): Path to the folder containing DICOM files.
        np.ndarray: 3D NumPy array of stacked DICOM slices with shape (z, x, y), where
            z is the number of slices, and (x, y) are the dimensions of each slice.
    Raises:
        Prints a warning if a file cannot be imported due to an IOError or invalid DICOM format.
    Notes:
        - The function sorts filenames using the `extract_number` key before loading.
        - Only valid DICOM files are loaded; invalid files are skipped with a printed message.
    
    """

    dicom_slices = []

    for root, _, filenames in os.walk(dicom_path_folder):
        filenames.sort(key=extract_number)
        for filename in filenames:
            dcm_path = Path(root, filename)
            try:
                # read in data and convert to array
                dicom = pydicom.dcmread(dcm_path)
                pixel_array = dicom.pixel_array
            except (IOError, pydicom.errors.InvalidDicomError) as e:
                print(f"Can't import {dcm_path.stem}: {e}")
            else:
                # append to list of slices
                dicom_slices.append(pixel_array)


    dicom_slices_np = np.array(dicom_slices)

    print(f'Loaded Dataset {os.path.basename(os.path.normpath(dicom_path_folder))}')

    return dicom_slices_np


def load_jp2_scan_to_np(jp2_path_folder):
    """
    Loads all JP2 image slices from a specified folder into a NumPy array.
    Args:
        jp2_path_folder (str): Path to the folder containing JP2 image files.
    Returns:
        np.ndarray: A NumPy array containing all loaded JP2 image slices, stacked along the first axis.
    Notes:
        - Only files with the ".jp2" extension are loaded.
        - The slices are sorted by filename before loading.
        - Prints the name of the loaded dataset (the folder name).
        - Requires the `glymur` and `numpy` libraries.
    """
    

    jp2_slices = []

    for file in sorted(os.listdir(jp2_path_folder)):
        if file.endswith(".jp2"):
            
            jp2 = glymur.Jp2k(os.path.join(jp2_path_folder,file))
            image_array = np.array(jp2[:])
            jp2_slices.append(image_array)

    jp2_slices = np.array(jp2_slices)

    print(f'Loaded Dataset {os.path.basename(os.path.normpath(jp2_path_folder))}')

    return jp2_slices


def load_tif_scan_to_np(tif_path_folder):
    """
    Loads a sequence of TIFF images from a specified folder and stacks them into a NumPy array.
    Args:
        tif_path_folder (str): Path to the folder containing TIFF (.tif or .tiff) image files.
    Returns:
        np.ndarray: A NumPy array containing the stacked image slices, where each slice corresponds to one TIFF file.
    Prints:
        The name of the loaded dataset (the folder name).
    """
    
    tif_slices = []

    for file in sorted(os.listdir(tif_path_folder)):
        if file.lower().endswith(('.tif', '.tiff')): 
            tif = Image.open(os.path.join(tif_path_folder,file))
            image_array = np.array(tif)
            tif_slices.append(image_array)

    tif_slices = np.array(tif_slices)

    print(f'Loaded Dataset {os.path.basename(os.path.normpath(tif_path_folder))}')

    return tif_slices

def get_meta_data_from_txt(dicom_path_folder):
    """
    Extracts DICOM metadata from a summary text file for a given folder.
    Given the path to a DICOM folder, this function locates a 'summary_table.txt' file in the parent directory,
    reads it as a tab-separated table, and retrieves metadata fields corresponding to the folder.
    Args:
        dicom_path_folder (str): Path to the DICOM folder whose metadata is to be extracted.
    Returns:
        dict: A dictionary containing the following metadata fields:
            - 'pixel_spacing': Pixel spacing value.
            - 'resolution_X': Resolution in the X dimension.
            - 'resolution_Y': Resolution in the Y dimension.
            - 'resolution_Z': Resolution in the Z dimension.
            - 'layer_thickness': Thickness of each layer.
            - 'layer_distance': Distance between layers.
            - 'Slope': Slope value for rescaling.
            - 'Intercept': Intercept value for rescaling.
            - 'RescaleType': Type of rescaling applied.
    Raises:
        FileNotFoundError: If the summary table file does not exist.
        KeyError: If the folder name or required columns are not found in the table.
        IndexError: If no matching row is found for the folder.
    """
    
    folder_meta_data = os.path.dirname(dicom_path_folder)
    path_summary = os.path.join(folder_meta_data, 'summary_table.txt')

    table_pd = pd.read_csv(path_summary, sep='\t')

    folder_name = os.path.basename(dicom_path_folder)


    pixel_spacing = table_pd.loc[table_pd['Folder'] == folder_name, 'Pixel Spacing'].values[0]
    resolution_X = table_pd.loc[table_pd['Folder'] == folder_name, 'Resolution (X)'].values[0]
    resolution_Y = table_pd.loc[table_pd['Folder'] == folder_name, 'Resolution (Y)'].values[0]
    resolution_Z = table_pd.loc[table_pd['Folder'] == folder_name, 'Resolution (Z)'].values[0]
    layer_thickness= table_pd.loc[table_pd['Folder'] == folder_name, 'Layer Thickness'].values[0]
    layer_distance = table_pd.loc[table_pd['Folder'] == folder_name, 'Layer Distance'].values[0]
    slope = table_pd.loc[table_pd['Folder'] == folder_name, 'Slope'].values[0]
    intercept = table_pd.loc[table_pd['Folder'] == folder_name, 'Intercept'].values[0]
    rescaletype = table_pd.loc[table_pd['Folder'] == folder_name, 'Rescale Type'].values[0]
    
    return {'pixel_spacing':pixel_spacing, 'resolution_X':resolution_X, 'resolution_Y':resolution_Y, 'resolution_Z':resolution_Z, 'layer_thickness':layer_thickness, 'layer_distance':layer_distance, 'Slope':slope, 'Intercept':intercept, 'RescaleType':rescaletype}
        

def dicom_2_HU(path_dicom):
    """
    Converts a DICOM scan to Hounsfield Units (HU).
    Parameters:
        path_dicom (str): Path to the DICOM scan directory or file.
    Returns:
        numpy.ndarray: The DICOM scan converted to Hounsfield Units (HU).
    Notes:
        - This function assumes that `load_dicom_scan_to_np` loads the DICOM scan as a NumPy array.
        - The function also assumes that `get_meta_data_from_txt` returns a dictionary containing
          the keys 'Slope' and 'Intercept' required for the HU conversion.
    """
    
    dicom_np = load_dicom_scan_to_np(path_dicom)
    meta_data = get_meta_data_from_txt(path_dicom)
    
    dicom_HU = dicom_np *float(meta_data['Slope']) + float(meta_data['Intercept'])
    
    return dicom_HU


def align_nifti_seg(seg_array):
    """
    Aligns a segmentation mask array to match the orientation of a DICOM numpy array.
    Parameters:
        seg_array (np.ndarray): The segmentation mask as a NumPy array.
    Returns:
        np.ndarray: The aligned segmentation mask.
    Notes:
        The function assumes that the input segmentation mask is in the format [z, x, y].
        It applies necessary transformations to match the orientation of the DICOM array.
    """
    
    # Necessary transformations to match the DICOM orientation
    seg_array = np.moveaxis(seg_array, -1, 0)
    seg_array = np.flip(seg_array, axis=0)
    seg_array = np.swapaxes(seg_array, 1, 2)
    seg_array = np.flip(seg_array, axis=2)
    seg_array = np.rot90(seg_array, k=2, axes=(1, 2))
    
    return seg_array

def apply_segment_dicom_np(dicom_np, path_seg, padding_value):
    """
    Applies a segmentation mask to a DICOM numpy array, setting voxels outside the mask to a specified padding value.
    Parameters:
        dicom_np (np.ndarray): The input DICOM image as a NumPy array.
        path_seg (str): Path to the segmentation file in NIfTI format generated by the TotalSegmentator Tool.

        padding_value (numeric): The value to assign to voxels outside the segmentation mask.
    Returns:
        np.ndarray: The modified DICOM numpy array with voxels outside the segmentation mask set to the padding value.
    Notes:
        The segmentation mask is loaded and transformed to match the orientation of the DICOM array before applying.
    """
    img_seg = nib.load(path_seg)
    seg_array = img_seg.get_fdata()       
    
    #Necessary transformations to match the DICOM orientation

    seg_array =align_nifti_seg(seg_array)
    dicom_np[seg_array == 0] = padding_value
    
    return dicom_np


def resample_np_2d_dicom(dicom_np, path_dicom, new_spacing):
    """
    Resamples a 3D numpy array representing a stack of 2D DICOM slices to a new pixel spacing.
    Parameters
    ----------
    dicom_np : np.ndarray
        A 3D numpy array of shape (num_slices, height, width) representing the DICOM image stack.
    path_dicom : str
        Path to the DICOM metadata text file, used to extract the original pixel spacing.
    new_spacing : tuple or list of float
        The desired pixel spacing in the (row, column) directions, e.g., (new_row_spacing, new_col_spacing).
    Returns
    -------
    resampled_data : np.ndarray
        A 3D numpy array of the resampled DICOM stack with updated spatial resolution.
    Notes
    -----
    - The function assumes that the input DICOM stack has uniform spacing in the x and y directions.
    - The function uses linear interpolation to resample each 2D slice independently.
    - The original pixel spacing is extracted from the metadata file at `path_dicom`.
    """

    meta_data = get_meta_data_from_txt(path_dicom)

    spacing_str = meta_data['pixel_spacing']

    import ast
    old_spacing = tuple(ast.literal_eval(spacing_str))

    # Define the original grid coordinates for x, y (assuming uniform spacing)
    x = np.arange(0, dicom_np.shape[1]) * old_spacing[0]
    y = np.arange(0, dicom_np.shape[2]) * old_spacing[1]

    # Define the new grid coordinates for x, y
    new_x = np.linspace(x.min(), x.max(), int(dicom_np.shape[1] * old_spacing[0] / new_spacing[0]))
    new_y = np.linspace(y.min(), y.max(), int(dicom_np.shape[2] * old_spacing[1] / new_spacing[1]))

    # Create the meshgrid for the new coordinates
    new_xx, new_yy = np.meshgrid(new_x, new_y, indexing='ij')

    # Initialize the interpolator for each z-slice
    resampled_data = np.empty((dicom_np.shape[0], len(new_x), len(new_y)))

    for z in range(dicom_np.shape[0]):
        # Create the interpolator for the current slice
        interpolator = RegularGridInterpolator((x, y), dicom_np[z, :, :])
        
        # Interpolate the slice onto the new grid
        points = np.array([new_xx.ravel(), new_yy.ravel()]).T
        resampled_slice = interpolator(points).reshape(len(new_x), len(new_y))
        
        # Store the interpolated slice in the resampled array
        resampled_data[z, :, :] = resampled_slice

    return resampled_data

def resample_np_2d_jp2(dicom_np, path_jp2, new_spacing):
    """
    Resamples a 3D numpy array (dicom_np) representing a stack of 2D images to a new pixel spacing,
    using the pixel size extracted from the folder name of the JP2 file.
    The function assumes that the folder name (from path_jp2) contains the pixel size in micrometers,
    formatted as "<number>um" (e.g., "0.5um"). The resampling is performed in the x and y dimensions
    for each z-slice using linear interpolation.
    Parameters
    ----------
    dicom_np : np.ndarray
        A 3D numpy array of shape (z, x, y) representing the image stack to be resampled.
    path_jp2 : str
        Path to the JP2 file or its containing folder. The folder name must include the pixel size in micrometers.
    new_spacing : tuple or list of float
        The desired pixel spacing in millimeters for the x and y dimensions, as (new_spacing_x, new_spacing_y).
    Returns
    -------
    resampled_data : np.ndarray
        A 3D numpy array of the resampled image stack with updated x and y dimensions according to new_spacing.
    Raises
    ------
    ValueError
        If the folder name does not contain a valid pixel size in micrometers.
    Notes
    -----
    - The function uses `RegularGridInterpolator` from scipy for interpolation.
    - The z-dimension is not resampled; only x and y are resampled for each slice.
    - The original pixel size is assumed to be uniform and is extracted from the folder name.
    """

    folder_name = os.path.basename(path_jp2)
    
    match = re.match(r"(\d+\.\d+)um", folder_name)
    if match:
        spacing_str = match.group(1)
        print(f"Extracted pixel size: {spacing_str}")
    else:
        raise ValueError("No pixel size found in the folder name. Cannot determine spacing.")

    old_spacing = float(spacing_str) * 10**(-3)
    old_spacing = float(spacing_str) * 10**(-3)

    # Define the original grid coordinates for x, y (assuming uniform spacing)
    x = np.arange(0, dicom_np.shape[1]) * old_spacing
    y = np.arange(0, dicom_np.shape[2]) * old_spacing

    # Define the new grid coordinates for x, y
    new_x = np.linspace(x.min(), x.max(), int(dicom_np.shape[1] * old_spacing / new_spacing[0]))
    new_y = np.linspace(y.min(), y.max(), int(dicom_np.shape[2] * old_spacing / new_spacing[1]))

    # Create the meshgrid for the new coordinates
    new_xx, new_yy = np.meshgrid(new_x, new_y, indexing='ij')

    # Initialize the interpolator for each z-slice
    resampled_data = np.empty((dicom_np.shape[0], len(new_x), len(new_y)))

    for z in range(dicom_np.shape[0]):
        # Create the interpolator for the current slice
        interpolator = RegularGridInterpolator((x, y), dicom_np[z, :, :])
        
        # Interpolate the slice onto the new grid
        points = np.array([new_xx.ravel(), new_yy.ravel()]).T
        resampled_slice = interpolator(points).reshape(len(new_x), len(new_y))
        
        # Store the interpolated slice in the resampled array
        resampled_data[z, :, :] = resampled_slice

    return resampled_data



def split_image_into_tiles_and_write(arr, seg_array, window_size, stride, threshold, percentile):
    """
    Splits a 3D image array and its corresponding segmentation array into smaller tiles, 
    filters the tiles based on a threshold and percentile, and returns the filtered tiles.
    Args:
        arr (np.ndarray): Input image array of shape (n_slices, height, width).
        seg_array (np.ndarray or None): Segmentation array of the same shape as `arr`, or None.
        window_size (int): Size of the square tile (height and width).
        stride (int): Step size for moving the window across the image.
        threshold (numeric): Value to compare against when filtering tiles.
        percentile (float): Maximum allowed fraction of pixels in a tile equal to `threshold` for the tile to be kept.
    Returns:
        tuple:
            - sliced_data (np.ndarray): Array of image tiles that passed the filtering, shape (num_tiles, window_size, window_size).
            - sliced_seg (np.ndarray or None): Array of corresponding segmentation tiles, or None if seg_array was None.
    Notes:
        - The function ensures that the entire image is covered by the tiles, including the edges.
        - Tiles are filtered out if the proportion of pixels equal to `threshold` exceeds `percentile`.
    """
    tile_array = []
    seg_tiles = [] if seg_array is not None else None

    h = arr.shape[1]
    w = arr.shape[2]
    n_slices = arr.shape[0]

    print(h, w)

    # Compute tile positions
    h_steps = list(range(0, h, stride))
    w_steps = list(range(0, w, stride))

    h_steps_cleaned = []
    w_steps_cleaned = []

    for i in range(len(h_steps)):
        if h_steps[i] + window_size <= h:
            h_steps_cleaned.append(h_steps[i])

    for k in range(len(w_steps)):
        if w_steps[k] + window_size <= w:
            w_steps_cleaned.append(w_steps[k])

    # Adjust steps to ensure the last tiles include the entire image
    if h % window_size != 0:
        h_steps_cleaned.append(h - window_size)
    if w % window_size != 0:
        w_steps_cleaned.append(w - window_size)

    for z in range(n_slices):
        for y in h_steps_cleaned:  # Note: y corresponds to rows
            for x in w_steps_cleaned:  # Note: x corresponds to columns
                y_end = y + window_size
                x_end = x + window_size

                tile = arr[z, y:y_end, x:x_end]
                tile_array.append(tile)
                if seg_array is not None:
                    tile_seg = seg_array[z, y:y_end, x:x_end]
                    seg_tiles.append(tile_seg)

    print(f'Shape before filtering {np.array(tile_array).shape}')

    tile_array = np.array(tile_array)
    if seg_array is not None:
        seg_tiles = np.array(seg_tiles)

    sliced_indices = [i for i in range(tile_array.shape[0]) if np.sum(tile_array[i] == threshold) / tile_array[i].size < percentile]

    sliced_data = np.array(tile_array[sliced_indices])

    if seg_array is not None:
        sliced_seg = np.array(seg_tiles[sliced_indices])
    else:
        sliced_seg = None

    print(f'Shape after filtering {sliced_data.shape}')

    return sliced_data, sliced_seg



def np_to_h5(array, path_dicom, path_h5):
    """
    Saves a NumPy array to an HDF5 file using the DICOM file's basename as the dataset name.
    If a dataset with the same name already exists in the HDF5 file, the function skips saving.
    Parameters:
        array (np.ndarray): The NumPy array to be saved.
        path_dicom (str): Path to the DICOM file; its basename is used as the dataset name.
        path_h5 (str): Path to the HDF5 file where the array will be saved.
    Returns:
        None
    Side Effects:
        Writes the NumPy array to the specified HDF5 file. Prints status messages indicating whether the dataset was saved or skipped.
    """
    dataset_name = os.path.basename(path_dicom) 
    
    # Open the HDF5 file in append mode
    with h5py.File(path_h5, 'a') as h5_file:
        # Check if the dataset already exists
        if dataset_name in h5_file:
            print(f"Dataset {dataset_name} already exists in {path_h5}, skipping.")
        else:
            # Save the NumPy array to the HDF5 file
            h5_file.create_dataset(dataset_name, data=array)
            print(f"Saved {dataset_name} to {path_h5}")
            

def contrast_stretch_with_clipping(image, low_cutoff, high_cutoff):
    """
    Applies contrast stretching to an image with intensity clipping.

    This function first clips the pixel values of the input image to the specified
    low and high cutoff values. It then linearly stretches the clipped intensities
    to the full range [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array.
    low_cutoff : float
        Lower intensity cutoff. Pixel values below this will be set to this value.
    high_cutoff : float
        Upper intensity cutoff. Pixel values above this will be set to this value.

    Returns
    -------
    np.ndarray
        The contrast-stretched image with values scaled to [0, 1].
    """
    # Clip the image at the specified cutoffs
    clipped = np.clip(image, low_cutoff, high_cutoff)
    # Stretch the intensities to the full range [0, 1]
    stretched = (clipped - low_cutoff) / (high_cutoff - low_cutoff)
    return stretched



def h5_to_np(path_h5):
    """
    Loads the first dataset from an HDF5 (.h5) file and returns it as a NumPy array.
    Parameters:
        path_h5 (str): Path to the HDF5 file.
    Returns:
        numpy.ndarray: The data from the first dataset in the HDF5 file.
    Raises:
        OSError: If the file cannot be opened.
        KeyError: If the file does not contain any datasets.
    Notes:
        Assumes the HDF5 file contains at least one dataset and loads the first one found.
    """

    with h5py.File(path_h5, 'r') as h5_file:
            
            dataset_name = list(h5_file.keys())[0]  # Get the first (and only) dataset name
            print(f'Loading dataset {dataset_name}')
            array = h5_file[dataset_name][:] 
            
    return array


def split_and_save_h5_separate(data, dataset_name, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Splits a 3D NumPy array into train, validation, and test sets, and saves each to separate HDF5 files.

    Parameters:
        data (np.ndarray): The input 3D NumPy array with shape [z, x, y].
        output_dir (str): The directory to save the output HDF5 files.
        train_ratio (float): Proportion of data to use for training. Default is 0.7.
        val_ratio (float): Proportion of data to use for validation. Default is 0.15.
        test_ratio (float): Proportion of data to use for testing. Default is 0.15.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        None
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The split ratios must sum to 1.0")

    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    # Shuffle the slices along the first axis
    np.random.shuffle(data)

    # Compute split indices
    z = data.shape[0]
    train_end = int(train_ratio * z)
    val_end = train_end + int(val_ratio * z)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Save each split into a separate HDF5 file
    train_file = os.path.join(output_dir, f"{dataset_name}_train_data.h5")
    val_file = os.path.join(output_dir, f"{dataset_name}_val_data.h5")
    test_file = os.path.join(output_dir, f"{dataset_name}_test_data.h5")

    with h5py.File(train_file, "w") as h5f:
        h5f.create_dataset("train", data=train_data)

    with h5py.File(val_file, "w") as h5f:
        h5f.create_dataset("val", data=val_data)

    with h5py.File(test_file, "w") as h5f:
        h5f.create_dataset("test", data=test_data)

    print(f"Train data saved to {train_file}")
    print(f"Validation data saved to {val_file}")
    print(f"Test data saved to {test_file}")