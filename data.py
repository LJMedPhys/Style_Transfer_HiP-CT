import h5py
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, h5_paths, transform=None):
        """
        Args:
            h5_paths (list of str): List of paths to H5 files.
            keys (list of str): List of keys in each H5 file containing the 2D slices.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.h5_paths = h5_paths
        self.keys = []
        for h5_path in self.h5_paths:
                with h5py.File(h5_path, 'r') as h5_file:
                    self.keys.append(list(h5_file.keys()))
        self.transform = transform
        self.slice_references = []
        
        # Collect references to all slices
        self._collect_slice_references()
    
    def _collect_slice_references(self):
        for k, h5_path in enumerate(self.h5_paths):
            with h5py.File(h5_path, 'r') as h5_file:
                for key in self.keys[k]:
                    # Assume each key in the H5 file is a dataset containing 2D slices
                    data = h5_file[key]
                    for i in range(data.shape[0]):
                        # Store a reference to the slice: (file_path, key, slice_index)
                        self.slice_references.append((h5_path, key, i))
    
    def __len__(self):
        return len(self.slice_references)
    
    def __getitem__(self, idx):
        h5_path, key, slice_index = self.slice_references[idx]
        
        # Load the specific slice on-the-fly
        with h5py.File(h5_path, 'r') as h5_file:
            slice_ = h5_file[key][slice_index, :, :]
        
        if self.transform:
            slice_ = self.transform(slice_)
        
        return slice_.type(torch.float32)
    

def split_dataset(dataset, test_split=0.2):
    """
    Split the dataset into training and testing sets.
    
    Args:
        dataset (H5Dataset): The dataset to split.
        test_split (float): Fraction of the dataset to use as the test set.
        
    Returns:
        train_dataset (H5Dataset): Training set.
        test_dataset (H5Dataset): Testing set.
    """
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
    
    
    
transform = transforms.Compose([
    transforms.ToTensor(),
])

class CombinedDataset(Dataset):
    def __init__(self, input_dataset, label_dataset, transform=None):
        """
        Combines an input dataset and a label dataset.
        Args:
            input_dataset: Dataset containing the input images.
            label_dataset: Dataset containing the corresponding labels (masks).
            transform: Optional transforms to apply to both inputs and labels.
        """
        assert len(input_dataset) == len(label_dataset), "Input and label datasets must have the same length!"
        self.input_dataset = input_dataset
        self.label_dataset = label_dataset
        self.transform = transform

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        input_image = self.input_dataset[idx]
        label_mask = self.label_dataset[idx]

        combined = torch.stack((input_image, label_mask), axis=0)

        # Apply transforms (if any) to both input and label
        if self.transform:
            combined = self.transform(combined)

        input_image = combined[0,:,:]
        label_mask = combined[1,:,:]


        return input_image, label_mask