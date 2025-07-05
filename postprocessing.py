import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import load_config_from_json
from model import Generator, Discriminator
from torchvision import transforms
from data import H5Dataset, CombinedDataset
import argparse

def parse_args():
    """
    Parse command-line arguments for the postprocessing script.

    Returns:
        argparse.Namespace: Parsed arguments including epoch, output path, config, and mask paths.
    """
    parser = argparse.ArgumentParser(description="Postprocessing script for style cycle model.")
    parser.add_argument('--epoch', type=int, default=199, help='Epoch to load checkpoint from')
    parser.add_argument('--output_path', type=str, default='./output.h5', help='Output HDF5 file path')
    parser.add_argument('--config', type=str, default='./configs_json/config_training.json', help='Path to config JSON')
    parser.add_argument('--mask_in_vivo_test', nargs='+', default=["path/to/clinical/masks.h5"], help='List of masks for the clinical CT test dataset paths')
    parser.add_argument('--mask_HiP_test', nargs='+', default=["path/to/HiP-CT/masks.h5"], help='List of masks for the HiP-CT test dataset paths')
    return parser.parse_args()

def main():
    """
    Main function for postprocessing the style cycle model outputs.

    Loads model checkpoints, processes test datasets, applies generators, masks outputs,
    and saves results to an HDF5 file.
    """
    args = parse_args()

    # Extract arguments
    epoch_to_load = args.epoch
    output_path = args.output_path
    path_config = args.config
    path_mask_in_vivo_val = args.mask_in_vivo_val
    path_mask_HiP_val = args.mask_HiP_val

    # Load configuration from JSON
    config = load_config_from_json(path_config)
    paths_A_test = config.paths_A_test
    paths_B_test = config.paths_B_test

    # Define transforms
    transform_comb = None
    transform = transforms.Compose([transforms.ToTensor()])

    # Prepare datasets and dataloaders
    A_test = H5Dataset(paths_A_test, transform=transform)
    B_test = H5Dataset(paths_B_test, transform=transform)
    A_test_mask = H5Dataset(path_mask_HiP_val, transform=transform)
    B_test_mask = H5Dataset(path_mask_in_vivo_val, transform=transform)
    
    print(len(A_test), len(A_test_mask))

    # Combine datasets with masks
    A_test_combined = CombinedDataset(A_test, A_test_mask, transform=transform_comb)
    B_test_combined = CombinedDataset(B_test, B_test_mask, transform=transform_comb)

    # DataLoaders for test datasets
    A_test_dl = DataLoader(A_test_combined, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
    B_test_dl = DataLoader(B_test_combined, batch_size=1, shuffle=False, num_workers=16, pin_memory=False)

    # Create iterators for dataloaders
    A_test_iter = iter(A_test_dl)
    B_test_iter = iter(B_test_dl)

    # Initialize models
    netG_A2B = Generator(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)
    netG_B2A = Generator(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)
    netD_A = Discriminator(input_shape=(1, config.image_size, config.image_size)).to(config.device)
    netD_B = Discriminator(input_shape=(1, config.image_size, config.image_size)).to(config.device)

    # Set models to evaluation mode
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    # Load model checkpoints
    checkpoint = torch.load(os.path.join(config.path_checkpoints, f'checkpoint_epoch_{epoch_to_load}.pth'))
    netG_A2B.load_state_dict(checkpoint['model_G_A2B_state_dict'])
    netG_B2A.load_state_dict(checkpoint['model_G_B2A_state_dict'])
    netD_A.load_state_dict(checkpoint['model_D_A_state_dict'])
    netD_B.load_state_dict(checkpoint['model_D_B_state_dict'])

    # Open HDF5 file for writing results
    with h5py.File(output_path, "a") as h5f:
        iterations = np.max((len(A_test_dl), len(B_test_dl)))
        for i in tqdm(range(iterations)):
            # Get next batch from A_test and B_test, restart iterator if needed
            try:
                batch_A, mask_A = next(A_test_iter)
            except StopIteration:
                A_test_iter = iter(A_test_dl)
                batch_A, mask_A = next(A_test_iter)
            try:
                batch_B, mask_B = next(B_test_iter)
            except StopIteration:
                B_test_iter = iter(B_test_dl)
                batch_B, mask_B = next(B_test_iter)

            # Move data to device
            batch_A = batch_A.to(config.device)
            batch_B = batch_B.to(config.device)
            mask_A = mask_A.to(config.device)
            mask_B = mask_B.to(config.device)

            # Forward pass through generators
            same_A = netG_A2B(batch_A)
            same_B = netG_B2A(batch_B)
            fake_B = netG_A2B(batch_A)
            fake_A = netG_B2A(batch_B)
            recovered_A = netG_B2A(fake_B)
            recovered_B = netG_A2B(fake_A)

            # Apply masks to generated images
            temp_fake_B = fake_B.detach().cpu().numpy()
            temp_mask_A = mask_A.detach().cpu().numpy()
            temp_fake_B[temp_mask_A == 0] = 0

            temp_fake_A = fake_A.detach().cpu().numpy()
            temp_mask_B = mask_B.detach().cpu().numpy()
            temp_fake_A[temp_mask_B == 0] = 0

            # Create datasets in HDF5 file on first iteration
            if i == 0:
                print('created datasets')
                h5f.create_dataset("batch_A", (0,) + batch_A.shape[2:], maxshape=(None,) + batch_A.shape[2:], dtype="float32")
                h5f.create_dataset("batch_B", (0,) + batch_B.shape[2:], maxshape=(None,) + batch_B.shape[2:], dtype="float32")
                h5f.create_dataset("fake_A", (0,) + fake_A.shape[2:], maxshape=(None,) + fake_A.shape[2:], dtype="float32")
                h5f.create_dataset("fake_B", (0,) + fake_B.shape[2:], maxshape=(None,) + fake_B.shape[2:], dtype="float32")
                h5f.create_dataset("same_A", (0,) + same_A.shape[2:], maxshape=(None,) + same_A.shape[2:], dtype="float32")
                h5f.create_dataset("same_B", (0,) + same_B.shape[2:], maxshape=(None,) + same_B.shape[2:], dtype="float32")
                h5f.create_dataset("recovered_A", (0,) + recovered_A.shape[2:], maxshape=(None,) + recovered_A.shape[2:], dtype="float32")
                h5f.create_dataset("recovered_B", (0,) + recovered_B.shape[2:], maxshape=(None,) + recovered_B.shape[2:], dtype="float32")
                h5f.create_dataset("masked_fake_A", (0,) + fake_A.shape[2:], maxshape=(None,) + fake_A.shape[2:], dtype="float32")
                h5f.create_dataset("masked_fake_B", (0,) + fake_B.shape[2:], maxshape=(None,) + fake_B.shape[2:], dtype="float32")

            # Append new data to datasets
            for dataset_name, data in zip(
                ["batch_A", "batch_B", "fake_A", "fake_B", "masked_fake_A", "masked_fake_B", "same_A", "same_B", "recovered_A", "recovered_B"],
                [batch_A.cpu().numpy(), batch_B.cpu().numpy(), fake_A.detach().cpu().numpy(), fake_B.detach().cpu().numpy(), temp_fake_A, temp_fake_B, same_A.detach().cpu().numpy(), same_B.detach().cpu().numpy(), recovered_A.detach().cpu().numpy(), recovered_B.detach().cpu().numpy()]
            ):
                h5f[dataset_name].resize((h5f[dataset_name].shape[0] + data.shape[0]), axis=0)
                h5f[dataset_name][-data.shape[0]:] = data

if __name__ == "__main__":
    main()
