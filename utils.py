import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import re
import os
# Load and transform images to tensors

def plot_batch_with_avg_intensity(batch, fake_batch, og_domain, new_domain, path):
    batch = batch.cpu().detach().numpy()  # Move to CPU for plotting
    fake_batch = fake_batch.cpu().detach().numpy()  # Move to CPU for plotting
    batch_size = batch.shape[0]
    fig, axs = plt.subplots(2, batch_size, figsize=(15, 6))

    for i in range(batch_size):
        real_image = batch[i]  # Remove channel dimension for grayscale
        fake_image = fake_batch[i]  # Remove channel dimension for grayscale

        avg_intensity_real = real_image.mean().item()  # Calculate average intensity of real image
        avg_intensity_fake = fake_image.mean().item()  # Calculate average intensity of fake image

        # Plot real images
        axs[0, i].imshow(real_image, cmap='gray')
        axs[0, i].set_title(f'{og_domain} Real (Avg Intensity: {avg_intensity_real:.4f})')
        axs[0, i].axis('off')

        # Plot generated (fake) images
        axs[1, i].imshow(fake_image, cmap='gray')
        axs[1, i].set_title(f'{new_domain} Fake (Avg Intensity: {avg_intensity_fake:.4f})')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(path)

def save_fake_batches_to_h5(iteration, num_batches, batch_A, batch_B, fake_A, fake_B, id_A, id_B, rec_A, rec_B, mask_A, mask_B, output_path='fake_batches.h5'):
    
    batch_A = batch_A.cpu().detach().numpy().squeeze()  
    batch_B = batch_B.cpu().detach().numpy().squeeze()  
    
    fake_A = fake_A.cpu().detach().numpy().squeeze()  
    fake_B = fake_B.cpu().detach().numpy().squeeze()
    
    id_A = id_A.cpu().detach().numpy().squeeze()  
    id_B = id_B.cpu().detach().numpy().squeeze()
    
    rec_A = rec_A.cpu().detach().numpy().squeeze()  
    rec_B = rec_B.cpu().detach().numpy().squeeze() 
    
    mask_A = mask_A.cpu().detach().numpy().squeeze()
    mask_B = mask_B.cpu().detach().numpy().squeeze()  

    # Create an HDF5 file and save the fake batches
    with h5py.File(output_path, 'a') as h5f:
        
        if 'fake_A' not in h5f.keys():
            print('Creating datasets')
            h5f.create_dataset('batch_A', shape=(num_batches, batch_A.shape[0], batch_A.shape[1]), dtype='float32')
            h5f.create_dataset('batch_B', shape=(num_batches, batch_B.shape[0], batch_B.shape[1]), dtype='float32')
            h5f.create_dataset('fake_A', shape=(num_batches, fake_A.shape[0], fake_A.shape[1]), dtype='float32')
            h5f.create_dataset('fake_B', shape=(num_batches, fake_B.shape[0], fake_B.shape[1]), dtype='float32')
            h5f.create_dataset('id_A', shape=(num_batches, id_A.shape[0], id_A.shape[1]), dtype='float32')
            h5f.create_dataset('id_B', shape=(num_batches, id_B.shape[0], id_B.shape[1]), dtype='float32')
            h5f.create_dataset('rec_A', shape=(num_batches, rec_A.shape[0], rec_A.shape[1]), dtype='float32')
            h5f.create_dataset('rec_B', shape=(num_batches, rec_B.shape[0], rec_B.shape[1]), dtype='float32')
            h5f.create_dataset('mask_A', shape=(num_batches, mask_A.shape[0], mask_A.shape[1]), dtype='float32')
            h5f.create_dataset('mask_B', shape=(num_batches, mask_B.shape[0], mask_B.shape[1]), dtype='float32')
            
        h5f['batch_A'][iteration, : ,:] = batch_A
        h5f['batch_B'][iteration, : ,:] = batch_B
        h5f['fake_A'][iteration, : ,:] = fake_A
        h5f['fake_B'][iteration, : ,:] = fake_B
        h5f['id_A'][iteration, : ,:] = id_A
        h5f['id_B'][iteration, : ,:] = id_B
        h5f['rec_A'][iteration, : ,:] = rec_A
        h5f['rec_B'][iteration, : ,:] = rec_B
        h5f['mask_A'][iteration, : ,:] = mask_A
        h5f['mask_B'][iteration, : ,:] = mask_B
            


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def preprocess_images(images, img_size=299):
    """
    Preprocess grayscale images to tensors suitable for FID computation.
    Assumes images are in the format (N, H, W) and range [0, 1].
    """
    transform = Compose([
        Resize((img_size, img_size)),  # Resize to Inception required size
    ])
    images = (images * 255).clamp(0, 255).to(torch.uint8)  # Scale and convert to uint8
    #images = images.unsqueeze(1)
    images = images.repeat(1, 3, 1, 1)  # Replicate channel dimension to RGB: (N, 3, H, W)
    preprocessed = transform(images) # Apply transform per image
    return preprocessed

def compute_fid_with_batches(real_images, generated_images, batch_size=32, device='cuda'):
    """
    Computes FID using torchmetrics for grayscale images with batching.
    :param real_images: numpy array of shape (N, H, W), range [0, 1]
    :param generated_images: numpy array of shape (N, H, W), range [0, 1]
    :param batch_size: Number of images to process in each batch.
    :param device: Device to perform computations ('cuda' or 'cpu')
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Preprocess images
    real_images = preprocess_images(real_images)
    generated_images = preprocess_images(generated_images)

    print('shape real images', real_images.shape)
    print('shape generated images', generated_images.shape)

    # Create DataLoaders for batching
    real_loader = DataLoader(TensorDataset(real_images), batch_size=batch_size, shuffle=False)
    generated_loader = DataLoader(TensorDataset(generated_images), batch_size=batch_size, shuffle=False)

    print("Updating FID with real images")
    # Update FID metric with real images
    for batch in tqdm(real_loader):
        fid.update(batch[0].to(device), real=True)

    print("Updating FID with fake images")
    # Update FID metric with generated images
    for batch in tqdm(generated_loader):
        fid.update(batch[0].to(device), real=False)

    # Compute FID score
    fid_score = fid.compute()
    return fid_score.item()


def find_last_epoch(path):
    
        numbers = [
        int(match.group(1))
        for filename in os.listdir(path)
        if (match := re.search(r'(\d+)(?=\.\w+$)', filename))
        ]
        return max(numbers)
