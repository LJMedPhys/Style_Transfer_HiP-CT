
from tqdm import tqdm
import os
import wandb
from config import load_config_from_json
from model import Generator, Discriminator, GenUNet
import torch.optim as optim
from losses import CycleGANLoss
from torchvision.transforms import v2
from data import H5Dataset
from torch.utils.data import DataLoader
import torch
import itertools
from random import randint
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import AverageMeter, preprocess_images, find_last_epoch
from torchmetrics.regression import MeanAbsoluteError as mae
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.regression import MeanSquaredError as mse
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import MeanMetric
import argparse

# Parse command-line arguments for configuration file, checkpoint loading, and wandb logging
parser = argparse.ArgumentParser(description="Train CycleGAN with configuration file and optional checkpoint loading.")
parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file.")
parser.add_argument('--load_checkpoint', action='store_true', help="Flag to load the last checkpoint if available.")
parser.add_argument('--log_wandb', action='store_true', help="Flag to log activity in wandb.")
args = parser.parse_args()

# Load the configuration file
path_config = args.config
config = load_config_from_json(path_config)

# Flags for loading checkpoint and logging to wandb
load_checkpoint = args.load_checkpoint
log_wandb = args.log_wandb

# Initialize wandb logging if the flag is set
if log_wandb:
    run = wandb.init(
        project="style transfer cycle gan",  # Set the project name for wandb
        name=config.name_run,  # Name of the run
        config={  # Log hyperparameters and metadata
            "learning_rate_D": config.learning_rate_D,
            "learning_rate_G": config.learning_rate_G,
            "epochs": config.epochs,
            "batch_size": config.batch_size
        },
    )

# Initialize the generator networks based on the configuration
if config.skip == 'True':
    # Use UNet-based generator if skip connections are enabled
    netG_A2B = GenUNet(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)
    netG_B2A = GenUNet(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)
else:
    # Use standard generator
    netG_A2B = Generator(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)
    netG_B2A = Generator(input_nc=1, n_residual_blocks=config.n_residual_blocks).to(config.device)

# Initialize the discriminator networks
netD_A = Discriminator(input_shape=(1, config.image_size, config.image_size)).to(config.device)
netD_B = Discriminator(input_shape=(1, config.image_size, config.image_size)).to(config.device)

# Define optimizers for the generator and discriminator networks
optimizer_G = optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), 
    lr=config.learning_rate_G, 
    betas=(config.beta1, config.beta2)
)
optimizer_D = optim.Adam(
    itertools.chain(netD_A.parameters(), netD_B.parameters()),
    lr=config.learning_rate_D, 
    betas=(config.beta1, config.beta2)
)

# Load the last checkpoint if the flag is set
if load_checkpoint:
    # Find the last saved epoch
    epoch_to_load = find_last_epoch(config.path_checkpoints)
    starting_epoch = epoch_to_load

    # Load the checkpoint file
    checkpoint = torch.load(os.path.join(config.path_checkpoints, f'checkpoint_epoch_{epoch_to_load}.pth'))

    # Restore the state of the models and optimizers
    netG_A2B.load_state_dict(checkpoint['model_G_A2B_state_dict'])
    netG_B2A.load_state_dict(checkpoint['model_G_B2A_state_dict'])
    netD_A.load_state_dict(checkpoint['model_D_A_state_dict'])
    netD_B.load_state_dict(checkpoint['model_D_B_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    print(f'Loaded Training for epoch {starting_epoch}')

# Define the loss function for CycleGAN
loss_fn = CycleGANLoss(config.device)


# Define the transformation pipeline for the datasets
transform = v2.Compose([
    v2.ToTensor(),  # Convert images to tensors
    v2.RandomCrop(size=(64, 64)),  # Randomly crop images to 64x64
    v2.RandomHorizontalFlip(),  # Randomly flip images horizontally
    v2.RandomVerticalFlip(),  # Randomly flip images vertically
])

# Create training and validation datasets for domain A
A_train = H5Dataset(config.paths_A_train, transform=transform)
A_val = H5Dataset(config.paths_A_val, transform=transform)

# Create data loaders for domain A
dataloader_A_train = DataLoader(A_train, batch_size=config.batch_size, shuffle=True, num_workers=18, pin_memory=False)
dataloader_A_val = DataLoader(A_val, batch_size=5, shuffle=False, num_workers=8, pin_memory=False)

# Create training and validation datasets for domain B
B_train = H5Dataset(config.paths_B_train, transform=transform)
B_val = H5Dataset(config.paths_B_val, transform=transform)

# Create data loaders for domain B
dataloader_B_train = DataLoader(B_train, batch_size=config.batch_size, shuffle=True, num_workers=18, pin_memory=False)
dataloader_B_val = DataLoader(B_val, batch_size=5, shuffle=False, num_workers=8, pin_memory=False)

# Print the number of samples in the training datasets
print(f'Number of samples in training dataset A: {len(dataloader_A_train) * config.batch_size}')
print(f'Number of samples in training dataset B: {len(dataloader_B_train) * config.batch_size}')

# Determine the number of batches for training
n_batch = min(len(dataloader_A_train), len(dataloader_B_train))
print(f'Number of batches for training: {n_batch} with batch size {config.batch_size}')

# Print the number of samples in the validation datasets
print(f'Number of samples in validation dataset A: {len(dataloader_A_val) * 5}')
print(f'Number of samples in validation dataset B: {len(dataloader_B_val) * 5}')

# Determine the number of batches for validation
n_batch_val = min(len(dataloader_A_val), len(dataloader_B_val))
print(f'Number of batches for validation: {n_batch_val} with batch size {1}')

# Initialize trackers for training losses
loss_id_A_tracker = MeanMetric()  # Tracker for identity loss in domain A
loss_id_B_tracker = MeanMetric()  # Tracker for identity loss in domain B
loss_A2B_tracker = MeanMetric()  # Tracker for GAN loss from A to B
loss_B2A_tracker = MeanMetric()  # Tracker for GAN loss from B to A
loss_cycle_A_tracker = MeanMetric()  # Tracker for cycle consistency loss in domain A
loss_cycle_B_tracker = MeanMetric()  # Tracker for cycle consistency loss in domain B
loss_D_A_tracker = MeanMetric()  # Tracker for discriminator loss in domain A
loss_D_B_tracker = MeanMetric()  # Tracker for discriminator loss in domain B
loss_G_tracker = MeanMetric()  # Tracker for generator loss
loss_D_tracker = MeanMetric()  # Tracker for total discriminator loss

# Initialize trackers for validation losses
loss_id_A_tracker_val = MeanMetric()  # Tracker for identity loss in domain A (validation)
loss_id_B_tracker_val = MeanMetric()  # Tracker for identity loss in domain B (validation)
loss_A2B_tracker_val = MeanMetric()  # Tracker for GAN loss from A to B (validation)
loss_B2A_tracker_val = MeanMetric()  # Tracker for GAN loss from B to A (validation)
loss_cycle_A_tracker_val = MeanMetric()  # Tracker for cycle consistency loss in domain A (validation)
loss_cycle_B_tracker_val = MeanMetric()  # Tracker for cycle consistency loss in domain B (validation)
loss_D_A_tracker_val = MeanMetric()  # Tracker for discriminator loss in domain A (validation)
loss_D_B_tracker_val = MeanMetric()  # Tracker for discriminator loss in domain B (validation)
loss_G_tracker_val = MeanMetric()  # Tracker for generator loss (validation)
loss_D_tracker_val = MeanMetric()  # Tracker for total discriminator loss (validation)

# Training loop for the specified number of epochs
for epoch in range(starting_epoch, config.epochs):
    # Initialize data loaders for the current epoch
    data_loader_A_iter = iter(dataloader_A_train)
    data_loader_B_iter = iter(dataloader_B_train)

    # Set all networks to training mode
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # Reset training loss trackers
    loss_id_A_tracker.reset()
    loss_id_B_tracker.reset()
    loss_A2B_tracker.reset()
    loss_B2A_tracker.reset()
    loss_cycle_A_tracker.reset()
    loss_cycle_B_tracker.reset()
    loss_D_A_tracker.reset()
    loss_D_B_tracker.reset()
    loss_G_tracker.reset()
    loss_D_tracker.reset()

    # Reset validation loss trackers
    loss_id_A_tracker_val.reset()
    loss_id_B_tracker_val.reset()
    loss_A2B_tracker_val.reset()
    loss_B2A_tracker_val.reset()
    loss_cycle_A_tracker_val.reset()
    loss_cycle_B_tracker_val.reset()
    loss_D_A_tracker_val.reset()
    loss_D_B_tracker_val.reset()
    loss_G_tracker_val.reset()
    loss_D_tracker_val.reset()

    # Iterate through batches for training
    for i in tqdm(range(n_batch)):
        # Load real images from domains A and B
        real_A = next(data_loader_A_iter).to(config.device)
        real_B = next(data_loader_B_iter).to(config.device)

        # Train Generators A2B and B2A
        # Freeze discriminator weights during generator training
        for param in netD_A.parameters():
            param.requires_grad = False
        for param in netD_B.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()  # Zero out gradients for generator optimizer

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = loss_fn(same_B, real_B, 'identity') * config.lambda_A * config.lambda_identity
        same_A = netG_B2A(real_A)
        loss_identity_A = loss_fn(same_A, real_A, 'identity') * config.lambda_B * config.lambda_identity

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = loss_fn(pred_fake, torch.ones_like(pred_fake), 'adversarial')

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = loss_fn(pred_fake, torch.ones_like(pred_fake), 'adversarial')

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_A = loss_fn(recovered_A, real_A, 'cycle') * config.lambda_A

        recovered_B = netG_A2B(fake_A)
        loss_cycle_B = loss_fn(recovered_B, real_B, 'cycle') * config.lambda_B

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B
        
        loss_id_A_tracker.update(loss_identity_A.item())
        loss_id_B_tracker.update(loss_identity_B.item())

        loss_A2B_tracker.update(loss_GAN_A2B.item())
        loss_B2A_tracker.update(loss_GAN_B2A.item())

        loss_cycle_A_tracker.update(loss_cycle_A.item())
        loss_cycle_B_tracker.update(loss_cycle_A.item())
        
        loss_G.backward()
        optimizer_G.step()
        

        for param in netD_A.parameters():
            param.requires_grad = True
        for param in netD_B.parameters():
            param.requires_grad = True

        # Discriminator A
        
        pred_real = netD_A(real_A)
        loss_D_real = loss_fn(pred_real, torch.ones_like(pred_real), 'adversarial')

        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake), 'adversarial')

        loss_D_A = (loss_D_real + loss_D_fake)
        

        # Discriminator B
        
        pred_real = netD_B(real_B)
        loss_D_real = loss_fn(pred_real, torch.ones_like(pred_real), 'adversarial')

        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake), 'adversarial')

        loss_D_B = (loss_D_real + loss_D_fake) 
        
        # Total discriminators
        
        loss_D = loss_D_A + loss_D_B
        
        loss_D_A_tracker.update(loss_D_A.item())
        loss_D_B_tracker.update(loss_D_B.item())

        loss_G_tracker.update(loss_G.item())
        loss_D_tracker.update(loss_D.item())
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        # After one epoch, process the validation dataset
        print('Processing validation dataset...')

        # Initialize metrics for validation
        psnr_A_rec_meter = AverageMeter()
        psnr_B_rec_meter = AverageMeter()

        psnr_A_id_meter = AverageMeter()
        psnr_B_id_meter = AverageMeter()

        mse_A_rec_meter = AverageMeter()
        mse_B_rec_meter = AverageMeter()

        mse_A_id_meter = AverageMeter()
        mse_B_id_meter = AverageMeter()

        mae_A_rec_meter = AverageMeter()
        mae_B_rec_meter = AverageMeter()

        mae_A_id_meter = AverageMeter()
        mae_B_id_meter = AverageMeter()

        ssim_A_meter = AverageMeter()
        ssim_B_meter = AverageMeter()

        # Set all networks to evaluation mode
        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()

        # Create iterators for validation data loaders
        data_loader_A_val_iter = iter(dataloader_A_val)
        data_loader_B_val_iter = iter(dataloader_B_val)

        # Randomly select a batch for logging
        random_batch = randint(0, n_batch_val - 1)
        
        with torch.no_grad():
            # Define ranges for SSIM and PSNR calculations
            range_A = 1.0
            range_B = 1.0

            # Initialize metrics for SSIM, PSNR, MSE, MAE, and FID
            ssim_A = ssim(data_range=range_A).to(config.device)
            ssim_B = ssim(data_range=range_B).to(config.device)
            psnr_A = psnr(data_range=range_A).to(config.device)
            psnr_B = psnr(data_range=range_B).to(config.device)
            mse_ = mse().to(config.device)
            mae_ = mae().to(config.device)

            fid_A = FrechetInceptionDistance(feature=2048).to(config.device)
            fid_B = FrechetInceptionDistance(feature=2048).to(config.device)

            # Iterate through validation batches
            for i in tqdm(range(n_batch_val)):
                # Load validation data for domains A and B
                A_val = next(data_loader_A_val_iter).to(config.device)
                B_val = next(data_loader_B_val_iter).to(config.device)

                # Generate identity, fake, and recovered images
                same_B_val = netG_A2B(B_val)
                same_A_val = netG_B2A(A_val)

                fake_B_val = netG_A2B(A_val)
                fake_A_val = netG_B2A(B_val)

                recovered_A_val = netG_B2A(fake_B_val)
                recovered_B_val = netG_A2B(fake_A_val)
                
                # Compute identity loss
                loss_identity_B_val = loss_fn(same_B_val, B_val, 'identity') * config.lambda_A * config.lambda_identity
                loss_identity_A_val = loss_fn(same_A_val, A_val, 'identity') * config.lambda_B * config.lambda_identity

                # Compute GAN loss
                pred_fake = netD_B(fake_B_val)
                loss_GAN_A2B_val = loss_fn(pred_fake, torch.ones_like(pred_fake), 'adversarial')
                
                pred_fake = netD_A(fake_A_val)
                loss_GAN_B2A_val = loss_fn(pred_fake, torch.ones_like(pred_fake), 'adversarial')
                
                # Compute cycle consistency loss
                loss_cycle_A_val = loss_fn(recovered_A_val, A_val, 'cycle') * config.lambda_A
                loss_cycle_B_val = loss_fn(recovered_B_val, B_val, 'cycle') * config.lambda_B
                
                # Compute total generator loss
                loss_G_val = loss_identity_A_val + loss_identity_B_val + loss_GAN_A2B_val + loss_GAN_B2A_val + loss_cycle_A_val + loss_cycle_B_val
                
                # Compute discriminator loss for domain A
                pred_real = netD_A(A_val)
                loss_D_real_val = loss_fn(pred_real, torch.ones_like(pred_real), 'adversarial')

                pred_fake = netD_A(fake_A_val)
                loss_D_fake_val = loss_fn(pred_fake, torch.zeros_like(pred_fake), 'adversarial')

                loss_D_A_val = (loss_D_real_val + loss_D_fake_val)

                # Compute discriminator loss for domain B
                pred_real = netD_B(B_val)
                loss_D_real_val = loss_fn(pred_real, torch.ones_like(pred_real), 'adversarial')

                pred_fake = netD_B(fake_B_val)
                loss_D_fake_val = loss_fn(pred_fake, torch.zeros_like(pred_fake), 'adversarial')

                loss_D_B_val = (loss_D_real_val + loss_D_fake_val) 
                
                # Compute total discriminator loss
                loss_D_val = loss_D_A_val + loss_D_B_val
                
                # Update validation loss trackers
                loss_id_A_tracker_val.update(loss_identity_A_val.item())
                loss_id_B_tracker_val.update(loss_identity_B_val.item())

                loss_A2B_tracker_val.update(loss_GAN_A2B_val.item())
                loss_B2A_tracker_val.update(loss_GAN_B2A_val.item())

                loss_cycle_A_tracker_val.update(loss_cycle_A_val.item())
                loss_cycle_B_tracker_val.update(loss_cycle_A_val.item())

                loss_D_A_tracker_val.update(loss_D_A_val.item())
                loss_D_B_tracker_val.update(loss_D_B_val.item())

                loss_G_tracker_val.update(loss_G_val.item())
                loss_D_tracker_val.update(loss_D_val.item())
                
                # Preprocess images for FID calculation
                A_val_pre = preprocess_images(A_val)
                B_val_pre = preprocess_images(B_val)

                fake_A_val_pre = preprocess_images(fake_A_val)
                fake_B_val_pre = preprocess_images(fake_B_val)

                # Update FID metrics
                fid_A.update(A_val_pre, real=True)
                fid_A.update(fake_A_val_pre, real=False)

                fid_B.update(B_val_pre, real=True)
                fid_B.update(fake_B_val_pre, real=False)

                # Calculate and update SSIM, MSE, MAE, and PSNR metrics
                psnr_A_rec_meter.update(psnr_A(A_val, recovered_A_val), A_val.size(0))
                psnr_B_rec_meter.update(psnr_B(B_val, recovered_B_val), B_val.size(0))

                psnr_A_id_meter.update(psnr_A(A_val, same_A_val), A_val.size(0))
                psnr_B_id_meter.update(psnr_B(B_val, same_B_val), B_val.size(0))

                mse_A_rec_meter.update(mse_(A_val, recovered_A_val), A_val.size(0))
                mse_B_rec_meter.update(mse_(B_val, recovered_B_val), B_val.size(0))

                mse_A_id_meter.update(mse_(A_val, same_A_val), A_val.size(0))
                mse_B_id_meter.update(mse_(B_val, same_B_val), B_val.size(0))

                mae_A_rec_meter.update(mae_(A_val, recovered_A_val), A_val.size(0))
                mae_B_rec_meter.update(mae_(B_val, recovered_B_val), B_val.size(0))

                mae_A_id_meter.update(mae_(A_val, same_A_val), A_val.size(0))
                mae_B_id_meter.update(mae_(B_val, same_B_val), B_val.size(0))

                ssim_A_meter.update(ssim_A(A_val, fake_B_val), A_val.size(0))
                ssim_B_meter.update(ssim_B(B_val, fake_A_val), B_val.size(0))
            
            if log_wandb:

                if i == random_batch:

                    A_val_wandb = wandb.Image(A_val, caption="Original Domain A")
                    B_gen_wandb = wandb.Image(fake_B_val, caption="Translated from A to B")
                    
                    B_val_wandb = wandb.Image(B_val, caption="Original Domain B")
                    A_gen_wandb = wandb.Image(fake_A_val, caption="Translated from B to A")

                    same_B_val_wandb = wandb.Image(same_B_val, caption="Identidity B in A2B")
                    same_A_val_wandb = wandb.Image(same_A_val, caption="Identidity A in B2A")

                    recovered_B_val_wandb = wandb.Image(recovered_B_val, caption="Recovered Image from Original B")
                    recovered_A_val_wandb = wandb.Image(recovered_A_val, caption="Recoverde Image from Original A")

    if log_wandb:
        wandb.log({"Original Domain H":A_val_wandb, "Translated from H to C":B_gen_wandb,
                    "Original Domain C":B_val_wandb, "Translated from C to H":A_gen_wandb,
                    "Original Domain H":A_val_wandb, "Identity H in C2H": same_A_val_wandb,
                    "Original Domain C":B_val_wandb, "Identity C in H2C": same_B_val_wandb,
                    "Original Domain H":A_val_wandb, "Recovered Image from Original H": recovered_A_val_wandb,
                    "Original Domain C":B_val_wandb, "Recovered Image from Original C": recovered_B_val_wandb,
                    "Loss GAN H2C":loss_A2B_tracker.compute(),
                    "Loss GAN C2H":loss_B2A_tracker.compute(),
                    "Loss Cycle H":loss_cycle_A_tracker.compute(),
                    "Loss Cycle C":loss_cycle_B_tracker.compute(),
                    "Loss Identity H":loss_id_A_tracker.compute(),
                    "Loss Identity C":loss_id_B_tracker.compute(),
                    "Loss Generator":loss_G_tracker.compute(),
                    "Loss D_H":loss_D_A_tracker.compute(),
                    "Loss D_C":loss_D_B_tracker.compute(),
                    "Loss D":loss_D_tracker.compute(),
                    
                    "Loss GAN H2C val":loss_A2B_tracker_val.compute(),
                    "Loss GAN C2H val":loss_B2A_tracker_val.compute(),
                    "Loss Cycle H val":loss_cycle_A_tracker_val.compute(),
                    "Loss Cycle C val":loss_cycle_B_tracker_val.compute(),
                    "Loss Identity H val":loss_id_A_tracker_val.compute(),
                    "Loss Identity C val":loss_id_B_tracker_val.compute(),
                    "Loss Generator val":loss_G_tracker_val.compute(),
                    "Loss D_H val":loss_D_A_tracker_val.compute(),
                    "Loss D_C val":loss_D_B_tracker_val.compute(),
                    "Loss D val":loss_D_tracker_val.compute(),
                    
                    "Epoch":epoch, "Batch": i,
                    "PSNR H rec" : psnr_A_rec_meter.avg,
                    "PSNR C rec" : psnr_B_rec_meter.avg,

                    "PSNR H id" :psnr_A_id_meter.avg,
                    "PSNR C id" :psnr_B_id_meter.avg,

                    "mse H rec" :mse_A_rec_meter.avg,
                    "mse C rec" :mse_B_rec_meter.avg,

                    "mse H id ":mse_A_id_meter.avg,
                    "mse C id ":mse_B_id_meter.avg,

                    "mae H rec":mae_A_rec_meter.avg,
                    "mae C rec": mae_B_rec_meter.avg,

                    "mae H id ":mae_A_id_meter.avg,
                    "mae C id ":mae_B_id_meter.avg,

                    "SSIM H to C ":ssim_A_meter.avg,
                    "SSIM C to H ":ssim_B_meter.avg,
                    
                    "FID H to C":fid_A.compute(),
                    "FID C to H":fid_B.compute()
                    }
                    )
    
    
    print(f'Epoch [{epoch+1}/{config.epochs}] Batch [{i+1}/{len(dataloader_A_train)}] '
            f'Loss G: {loss_G.item():.4f} '
            f'Loss D: {loss_D.item():.4f} ')
            
    
            
    torch.save({
    'epoch': epoch + 1,
    'model_G_A2B_state_dict': netG_A2B.state_dict(),
    'model_G_B2A_state_dict': netG_B2A.state_dict(),
    'model_D_A_state_dict': netD_A.state_dict(),
    'model_D_B_state_dict': netD_B.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'loss_G': loss_G.item(),
    'loss_D_A': loss_D_A.item(),
    'loss_D_B': loss_D_B.item()
    }, os.path.join(config.path_checkpoints, f'checkpoint_epoch_{epoch+1}.pth'))

    print(f"Model saved after epoch {epoch+1}.")