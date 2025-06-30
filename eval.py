import argparse
from tqdm import tqdm
import torch
from torchmetrics.regression import MeanAbsoluteError as mae
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torchmetrics.regression import MeanSquaredError as mse
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import h5py
from torchmetrics.image.fid import FrechetInceptionDistance as frechet

def tf_to_np(tensor):
    return tensor.detach().cpu().squeeze().numpy()

def evaluate(path_post):
    with h5py.File(path_post, 'a') as h5f:
        n_samples = h5f['fake_A'][:].shape[0]
        batch_A = torch.tensor(h5f['batch_A'][:])
        batch_B = torch.tensor(h5f['batch_B'][:])
        fake_A = torch.tensor(h5f['fake_A'][:])
        fake_B = torch.tensor(h5f['fake_B'][:])
        id_A = torch.tensor(h5f['same_A'][:])
        id_B = torch.tensor(h5f['same_B'][:])
        rec_A = torch.tensor(h5f['recovered_A'][:])
        rec_B = torch.tensor(h5f['recovered_B'][:])
        masked_fake_A = torch.tensor(h5f['masked_fake_A'][:])
        masked_fake_B = torch.tensor(h5f['masked_fake_B'][:])
            
        range_A = torch.max(batch_A) - torch.min(batch_A)
        range_B = torch.max(batch_B) - torch.min(batch_B)

        # Initialize metrics
        
        ssim_A = ssim(data_range=range_A)
        ssim_B = ssim(data_range=range_B)
        psnr_A = psnr(data_range=range_A)
        psnr_B = psnr(data_range=range_B)
        mse_ = mse()
        mae_ = mae()
        frechet_dis_A = frechet(feature=64)
        frechet_dis_B = frechet(feature=64)
        
        for i in tqdm(range(0, batch_A.shape[0], 50)):
            frechet_dis_A.update(torch.unsqueeze(batch_A[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=True)
            frechet_dis_A.update(torch.unsqueeze(fake_B[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=False)
            frechet_dis_B.update(torch.unsqueeze(batch_B[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=True)
            frechet_dis_B.update(torch.unsqueeze(fake_A[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=False)
        
        # Existing metrics
        psnr_A_rec = psnr_A(batch_A, rec_A)
        psnr_B_rec = psnr_B(batch_B, rec_B)
        psnr_A_id = psnr_A(batch_A, id_A)
        psnr_B_id = psnr_B(batch_B, id_B)
        mse_A_rec = mse_(batch_A, rec_A)
        mse_B_rec = mse_(batch_B, rec_B)
        mse_A_id = mse_(batch_A, id_A)
        mse_B_id = mse_(batch_B, id_B)
        mae_A_rec = mae_(batch_A, rec_A)
        mae_B_rec = mae_(batch_B, rec_B)
        mae_A_id = mae_(batch_A, id_A)
        mae_B_id = mae_(batch_B, id_B)
        ssim_A_ = ssim_A(torch.unsqueeze(batch_A, 1), torch.unsqueeze(fake_B, 1))
        ssim_B_ = ssim_B(torch.unsqueeze(batch_B, 1), torch.unsqueeze(fake_A, 1))

        # Masked metrics
        psnr_A_masked = psnr_A(batch_A, masked_fake_B)
        psnr_B_masked = psnr_B(batch_B, masked_fake_A)
        mse_A_masked = mse_(batch_A, masked_fake_B)
        mse_B_masked = mse_(batch_B, masked_fake_A)
        mae_A_masked = mae_(batch_A, masked_fake_B)
        mae_B_masked = mae_(batch_B, masked_fake_A)
        ssim_A_masked = ssim_A(torch.unsqueeze(batch_A, 1), torch.unsqueeze(masked_fake_B, 1))
        ssim_B_masked = ssim_B(torch.unsqueeze(batch_B, 1), torch.unsqueeze(masked_fake_A, 1))

        # FID for masked_fake_A and masked_fake_B
        frechet_dis_masked_A = frechet(feature=64)
        frechet_dis_masked_B = frechet(feature=64)
        for i in tqdm(range(0, batch_A.shape[0], 50)):
            frechet_dis_masked_A.update(torch.unsqueeze(batch_B[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=True)
            frechet_dis_masked_A.update(torch.unsqueeze(masked_fake_A[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=False)
            frechet_dis_masked_B.update(torch.unsqueeze(batch_A[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=True)
            frechet_dis_masked_B.update(torch.unsqueeze(masked_fake_B[i:i + 50, :, :].type(torch.uint8), 1).repeat(1,3,1,1)*255, real=False)

        print(f"PSNR A rec {psnr_A_rec}")
        print(f"PSNR B rec {psnr_B_rec}")
        print(f"PSNR A id {psnr_A_id}")
        print(f"PSNR B id {psnr_B_id}")
        print(f"mse A rec {mse_A_rec}")
        print(f"mse B rec {mse_B_rec}")
        print(f"mse A id {mse_A_id}")
        print(f"mse B id {mse_B_id}")
        print(f"mae A rec {mae_A_rec}")
        print(f"mae B rec {mae_B_rec}")
        print(f"mae A id {mae_A_id}")
        print(f"mae B id {mae_B_id}")
        print(f"SSIM A {ssim_A_}")
        print(f"SSIM B {ssim_B_}")
        print(f'FID A {frechet_dis_A.compute()}')
        print(f'FID B {frechet_dis_B.compute()}')
        print(f"PSNR A masked_fake_B {psnr_A_masked}")
        print(f"PSNR B masked_fake_A {psnr_B_masked}")
        print(f"mse A masked_fake_B {mse_A_masked}")
        print(f"mse B masked_fake_A {mse_B_masked}")
        print(f"mae A masked_fake_B {mae_A_masked}")
        print(f"mae B masked_fake_A {mae_B_masked}")
        print(f"SSIM A masked_fake_B {ssim_A_masked}")
        print(f"SSIM B masked_fake_A {ssim_B_masked}")
        print(f'FID masked_fake_A {frechet_dis_masked_A.compute()}')
        print(f'FID masked_fake_B {frechet_dis_masked_B.compute()}')

def main():
    parser = argparse.ArgumentParser(description="Evaluate model metrics from H5 file.")
    parser.add_argument('--path_post', type=str, required=True, help='Path to the processed H5 file')
    args = parser.parse_args()
    evaluate(args.path_post)

if __name__ == "__main__":
    main()
