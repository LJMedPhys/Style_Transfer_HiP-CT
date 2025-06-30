import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import torch


class CycleGANLoss:
    def __init__(self, device):
        self.adversarial_loss = nn.MSELoss().to(device)
        self.cycle_loss = nn.L1Loss().to(device)
        self.identity_loss = nn.L1Loss().to(device)

    def __call__(self, pred, target, loss_type):
        if loss_type == 'adversarial':
            return self.adversarial_loss(pred, target)
        elif loss_type == 'cycle':
            return self.cycle_loss(pred, target)
        elif loss_type == 'identity':
            return self.identity_loss(pred, target)
        

class Multiscale_SSIM:
    def __init__(self, pred):
        self.range = 1.0
        self.kernel_sizes = []
        self.shape = list(pred.shape[-2:])
        self.ssim_list = []
        
        for i in range(4):
            if i ==0:
                if self.shape[0]%2 == 0:
                    self.shape[0] -= 1
                if self.shape[1]%2 == 0:
                    self.shape[1] -= 1
                self.ssim_list.append(ssim(gaussian_kernel=False, kernel_size=(self.shape[0], self.shape[1]),data_range=self.range))
            else:
                self.ssim_list.append(ssim(gaussian_kernel=False, kernel_size=(self.shape[0]//(i*2), self.shape[1]//(i*2)),data_range=self.range))
                
            
    def __call__(self, pred, target):
        
        losses = []
        for loss_ssim in self.ssim_list:
            losses.append(loss_ssim(pred, target))
            
        return(torch.mean(torch.tensor(losses)))
                
    
               
               
                
                
                
        
        
        