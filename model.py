import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)



class Generator(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        out_features = 64
        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, out_features, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = out_features
        
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [nn.Conv2d(out_features, input_nc, 7, padding=3, padding_mode='reflect'), nn.Sigmoid()] 
        
        self.layers = nn.Sequential(*model)
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.layers(x)
    

class GenUNet(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=9):
        super(GenUNet, self).__init__()

        out_features = 64
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, out_features, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(out_features, out_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(out_features * 4) for _ in range(n_residual_blocks)]
        )

        # Upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 4 + out_features * 4, out_features * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_features * 2 + out_features * 2, out_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(out_features, input_nc, 7, padding=3, padding_mode='reflect'),
            nn.ReLU()
        )

        self.apply(weights_init_normal)

    def forward(self, x):
        # Initial convolution
        x1 = self.initial(x)

        # Downsampling
        d1 = self.down1(x1)
        d2 = self.down2(d1)

        # Residual blocks
        r = self.residuals(d2)

        # Upsampling with skip connections
        u1 = self.up1(torch.cat([r, d2], dim=1))
        u2 = self.up2(torch.cat([u1, d1], dim=1))

        # Final output
        return self.final(u2)
    
    
    
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    



class Discriminator(nn.Module):
    def __init__(self, input_shape: tuple[1, 256, 256]):
        super().__init__()
        channels, height, width = input_shape
        
        self.layers = nn.Sequential(
            DiscriminatorBlock(channels, 64, normalize=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, padding=1)
        ) 
        
    def forward(self, img):
        return self.layers(img)