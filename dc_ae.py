import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
    def space_to_channel(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//2, 2, w//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c*4, h//2, w//2)
        return x
        
    def channel_average(self, x, out_channels):
        x = x.chunk(2, dim=1)
        return sum(x) / 2
        
    def forward(self, x):
        # Residual path with space-to-channel shortcut
        shortcut = self.space_to_channel(x)
        shortcut = self.channel_average(shortcut, shortcut.size(1)//2)
        
        # Main path
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)
        
        return x + shortcut

class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
    def channel_to_space(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c//4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c//4, h*2, w*2)
        return x
        
    def channel_duplicate(self, x, target_channels):
        repeats = target_channels // x.size(1)
        return torch.cat([x] * repeats, dim=1)
        
    def forward(self, x):
        # Residual path with channel-to-space shortcut
        shortcut = self.channel_to_space(x)
        
        # Main path
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)

        shortcut = self.channel_duplicate(shortcut, x.size(1))

        return x + shortcut

class DeepCompressionAutoencoder(nn.Module):
    def __init__(self, spatial_compression=64, in_channels=3, latent_channels=128):
        super().__init__()
        
        # Calculate number of downsampling stages needed
        self.num_stages = int(torch.log2(torch.tensor(spatial_compression)))
        initial_channels = 64
        
        # Encoder
        self.initial_conv = nn.Conv2d(in_channels, initial_channels, 3, padding=1)
        
        # Build encoder stages
        self.encoder_stages = nn.ModuleList()
        current_channels = initial_channels
        for i in range(self.num_stages):
            # out_channels = min(current_channels * 2, 1024)
            out_channels = current_channels * 2
            self.encoder_stages.append(ResidualDownsampleBlock(current_channels, out_channels))
            current_channels = out_channels
            
        # Middle stage
        self.middle_conv1 = nn.Conv2d(current_channels, current_channels, 1)
        self.middle_conv2 = nn.Conv2d(current_channels, latent_channels, 1)
        
        # Build decoder stages
        self.decoder_stages = nn.ModuleList()
        current_channels = latent_channels
        for i in range(self.num_stages):
            out_channels = max(current_channels // 2, initial_channels)
            self.decoder_stages.append(ResidualUpsampleBlock(current_channels, out_channels))
            current_channels = out_channels
            
        # Final output
        self.final_conv = nn.Conv2d(current_channels, in_channels, 3, padding=1)
        
    def encode(self, x):
        x = self.initial_conv(x)
        
        # Encoder path
        for stage in self.encoder_stages:
            x = stage(x)
            
        # Middle stage
        x = self.middle_conv1(x)
        x = F.silu(x)
        latent = self.middle_conv2(x)
        
        return latent
        
    def decode(self, latent):
        x = latent
        
        # Decoder path
        for stage in self.decoder_stages:
            x = stage(x)
            
        # Final output
        x = self.final_conv(x)
        return x
        
    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

if __name__ == "__main__":
    # Instantiate the model
    f128_model = DeepCompressionAutoencoder(spatial_compression=64, latent_channels=128)
    print(f128_model)
    # Print parameter count
    print(sum(p.numel() for p in f128_model.parameters() if p.requires_grad))
    # Print storage size of weights
    b = sum(p.numel() * p.element_size() for p in f128_model.parameters() if p.requires_grad)
    print(f"Model size: {b / 1024 ** 3:.2f} GB")