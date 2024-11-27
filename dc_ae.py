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

class ConvBlock(nn.Module):
    """Additional convolution block for increasing depth"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x + residual

class EncoderStage(nn.Module):
    """Enhanced encoder stage with multiple conv blocks"""
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        self.downsample = ResidualDownsampleBlock(in_channels, out_channels)
        self.blocks = nn.ModuleList([
            ConvBlock(out_channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class DecoderStage(nn.Module):
    """Enhanced decoder stage with multiple conv blocks"""
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        self.upsample = ResidualUpsampleBlock(in_channels, out_channels)
        self.blocks = nn.ModuleList([
            ConvBlock(out_channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class DeepCompressionAutoencoder(nn.Module):
    def __init__(self, spatial_compression=64, in_channels=3, latent_channels=128, initial_channels=64, num_blocks=3):
        super().__init__()
        
        # Calculate number of downsampling stages needed
        self.num_stages = int(torch.log2(torch.tensor(spatial_compression)))
        initial_channels = initial_channels
        
        # Encoder
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, 3, padding=1),
            nn.GroupNorm(8, initial_channels),
            nn.SiLU(),
            ConvBlock(initial_channels),
            ConvBlock(initial_channels)
        )
        
        # Build encoder stages
        self.encoder_stages = nn.ModuleList()
        current_channels = initial_channels
        for i in range(self.num_stages):
            out_channels = current_channels * 2
            self.encoder_stages.append(
                EncoderStage(current_channels, out_channels, num_blocks)
            )
            current_channels = out_channels
            
        # Middle stage - increased capacity
        middle_channels = current_channels * 2
        self.middle = nn.Sequential(
            nn.Conv2d(current_channels, middle_channels, 1),
            nn.GroupNorm(8, middle_channels),
            nn.SiLU(),
            *[ConvBlock(middle_channels) for _ in range(num_blocks)],
            nn.Conv2d(middle_channels, latent_channels, 1)
        )
        
        # Build decoder stages
        self.decoder_stages = nn.ModuleList()
        current_channels = latent_channels
        for i in range(self.num_stages):
            out_channels = max(current_channels // 2, initial_channels // 2)
            self.decoder_stages.append(
                DecoderStage(current_channels, out_channels, num_blocks)
            )
            current_channels = out_channels
            
        # Final output with additional processing
        self.final_blocks = nn.Sequential(
            ConvBlock(current_channels),
            ConvBlock(current_channels),
            nn.Conv2d(current_channels, in_channels, 3, padding=1)
        )
        
    def encode(self, x):
        x = self.initial_conv(x)
        
        # Encoder path
        for stage in self.encoder_stages:
            x = stage(x)
            
        # Middle stage
        latent = self.middle(x)
        return latent
        
    def decode(self, latent):
        x = latent
        
        # Decoder path
        for stage in self.decoder_stages:
            x = stage(x)
            
        # Final output
        x = self.final_blocks(x)
        return x
        
    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

if __name__ == "__main__":
    # Instantiate the model
    f128_model = DeepCompressionAutoencoder(spatial_compression=8, latent_channels=16)
    print(f128_model)
    # Print parameter count
    print(sum(p.numel() for p in f128_model.parameters() if p.requires_grad))
    # Print storage size of weights
    b = sum(p.numel() * p.element_size() for p in f128_model.parameters() if p.requires_grad)
    print(f"Model size: {b / 1024 ** 3:.2f} GB")