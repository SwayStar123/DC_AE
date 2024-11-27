import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import lpips
from pytorch_msssim import SSIM
from torch.amp import GradScaler, autocast
from pathlib import Path
import torchvision
from pathlib import Path
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

class ImageLogger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_reconstructions(self, original_images, reconstructed_images, phase, epoch):
        # Select 2 random images
        if original_images.size(0) > 2:
            idx = torch.randperm(original_images.size(0))[:2]
            original_images = original_images[idx]
            reconstructed_images = reconstructed_images[idx]
            
        # Denormalize images
        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)
            
        original_images = denorm(original_images)
        reconstructed_images = denorm(reconstructed_images)
        
        # Create grid
        grid = torch.cat([
            torch.cat([original_images[0], original_images[1]], dim=2),
            torch.cat([reconstructed_images[0], reconstructed_images[1]], dim=2)
        ], dim=1)
        
        # Save image
        torchvision.utils.save_image(
            grid,
            self.log_dir / f'reconstruction_phase{phase}_epoch{epoch}.png'
        )

class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files = []
        
    def save_checkpoint(self, state, phase, epoch):
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_phase{phase}_epoch{epoch}.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Save the checkpoint
        torch.save(state, checkpoint_path)
        self.checkpoint_files.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                
    def load_latest_checkpoint(self):
        if not self.checkpoint_files:
            # Check for existing checkpoints in directory
            checkpoint_files = sorted(self.save_dir.glob("checkpoint_phase*.pt"))
            if not checkpoint_files:
                return None
            self.checkpoint_files = checkpoint_files[-self.max_checkpoints:]
            
        latest_checkpoint = self.checkpoint_files[-1]
        if latest_checkpoint.exists():
            return torch.load(latest_checkpoint)
        return None

class DCAutoEncoderTrainer:
    def __init__(self, model, device='cuda', lr=1e-4, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.ssim_loss = SSIM(data_range=1.0, size_average=True)
        self.discriminator = self._build_discriminator().to(device)
        self.scaler = GradScaler()
        self.image_logger = ImageLogger()
        
        # Optimizers
        self.opt_ae = optim.AdamW(self.model.parameters(), lr=lr)
        self.opt_disc = optim.AdamW(self.discriminator.parameters(), lr=lr)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Try to load latest checkpoint
        self.resume_from_checkpoint()
        
    def _build_discriminator(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 3, 1, 1)
        )
        
    def resume_from_checkpoint(self):
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_ae.load_state_dict(checkpoint['optimizer_ae_state_dict'])
            self.opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
            print(f"Resumed from checkpoint: Phase {checkpoint['phase']}, Epoch {checkpoint['epoch']}")
            
    def save_checkpoint(self, phase, epoch, extra_state=None):
        state = {
            'phase': phase,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_ae_state_dict': self.opt_ae.state_dict(),
            'optimizer_disc_state_dict': self.opt_disc.state_dict(),
        }
        if extra_state:
            state.update(extra_state)
        
        self.checkpoint_manager.save_checkpoint(state, phase, epoch)
        
    def train_phase1(self, train_loader, epochs, start_epoch=0):
        """Low-resolution full training phase"""
        print("Phase 1: Low-resolution full training")
        
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            for batch in tqdm(train_loader, total=len(train_loader)):
                images = batch[0].to(self.device)
                
                with autocast('cuda'):
                    reconstructed = self.model(images)

                    l1_loss = nn.functional.l1_loss(reconstructed, images)
                    lpips_loss = self.lpips_loss(reconstructed, images).mean()
                    # ssim_loss = 1 - self.ssim_loss(reconstructed, images)
                    
                    # loss = l1_loss + lpips_loss + 0.1 * ssim_loss
                    loss = l1_loss + lpips_loss
                    total_loss += loss.item()
                
                self.opt_ae.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt_ae)
                self.scaler.update()
            
            self.image_logger.log_reconstructions(
                images.cpu(),
                reconstructed.detach().cpu(),
                phase=1,
                epoch=epoch+1
            )
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(
                phase=1, 
                epoch=epoch+1,
                extra_state={'avg_loss': avg_loss}
            )
                
    def train_phase2(self, high_res_loader, epochs, start_epoch=0):
        """High-resolution latent adaptation phase"""
        print("Phase 2: High-resolution latent adaptation")
        
        # Freeze all except middle layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.middle.parameters():
            param.requires_grad = True
            
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            for batch in tqdm(high_res_loader, total=len(high_res_loader)):
                images = batch[0].to(self.device)
                
                with autocast('cuda'):
                    reconstructed = self.model(images)
                    
                    l1_loss = nn.functional.l1_loss(reconstructed, images)
                    lpips_loss = self.lpips_loss(reconstructed, images).mean()
                    
                    loss = l1_loss + lpips_loss
                    total_loss += loss.item()
                
                self.opt_ae.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt_ae)
                self.scaler.update()

            self.image_logger.log_reconstructions(
                images.cpu(),
                reconstructed.detach().cpu(),
                phase=2,
                epoch=epoch+1
            )
            
            avg_loss = total_loss / len(high_res_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(
                phase=2,
                epoch=epoch+1,
                extra_state={'avg_loss': avg_loss}
            )
                
    def train_phase3(self, train_loader, epochs, start_epoch=0):
        """Low-resolution local refinement phase with GAN"""
        print("Phase 3: Low-resolution local refinement with GAN")
        
        # Freeze all except decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder_stages.parameters():
            param.requires_grad = True
        for param in self.model.final_blocks.parameters():
            param.requires_grad = True
            
        # First train only GAN for 10% of epochs
        gan_only_epochs = max(1, epochs // 10)
        
        current_epoch = start_epoch
        if current_epoch < gan_only_epochs:
            print(f"Training GAN only for {gan_only_epochs - current_epoch} epochs")
            
            for epoch in range(current_epoch, gan_only_epochs):
                total_d_loss = 0
                for batch in tqdm(train_loader, total=len(train_loader)):
                    images = batch[0].to(self.device)
                    
                    with autocast('cuda'):
                        reconstructed = self.model(images)

                        real_pred = self.discriminator(images)
                        fake_pred = self.discriminator(reconstructed.detach())
                        
                        d_loss = (
                            nn.functional.binary_cross_entropy_with_logits(
                                real_pred, torch.ones_like(real_pred)
                            ) +
                            nn.functional.binary_cross_entropy_with_logits(
                                fake_pred, torch.zeros_like(fake_pred)
                            )
                        )
                        total_d_loss += d_loss.item()
                    
                    self.opt_disc.zero_grad()
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.opt_disc)
                    self.scaler.update()
                
                avg_d_loss = total_d_loss / len(train_loader)
                print(f"GAN Epoch {epoch+1}/{gan_only_epochs}, Discriminator Loss: {avg_d_loss:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(
                    phase=3,
                    epoch=epoch+1,
                    extra_state={
                        'gan_only_phase': True,
                        'avg_d_loss': avg_d_loss
                    }
                )
        
        # Then train both together
        print("Training both GAN and decoder head")
        start = max(gan_only_epochs, current_epoch)
        for epoch in range(start, epochs):
            total_g_loss = 0
            total_d_loss = 0
            
            for batch in tqdm(train_loader, total=len(train_loader)):
                images = batch[0].to(self.device)
                
                # Train discriminator
                with autocast('cuda'):
                    reconstructed = self.model(images)
                    
                    real_pred = self.discriminator(images)
                    fake_pred = self.discriminator(reconstructed.detach())
                    
                    d_loss = (
                        nn.functional.binary_cross_entropy_with_logits(
                            real_pred, torch.ones_like(real_pred)
                        ) +
                        nn.functional.binary_cross_entropy_with_logits(
                            fake_pred, torch.zeros_like(fake_pred)
                        )
                    )
                    total_d_loss += d_loss.item()
                
                self.opt_disc.zero_grad()
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.opt_disc)
                
                # Train generator (decoder head)
                with autocast('cuda'):
                    reconstructed = self.model(images)
                    fake_pred = self.discriminator(reconstructed)
                    
                    l1_loss = nn.functional.l1_loss(reconstructed, images)
                    lpips_loss = self.lpips_loss(reconstructed, images).mean()
                    gan_loss = nn.functional.binary_cross_entropy_with_logits(
                        fake_pred, torch.ones_like(fake_pred)
                    )
                    
                    g_loss = l1_loss + lpips_loss + 0.1 * gan_loss
                    total_g_loss += g_loss.item()
                
                self.opt_ae.zero_grad()
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.opt_ae)
                self.scaler.update()
            
            self.image_logger.log_reconstructions(
                images.cpu(),
                reconstructed.detach().cpu(),
                phase=3,
                epoch=epoch+1
            )

            avg_g_loss = total_g_loss / len(train_loader)
            avg_d_loss = total_d_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(
                phase=3,
                epoch=epoch+1,
                extra_state={
                    'gan_only_phase': False,
                    'avg_g_loss': avg_g_loss,
                    'avg_d_loss': avg_d_loss
                }
            )

class CelebAHQDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)
        return (image,)  # Return as tuple to match CIFAR format

def get_dataloaders(batch_size_low=16, batch_size_high=4, num_workers=4):
    # Load dataset
    dataset = load_dataset("mattymchen/celeba-hq", cache_dir='../../datasets/CelebA-HQ')
    
    # Setup transforms
    low_res_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    high_res_transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    low_res_dataset = CelebAHQDataset(
        dataset['train'],
        transform=low_res_transform
    )
    
    high_res_dataset = CelebAHQDataset(
        dataset['train'],
        transform=high_res_transform
    )
    
    # Create dataloaders
    low_res_loader = DataLoader(
        low_res_dataset,
        batch_size=batch_size_low,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    high_res_loader = DataLoader(
        high_res_dataset,
        batch_size=batch_size_high,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return low_res_loader, high_res_loader

if __name__ == "__main__":
    from dc_ae import DeepCompressionAutoencoder

    datasets.config.HF_HUB_OFFLINE = 1
    
    low_res_loader, high_res_loader = get_dataloaders(batch_size_low=20, batch_size_high=2)

    # Create model and trainer
    model = DeepCompressionAutoencoder(spatial_compression=64, latent_channels=1024, initial_channels=16)
    trainer = DCAutoEncoderTrainer(model, checkpoint_dir='dc_ae_checkpoints')

    # Print parameters
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Discriminator parameters: {sum(p.numel() for p in trainer.discriminator.parameters() if p.requires_grad)}")
    
    # Three-phase training
    # trainer.train_phase1(low_res_loader, epochs=10)
    trainer.train_phase2(high_res_loader, epochs=1)
    # trainer.train_phase3(low_res_loader, epochs=10)