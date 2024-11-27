import sys
sys.path.append('../dcae')

from dcae import DCAE
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision

if __name__ == "__main__":


    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    # Load image from image.png
    img = Image.open("anime.jpg").convert("RGB")
    img = transform(img)[None].to(torch.bfloat16).to(device)
    image = denorm(img).cpu()

    # dc_ae_f32 = DCAE("dc-ae-f32c32-mix-1.0", device=device, dtype=torch.bfloat16, cache_dir="../../models/dc_ae").eval()

    # latent_f32 = dc_ae_f32.encode(img)
    # recon_f32 = dc_ae_f32.decode(latent_f32)
    # reconstructed_image_f32 = denorm(recon_f32).to(torch.float32).cpu()

    # del dc_ae_f32
    # del latent_f32
    # del recon_f32

    dc_ae_f64 = DCAE("dc-ae-f64c128-mix-1.0", device=device, dtype=torch.bfloat16, cache_dir="../../models/dc_ae").eval()

    latent_f64 = dc_ae_f64.encode(img)
    recon_f64 = dc_ae_f64.decode(latent_f64)
    reconstructed_image_f64 = denorm(recon_f64).to(torch.float32).cpu()

    del dc_ae_f64
    del latent_f64
    del recon_f64

    # dc_ae_f128 = DCAE("dc-ae-f128c512-mix-1.0", device=device, dtype=torch.bfloat16, cache_dir="../../models/dc_ae").eval()

    # latent_f128 = dc_ae_f128.encode(img)
    # recon_f128 = dc_ae_f128.decode(latent_f128)
    # reconstructed_image_f128 = denorm(recon_f128).to(torch.float32).cpu()

    # del dc_ae_f128
    # del latent_f128
    # del recon_f128

    # Create grid
    grid = torch.cat([
        image, reconstructed_image_f64
    ], dim=2)

    grid = grid.to(torch.float32)

    torchvision.utils.save_image(
        grid,
        "test_log/og_vs_recon.png",
        normalize=False,
    )