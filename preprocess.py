import torch
from torchvision import transforms

import sys
sys.path.append('../dcae')

from dcae import DCAE
from datasets import load_dataset, Features, Array3D, Value, Image

# Constants
COMPRESSION_FACTOR = "f32"  # Options: "f32", "f64", "f128"
OG_DATASET = "pravsels/FFHQ_1024"
UPLOAD_DATASET = f"SwayStar123/FFHQ_1024_DC-AE_{COMPRESSION_FACTOR}"
MODEL_PATHS = {
    "f32": "dc-ae-f32c32-mix-1.0",
    "f64": "dc-ae-f64c128-mix-1.0",
    "f128": "dc-ae-f128c512-mix-1.0"
}
CACHE_DIR = "../../models/dc_ae"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
BATCH_SIZE = 32

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
])

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

def main():
    model_name = MODEL_PATHS[COMPRESSION_FACTOR]
    dc_ae = DCAE(model_name, device=DEVICE, dtype=DTYPE, cache_dir=CACHE_DIR).eval()

    # Get the shape of the latent representations
    dummy_input = torch.randn(1, 3, 1024, 1024).to(DTYPE).to(DEVICE)
    with torch.no_grad():
        dummy_latent = dc_ae.encode(dummy_input).cpu()
    latent_shape = dummy_latent.shape[1:]
    print(f"Latent shape: {latent_shape}")
    
    features = Features({
        'image': Image(),
        'label': Value('int64'),
        'latent': Array3D(dtype='bfloat16', shape=latent_shape)
    })
    
    dataset = load_dataset(OG_DATASET, split="train")

    def process_batch(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        img_tensors = torch.stack([transform(img) for img in images]).to(DTYPE).to(DEVICE)
        with torch.no_grad():
            latents = dc_ae.encode(img_tensors).cpu().to(torch.float16).numpy()
        batch["latent"] = latents
        return batch

    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        features=features,
    )

    processed_dataset.push_to_hub(
        repo_id=UPLOAD_DATASET
    )
    
    print(f"Dataset uploaded to Hugging Face Hub: {UPLOAD_DATASET}")

if __name__ == "__main__":
    main()