import os
from PIL import Image
import torch

def save_images(images: list[Image.Image], output_path: str):
    # Save images
    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
    else:
        base_name = output_path.rsplit(".", 1)
        for i, img in enumerate(images):
            if len(base_name) > 1:
                output_path = f"{base_name[0]}_{i}.{base_name[1]}"
            else:
                output_path = f"{output_path}_{i}"
            img.save(output_path)
            print(f"Saved image to {output_path}")

def apply_swap_guidance(
        noise_pred: torch.Tensor,
        current_timestep: int,
        total_timesteps: int,
        gamma: float = 3.5,
    ) -> torch.Tensor:
        """
        Apply swap-guidance to the predicted noise.
        The first 4 samples are conditioned (A,B,C,D), the last 4 are default (A,B,C,D).
        Args:
            noise_pred: Predicted noise tensor
            current_timestep: Current timestep index
            total_timesteps: Total number of timesteps
            gamma: Swap-guidance strength
        Returns:
            Modified noise prediction with swap-guidance applied
        """
        modified_noise_pred = noise_pred[4:, ...].clone() # First 4 are conditioned
        default_noise_pred = noise_pred[:4, ...].clone() # Last 4 are default
        reweighting_factor = gamma * (total_timesteps - current_timestep) / total_timesteps
        noise_pred = default_noise_pred + (modified_noise_pred - default_noise_pred) * reweighting_factor
        return torch.cat([noise_pred, default_noise_pred], dim=0)


def adain(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply Adaptive Instance Normalization (AdaIN).
    
    Args:
        x: Content tensor
        y: Style tensor
        eps: Small value to avoid division by zero
        
    Returns:
        AdaIN applied tensor
    """
    x = x.unsqueeze(0)        
    y = y.unsqueeze(0)
    mu_x = x.mean(dim=[2, 3], keepdim=True)
    sigma_x = x.std(dim=[2, 3], keepdim=True) + eps
    
    mu_y = y.mean(dim=[2, 3], keepdim=True)
    sigma_y = y.std(dim=[2, 3], keepdim=True) + eps
    
    normalized_x = (x - mu_x) / sigma_x
    adain_x = normalized_x * sigma_y + mu_y
    
    return adain_x.squeeze(0)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
default_seed = os.environ.get('SEED', 42)
SEED=int(default_seed)
