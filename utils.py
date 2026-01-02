import os
from pathlib import Path
from PIL import Image
import torch


def get_name_without_extension(file_path: Path) -> str:
    for name in [
        "backpack",
        "bear",
        "can",
        "dog",
    ]:
        if name in file_path.stem.lower():
            return name
    raise ValueError(f"Unknown file name in path: {file_path}")


def save_images(images: list[Image.Image], output_path: Path):
    # Save images
    output_path = Path(output_path)
    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
    else:
        for i, img in enumerate(images):
            name = get_name_without_extension(output_path)
            if output_path.suffix:
                new_path = output_path.with_name(f"{output_path.stem}_{i}{output_path.suffix}")
            else:
                new_path = output_path.with_name(f"{output_path.name}_{i}")
            img.save(new_path)
            print(f"Saved image to {new_path}")


def save_merged(
    original: list[Image.Image],
    ground_truth: Image.Image,
    generated: list[Image.Image],
    output_path: Path = Path("merged_output.png"),
):
    """Save a merged image showing original + ground_truth on the first row and generated on the second row."""
    first_row = [*original, ground_truth]
    second_row = [*generated]

    if first_row:
        widths1, heights1 = zip(*(im.size for im in first_row))
        row1_width = sum(widths1)
        row1_height = max(heights1)
    else:
        row1_width = row1_height = 0

    if second_row:
        widths2, heights2 = zip(*(im.size for im in second_row))
        row2_width = sum(widths2)
        row2_height = max(heights2)
    else:
        row2_width = row2_height = 0

    total_width = max(row1_width, row2_width)
    total_height = row1_height + row2_height

    new_im = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    x_offset = 0
    for im in first_row:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in second_row:
        new_im.paste(im, (x_offset, row1_height))
        x_offset += im.size[0]

    new_im.save(output_path)
    print(f"Saved merged image to {output_path}")


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
    modified_noise_pred, default_noise_pred = noise_pred.chunk(2)
    reweighting_factor = gamma * (total_timesteps - current_timestep) / total_timesteps
    noise_pred = (
        default_noise_pred
        + (modified_noise_pred - default_noise_pred) * reweighting_factor
    )
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
default_seed = os.environ.get("SEED", 42)
SEED = int(default_seed)
