"""
Latent Diffusion Image Generation Script

This module provides modular building blocks for latent diffusion image generation.
The architecture is designed to allow easy modification of U-Net attention mechanisms.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datamodule import PairsDataset, PairsInput
from ml_modules.pipeline import GenerationConfig, create_pipeline
from utils import save_images, save_merged


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate images using latent diffusion models"
    )
    parser.add_argument(
        "--prompt", type=str, default="Bear Plushie", help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="", help="Negative prompt"
    )
    parser.add_argument(
        "--model", type=str, default="sd-legacy/stable-diffusion-v1-5", help="Model ID"
    )
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument(
        "--steps", type=int, default=70, help="Number of inference steps"
    )
    parser.add_argument(
        "--contrast-strength", type=float, default=1.67, help="Contrast strength (beta)"
    )
    parser.add_argument(
        "--attention-temperature", type=float, default=0.4, help="Attention temperature"
    )
    parser.add_argument(
        "--swap-guidance",
        type=float,
        default=3.5,
        help="Swap-guidance strength (gamma)",
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "euler", "pndm"],
        help="Scheduler type",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model: {args.model}")

    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        contrast_strength=args.contrast_strength,
        attention_temperature=args.attention_temperature,
        swap_guidance=args.swap_guidance,
        num_images=args.num_images,
        seed=args.seed,
    )
    pipeline = create_pipeline(
        config=config, model_id=args.model, device=device, scheduler_type=args.scheduler
    )

    dl = DataLoader(
        dataset=PairsDataset(
            root="inputs", target_size=(args.height, args.width)
        ),  # Dummy dataset for samples
        batch_size=1,
        collate_fn=lambda x: x[0],
        shuffle=False,
    )

    for batch in tqdm(dl, desc="Batches", position=0):
        assert isinstance(batch, PairsInput), (
            f"Batch should be of type PairsInput, got {type(batch)}"
        )
        (
            input_image,
            guid_image,
            guid_ground_truth,
            ground_truth_mask,
            input_category,
        ) = batch.to_tuple()
        out_dir = Path(args.outdir) / input_category
        out_dir.mkdir(parents=True, exist_ok=True)

        samples = [input_image, guid_image, guid_ground_truth]
        images = pipeline.generate(samples, config, callback=None)
        save_images(images, out_dir / "generated.png")
        save_merged(samples, ground_truth_mask, images, out_dir / "merged_output.png")


if __name__ == "__main__":
    main()
