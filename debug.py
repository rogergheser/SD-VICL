"""
Latent Diffusion Image Generation Script

This module provides modular building blocks for latent diffusion image generation.
The architecture is designed to allow easy modification of U-Net attention mechanisms.
"""

from functools import partial
from pathlib import Path
import torch
from ml_modules.pipeline import GenerationConfig, create_pipeline
from PIL import Image
from utils import save_images, save_merged

DEBUG = True


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate images using latent diffusion models"
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="", help="Negative prompt"
    )
    parser.add_argument(
        "--model", type=str, default="sd-legacy/stable-diffusion-v1-5", help="Model ID"
    )
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
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

    print(f"Generating image with prompt: {args.prompt}")

    def progress_callback(step, timestep, latents):
        print(f"Step {step + 1}/{args.steps}")

    def save_call_back(step, timestep, latents, gt_mask, original_images):
        if step % 10 == 0 or step == args.steps - 1:
            print(f"Step {step + 1}/{args.steps}")
            images = pipeline.latent_processor.decode_latents(latents)
            save_merged(
                original_images,
                gt_mask,
                images,
                Path(args.outdir) / f"debug_step_{step + 1:03d}.png",
            )

    samples_paths = [
        "inputs/bear/images/bear_plushie_01.jpg",  # Q
        # "inputs/backpack/images/backpack_05.jpg",  # Q
        "inputs/backpack/images/backpack_02.jpg",  # K
        "inputs/backpack/alphas/backpack_02.png",  # V
    ]
    gt_mask_path = samples_paths[0].replace("images", "alphas").replace(".jpg", ".png")

    samples = [
        Image.open(p).convert("RGB").resize((args.width, args.height))
        for p in samples_paths
    ]
    images = pipeline.generate(
        samples,
        config,
        callback=partial(
            save_call_back,
            gt_mask=Image.open(gt_mask_path)
            .convert("L")
            .resize((args.width, args.height)),
            original_images=samples,
        ),
        debug=True,
    )
    save_images(images, Path(args.outdir) / "backpack.png")


if __name__ == "__main__":
    main()
