"""
Latent Diffusion Image Generation Script

This module provides modular building blocks for latent diffusion image generation.
The architecture is designed to allow easy modification of U-Net attention mechanisms.
"""

import torch
from PIL import Image
from models.pipeline import GenerationConfig, create_pipeline
from utils import save_images

def main():
    """Main entry point for CLI usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate images using latent diffusion models")
    parser.add_argument("--prompt", type=str, default="Bear Plushie", help="Text prompt for generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--model", type=str, default="sd-legacy/stable-diffusion-v1-5", help="Model ID")
    # parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3-medium_amdgpu", help="Model ID")
    # parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1-base", help="Model ID")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=70, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Classifier-free guidance scale")
    parser.add_argument("--contrast-strength", type=float, default=1.67, help="Contrast strength (beta)")
    parser.add_argument("--attention-temperature", type=float, default=0.4, help="Attention temperature")
    parser.add_argument("--swap-guidance", type=float, default=3.5, help="Swap-guidance strength (gamma)")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim", "euler", "pndm"], help="Scheduler type")
    
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
        guidance_scale=args.guidance_scale,
        contrast_strength=args.contrast_strength,
        attention_temperature=args.attention_temperature,
        swap_guidance=args.swap_guidance,
        num_images=args.num_images,
        seed=args.seed
    )
    pipeline = create_pipeline(
        config=config,
        model_id=args.model,
        device=device,
        scheduler_type=args.scheduler
    )
    
    print(f"Generating image with prompt: {args.prompt}")
    
    def progress_callback(step, timestep, latents):
        print(f"Step {step + 1}/{args.steps}")

    samples_paths = [
        "inputs/bear/images/bear_plushie_01.jpg", # Q
        "inputs/backpack/images/backpack_02.jpg", # K 
        "inputs/backpack/alphas/backpack_02.png", # V
    ]

    samples = [Image.open(p).convert("RGB").resize((256, 256)) for p in samples_paths]

    images = pipeline.generate(samples, config, callback=progress_callback)
    save_images(images, args.output)

if __name__ == "__main__":
    main()
