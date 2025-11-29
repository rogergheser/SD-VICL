"""
Hydra-based Experiment Runner for SD-VICL

This module provides a Hydra-powered entry point for running image generation experiments.
Use YAML configuration files for reproducible experiment setup.

Usage:
    # Run with default config
    python run_experiment.py
    
    # Override parameters
    python run_experiment.py generation.prompt="a cat in space" generation.seed=42
    
    # Use a different config file
    python run_experiment.py --config-name=my_experiment
    
    # Multi-run with parameter sweeps
    python run_experiment.py -m generation.seed=1,2,3,4,5
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from generate import create_pipeline, GenerationConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """
    Run an image generation experiment using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object containing all experiment parameters
    """
    # Print configuration for experiment tracking
    print("=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)
    
    # Extract configuration values
    model_id = cfg.model.id
    scheduler_type = cfg.model.scheduler
    device = cfg.hardware.device
    output_path = cfg.output.path
    
    # Create pipeline
    print(f"Loading model: {model_id}")
    pipeline = create_pipeline(
        model_id=model_id,
        device=device,
        scheduler_type=scheduler_type
    )
    
    # Create generation config
    config = GenerationConfig(
        prompt=cfg.generation.prompt,
        negative_prompt=cfg.generation.negative_prompt,
        height=cfg.generation.height,
        width=cfg.generation.width,
        num_inference_steps=cfg.generation.num_inference_steps,
        guidance_scale=cfg.generation.guidance_scale,
        num_images=cfg.generation.num_images,
        seed=cfg.generation.seed
    )
    
    print(f"Generating image with prompt: {config.prompt}")
    
    # Progress callback
    total_steps = cfg.generation.num_inference_steps
    def progress_callback(step, timestep, latents):
        print(f"Step {step + 1}/{total_steps}")
    
    # Generate images
    images = pipeline.generate(config, callback=progress_callback)
    
    # Save images
    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved image to {output_path}")
    else:
        base_name = output_path.rsplit(".", 1)
        for i, img in enumerate(images):
            if len(base_name) > 1:
                save_path = f"{base_name[0]}_{i}.{base_name[1]}"
            else:
                save_path = f"{output_path}_{i}"
            img.save(save_path)
            print(f"Saved image to {save_path}")
    
    print("=" * 50)
    print("Experiment completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_experiment()
