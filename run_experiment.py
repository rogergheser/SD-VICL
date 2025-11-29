"""
Hydra-based Experiment Runner for SD-VICL

This module provides a Hydra-powered entry point for running visual in-context learning
experiments. The model learns from context image pairs (input + target) to generate
the expected output for a query image.

Usage:
    # Run with default config
    python run_experiment.py
    
    # Override parameters
    python run_experiment.py context.query=path/to/query.png experiment.seed=42
    
    # Specify context pairs and query
    python run_experiment.py \\
        context.pairs.0.input=img1.png context.pairs.0.target=mask1.png \\
        context.pairs.1.input=img2.png context.pairs.1.target=mask2.png \\
        context.query=query.png
    
    # Use a different config file
    python run_experiment.py --config-name=segmentation_experiment
    
    # Multi-run with parameter sweeps
    python run_experiment.py -m experiment.seed=1,2,3,4,5
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a visual in-context learning experiment using Hydra configuration.
    
    The experiment takes context image pairs (input + target) and a query image,
    then generates the expected output for the query based on the learned pattern.
    
    Args:
        cfg: Hydra configuration object containing all experiment parameters
    """
    # Print configuration for experiment tracking
    print("=" * 50)
    print("SD-VICL Experiment Configuration:")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)
    
    # Extract configuration values
    model_id = cfg.model.id
    scheduler_type = cfg.model.scheduler
    device = cfg.hardware.device
    output_path = cfg.output.path
    
    # Load context pairs
    context_pairs = []
    for pair in cfg.context.pairs:
        if pair.input is not None and pair.target is not None:
            context_pairs.append({
                'input': pair.input,
                'target': pair.target
            })
    
    query_image = cfg.context.query
    
    # Validate configuration
    if len(context_pairs) == 0:
        print("Warning: No valid context pairs provided. At least one pair is recommended.")
    
    if query_image is None:
        print("Warning: No query image specified. Set context.query to run inference.")
    
    print(f"Model: {model_id}")
    print(f"Context pairs: {len(context_pairs)}")
    print(f"Query image: {query_image}")
    
    # Placeholder for in-context learning pipeline
    # TODO: Implement the visual in-context learning pipeline
    # This will involve:
    # 1. Loading and encoding context image pairs
    # 2. Loading the query image
    # 3. Running the diffusion process with context conditioning
    # 4. Generating the output based on learned pattern
    
    print("=" * 50)
    print("Note: In-context learning pipeline implementation pending.")
    print("Configuration loaded successfully for:")
    print(f"  - {len(context_pairs)} context pair(s)")
    print(f"  - Query: {query_image}")
    print(f"  - Output: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_experiment()
