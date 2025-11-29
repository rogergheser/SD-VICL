# SD-VICL
Repo to reproduce the work in https://arxiv.org/pdf/2508.09949

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Hydra-based Experiments (Recommended)

Use Hydra for reproducible in-context learning experiment configuration:

```bash
# Run with default config
python run_experiment.py

# Specify context pairs and query image for in-context learning
python run_experiment.py \
    context.pairs.0.input=image1.png context.pairs.0.target=mask1.png \
    context.pairs.1.input=image2.png context.pairs.1.target=mask2.png \
    context.query=query.png

# Override experiment parameters
python run_experiment.py experiment.seed=42 model.scheduler=ddim

# Multi-run with parameter sweeps
python run_experiment.py -m experiment.seed=1,2,3,4,5
```

Configuration is defined in `conf/config.yaml`. Create custom config files in the `conf/` directory for different experiments (e.g., segmentation, depth estimation, etc.).

### Python API

```python
from generate import create_pipeline, GenerationConfig

# Create pipeline
pipeline = create_pipeline(model_id="stabilityai/stable-diffusion-2-1-base")

# Configure generation
config = GenerationConfig(
    prompt="a photograph of an astronaut riding a horse",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# Generate images
images = pipeline.generate(config)
images[0].save("output.png")
```

## Architecture

The codebase is designed with modularity in mind to allow easy modification of U-Net attention mechanisms:

### Components

- **`ModelLoader`**: Handles loading of model components (VAE, U-Net, Text Encoder, Scheduler)
- **`TextEncoder`**: Wraps text encoding for conditioning
- **`LatentProcessor`**: Handles latent space operations (encode/decode)
- **`UNetWrapper`**: Wraps U-Net for easy attention modification
- **`DiffusionPipeline`**: Main pipeline orchestrating generation

### Modifying U-Net Attention

The `UNetWrapper` class is designed for extending with custom attention mechanisms:

```python
from generate import UNetWrapper, create_pipeline

class CustomUNetWrapper(UNetWrapper):
    def forward(self, latents, timestep, encoder_hidden_states, **kwargs):
        # Custom attention logic here
        # Access attention layers via self.get_attention_layers()
        return super().forward(latents, timestep, encoder_hidden_states, **kwargs)

# Use custom wrapper
pipeline = create_pipeline()
pipeline.unet_wrapper = CustomUNetWrapper(
    pipeline.unet_wrapper.unet,
    pipeline.device,
    pipeline.dtype
)
```
