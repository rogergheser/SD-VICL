# SD-VICL
Repo to reproduce the work in https://arxiv.org/pdf/2508.09949 from scratch.
The authors never shared their code so this work is mine and took inspiration from (Cross-Image Attention)[https://github.com/garibida/cross-image-attention] for the AdaIn



## Installation

```bash
pip install -r requirements.txt
```

## Dataset specifications
This approach has not been properly tested with a dataset and only a qualitative analysis of results has been done.
Some test were done on a dataset from the ICM-57 dataset from In-Context Matting [paper](https://arxiv.org/abs/2403.15789) and hence the dataloader is configured to work with this.
For info the structure to follow to be compliant with this repo is the following:
- data
  - category
        - alphas
            # gt masks go here as .png
        - images
            # samples go here as .jpg

### Usage
To execute the main generation scripts:
```python
python generate.py # for full dataset with default at full resolution (512x512)
python debug.py # for single sample test on 256x256 resolution resized image
```

You may specify specific parameters, see how:
```python
python generate.py --help
```

## Architecture

The codebase is designed with modularity in mind to allow easy modification of U-Net attention mechanisms:

### Components

- **`ModelLoader`**: Handles loading of model components (VAE, U-Net, Text Encoder, Scheduler)
- **`TextEncoder`**: Wraps text encoding for conditioning
- **`LatentProcessor`**: Handles latent space operations (encode/decode)
- **`UNetWrapper`**: Wraps U-Net for easy attention modification
- **`DiffusionPipeline`**: Main pipeline orchestrating generation

