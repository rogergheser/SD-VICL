from dataclasses import dataclass
import torch
from typing import Callable
from models.ldm_model_loader import ModelLoader

from models.ldm_modules import LatentProcessor, UNetWrapper, TextEncoder
from PIL import Image

from utils import DEVICE, DTYPE, SEED, adain


@dataclass
class GenerationConfig:
    """Configuration for image generation."""

    prompt: str
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    num_inference_steps: int = 70
    guidance_scale: float = 7.5
    contrast_strength: float = 1.67  # beta
    attention_temperature: float = 0.4
    swap_guidance: float = 3.5  # gamma
    num_images: int = 1
    seed: int = SEED
    device: str = DEVICE
    dtype: torch.dtype = DTYPE


class DiffusionPipeline:
    """
    Main pipeline for latent diffusion image generation.

    This modular pipeline allows easy customization of individual components.
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        config: GenerationConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        scheduler_type: str = "ddpm",
    ):
        """
        Initialize the diffusion pipeline.

        Args:
            model_loader: ModelLoader instance
            device: Device to run on
            dtype: Data type
            scheduler_type: Type of scheduler to use
        """
        self.device = device
        self.dtype = dtype

        # Load components
        components = model_loader.load_all(device, dtype, scheduler_type)

        self.text_encoder_wrapper = TextEncoder(
            components["tokenizer"], components["text_encoder"], device
        )
        self.latent_processor = LatentProcessor(components["vae"], device, dtype)
        self.unet_wrapper = UNetWrapper(
            components["unet"],
            attn_temperature=config.attention_temperature,  # ?
            device=device,
            dtype=dtype,
        )
        self.scheduler = components["scheduler"]

    @torch.no_grad()
    def generate(
        self,
        samples: list[Image.Image],
        config: GenerationConfig,
        callback: Callable | None = None,
    ) -> list[Image.Image]:
        """
        Generate images from text prompt.

        Args:
            config: Generation configuration
            callback: Optional callback function called at each step

        Returns:
            List of generated PIL Images
        """
        # Encode text prompt
        text_embeddings = self.text_encoder_wrapper.encode([config.prompt] * 4)

        # Handle classifier-free guidance
        if config.guidance_scale > 1.0:
            uncond_embeddings = self.text_encoder_wrapper.encode(
                config.negative_prompt if config.negative_prompt else ""
            )
            uncond_embeddings = uncond_embeddings.repeat(config.num_images * 4, 1, 1)
            # Concatenate: [uncond_batch, cond_batch]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # We use C to create the latent representation for D as they should be structurally similar
        # Get initial latents
        _latents = [self.latent_processor.encode_image(img) for img in samples]
        _latents = _latents + [_latents[0].detach().clone()]
        latents = torch.cat(_latents, dim=0)
        assert latents.shape[0] == 4, "Latent batch size mismatch"

        # Set up scheduler
        self.scheduler.set_timesteps(config.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            if config.guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.unet_wrapper.forward(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                current_timestep=i,
                total_timesteps=config.num_inference_steps,
                # attention_temperature=config.attention_temperature
            )

            # Apply classifier-free guidance
            noise_cond, noise_uncond = noise_pred.chunk(2)
            reweighting_factor = (
                config.swap_guidance
                * (config.num_inference_steps - i)
                / config.num_inference_steps
            )
            noise_pred = noise_uncond + reweighting_factor * (noise_cond - noise_uncond)

            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            d_idx = 3
            v_idx = 2
            latents[d_idx, ...] = adain(latents[d_idx, ...], latents[v_idx, ...])

            # Call callback if provided
            if callback is not None:
                callback(i, t, latents)

        # Decode to images
        images = self.latent_processor.decode_latents(latents)

        return images


def create_pipeline(
    config: GenerationConfig,
    model_id: str = "sd-legacy/stable-diffusion-v1-5",
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
    scheduler_type: str = "ddpm",
) -> DiffusionPipeline:
    """
    Create a diffusion pipeline with default settings.

    Args:
        model_id: HuggingFace model ID or local path
        device: Device to run on (auto-detected if None)
        dtype: Data type (auto-detected if None)
        scheduler_type: Type of scheduler to use

    Returns:
        Configured DiffusionPipeline
    """
    model_loader = ModelLoader(model_id)
    pipeline = DiffusionPipeline(model_loader, config, device, dtype, scheduler_type)

    return pipeline
