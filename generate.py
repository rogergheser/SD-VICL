"""
Latent Diffusion Image Generation Script

This module provides modular building blocks for latent diffusion image generation.
The architecture is designed to allow easy modification of U-Net attention mechanisms.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, List, Callable
import torch
from PIL import Image

ATTN_OUTPUTS = defaultdict(lambda: defaultdict(list))
@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    prompt: str
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32


class ModelLoader:
    """Handles loading of diffusion model components."""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1-base"):
        """
        Initialize the model loader.
        
        Args:
            model_id: HuggingFace model ID or local path
        """
        self.model_id = model_id
        self._vae = None
        self._unet = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
    
    def load_vae(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """Load the Variational Autoencoder."""
        from diffusers import AutoencoderKL
        
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=dtype
            ).to(device)
        return self._vae
    
    def load_unet(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """
        Load the U-Net model.
        
        This is the component to modify for custom attention mechanisms.
        The attention layers are accessible via unet.down_blocks, unet.mid_block, and unet.up_blocks
        """
        from diffusers import UNet2DConditionModel
        
        if self._unet is None:
            self._unet = UNet2DConditionModel.from_pretrained(
                self.model_id,
                subfolder="unet",
                torch_dtype=dtype
            ).to(device)
        return self._unet
    
    def load_text_encoder(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """Load the text encoder (CLIP)."""
        from transformers import CLIPTextModel
        
        if self._text_encoder is None:
            self._text_encoder = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=dtype
            ).to(device)
        return self._text_encoder
    
    def load_tokenizer(self):
        """Load the tokenizer."""
        from transformers import CLIPTokenizer
        
        if self._tokenizer is None:
            self._tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id,
                subfolder="tokenizer"
            )
        return self._tokenizer
    
    def load_scheduler(self, scheduler_type: str = "ddpm"):
        """
        Load the noise scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('ddpm', 'ddim', 'euler', 'pndm')
        """
        from diffusers import (
            DDPMScheduler,
            DDIMScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler
        )
        
        scheduler_map = {
            "ddpm": DDPMScheduler,
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "pndm": PNDMScheduler,
        }
        
        if scheduler_type not in scheduler_map:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from {list(scheduler_map.keys())}")
        
        self._scheduler = scheduler_map[scheduler_type].from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        return self._scheduler
    
    def load_all(self, device: str = "cuda", dtype: torch.dtype = torch.float16, scheduler_type: str = "ddpm"):
        """Load all model components."""
        return {
            "vae": self.load_vae(device, dtype),
            "unet": self.load_unet(device, dtype),
            "text_encoder": self.load_text_encoder(device, dtype),
            "tokenizer": self.load_tokenizer(),
            "scheduler": self.load_scheduler(scheduler_type),
        }


class TextEncoder:
    """Handles text encoding for conditioning."""
    
    def __init__(self, tokenizer, text_encoder, device: str = "cuda"):
        """
        Initialize the text encoder wrapper.
        
        Args:
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder model
            device: Device to run on
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
    
    def encode(self, prompt: Union[str, List[str]], max_length: int = 77) -> torch.Tensor:
        """
        Encode text prompt(s) to embeddings.
        
        Args:
            prompt: Text prompt or list of prompts
            max_length: Maximum token length
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings


class LatentProcessor:
    """Handles latent space operations."""
    
    def __init__(self, vae, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """
        Initialize the latent processor.
        
        Args:
            vae: Variational Autoencoder model
            device: Device to run on
            dtype: Data type
        """
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # Latent channels from VAE config (typically 4 for SD models)
        self.latent_channels = self.vae.config.latent_channels
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode an image to latent space."""
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Decode latents to images.
        
        Args:
            latents: Latent tensor
            
        Returns:
            List of PIL Images
        """
        latents = latents / self.vae.config.scaling_factor
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        images = []
        for img in image:
            img = (img * 255).round().astype("uint8")
            images.append(Image.fromarray(img))
        
        return images
    
    def get_initial_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate initial random latents.
        
        Args:
            batch_size: Number of images to generate
            height: Image height
            width: Image width
            generator: Random generator for reproducibility
            
        Returns:
            Initial latent tensor
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        
        latents = torch.randn(
            (batch_size, self.latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        return latents


class UNetWrapper:
    """
    Wrapper for U-Net that allows easy modification of attention mechanisms.
    
    This is the main class to extend for custom attention passes.
    Override the `forward` method or use hooks to modify attention behavior.
    
    Note: The attention hook infrastructure is provided as a foundation for
    future custom attention implementations. To implement custom attention:
    1. Subclass UNetWrapper
    2. Override the `forward` method
    3. Use `get_attention_layers()` to access attention modules
    """
    
    def __init__(self, unet, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """
        Initialize the U-Net wrapper.
        
        Args:
            unet: U-Net model
            device: Device to run on
            dtype: Data type
        """
        self.unet = unet
        self.device = device
        self.dtype = dtype
        self._attention_hooks = {}
        self.register_attention_hook()
    
    def register_attention_hook(self) -> None:
        """
        Register forward hooks on the Q/K/V linear submodules inside attention blocks.
        After calling this, ATTN_OUTPUTS will have entries like:
        ATTN_OUTPUTS['unet.down_blocks.3.attn_1.to_q']['outputs'] = [tensor, ...]
        """
        # clear existing hooks first
        self.clear_attention_hooks()

        attn_layers = self.get_attention_layers()  # list of (full_name, module)

        def make_hook(full_child_name, which):
            # TODO change hook so that it keeps a counter to understand which sample it is processing and can decide what to save accordingly
            # returns a hook that stores the output tensor
            def hook(module, inp, out):
                # out may be a tensor or tuple - keep it simple and store tensor(s)
                # detach to avoid grads and optionally move to cpu to avoid GPU memory growth
                if isinstance(out, torch.Tensor):
                    saved = out.detach().cpu()  # remove .cpu() if you want to keep cuda tensors
                else:
                    # tuple/list of tensors
                    saved = tuple(o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in out)
                ATTN_OUTPUTS[full_child_name][which].append(saved)
            return hook

        for parent_name, attn_module in attn_layers:
            # iterate child modules to find to_q/to_k/to_v (attribute names vary by implementation)
            for child_name, child_mod in attn_module.named_modules():
                lname = child_name.lower()
                if 'to_q' in lname or child_name.lower().endswith('.q') or lname == 'q':
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'q'))
                    self._attention_hooks[full_child_name] = handle
                    ATTN_OUTPUTS[full_child_name]  # ensure key exists
                elif 'to_k' in lname or child_name.lower().endswith('.k') or lname == 'k':
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'k'))
                    self._attention_hooks[full_child_name] = handle
                    ATTN_OUTPUTS[full_child_name]
                elif 'to_v' in lname or child_name.lower().endswith('.v') or lname == 'v':
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'v'))
                    self._attention_hooks[full_child_name] = handle
                    ATTN_OUTPUTS[full_child_name]
                elif '' in lname:
                    # This is the attention module, we will work with this later
                    continue

        # fallback: if nothing matched (different naming), scan immediate attributes on attn_module
        if len(self._attention_hooks) == 0:
            for parent_name, attn_module in attn_layers:
                for attr in ('to_q', 'to_k', 'to_v', 'q_proj', 'k_proj', 'v_proj'):
                    if hasattr(attn_module, attr):
                        child_mod = getattr(attn_module, attr)
                        full_child_name = f"{parent_name}.{attr}"
                        handle = child_mod.register_forward_hook(make_hook(full_child_name, attr[-1]))  # 'q','k','v'
                        self._attention_hooks[full_child_name] = handle
                        ATTN_OUTPUTS[full_child_name]

    def clear_attention_hooks(self) -> None:
        """Clear all registered attention hooks."""
        for handle in list(self._attention_hooks.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._attention_hooks.clear()

    def get_attention_layers(self):
        """
        Get all attention layers in the U-Net.
        
        Returns a list of attention modules that can be modified.
        """
        attention_layers = []
        
        for name, module in self.unet.named_modules():
            if 'attn' in name.lower():
                attention_layers.append((name, module))
        
        return attention_layers
    
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        This method can be overridden to implement custom attention behavior.
        
        Args:
            latents: Noisy latent tensor
            timestep: Current timestep
            encoder_hidden_states: Text embeddings
            
        Returns:
            Predicted noise
        """
        dict_map = "ABCD"
        self_attn_outputs = {}
        for i in range(3):
            # Forward pass with current forward hooks to extract QKV values
            noise_pred = self.unet(
                latents[i],
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            ).sample

            self_attn_outputs[dict_map[i]] = ATTN_OUTPUTS.copy()
            # Clear ATTN_OUTPUTS for next pass
            for key in ATTN_OUTPUTS.keys():
                ATTN_OUTPUTS[key]['q'].clear()
                ATTN_OUTPUTS[key]['k'].clear()
                ATTN_OUTPUTS[key]['v'].clear()
        

        # TODO Register different forward hook to adapt the custom attention mechanism for D
        # The forward hook will take care of modifying the inputs to the attention mechanisms
        noise_pred = self.unet(
                latents[-1],
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            ).sample

        return noise_pred


class DiffusionPipeline:
    """
    Main pipeline for latent diffusion image generation.
    
    This modular pipeline allows easy customization of individual components.
    """
    
    def __init__(
        self,
        model_loader: ModelLoader,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        scheduler_type: str = "ddpm"
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
            components["tokenizer"],
            components["text_encoder"],
            device
        )
        self.latent_processor = LatentProcessor(
            components["vae"],
            device,
            dtype
        )
        self.unet_wrapper = UNetWrapper(
            components["unet"],
            device,
            dtype
        )
        self.scheduler = components["scheduler"]
    
    @torch.no_grad()
    def generate(self, config: GenerationConfig, callback: Optional[Callable] = None) -> List[Image.Image]:
        """
        Generate images from text prompt.
        
        Args:
            config: Generation configuration
            callback: Optional callback function called at each step
            
        Returns:
            List of generated PIL Images
        """
        # Set up generator for reproducibility
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
        
        # Encode text prompt
        text_embeddings = self.text_encoder_wrapper.encode(config.prompt)
        
        # Handle classifier-free guidance - duplicate embeddings for batch
        if config.num_images > 1:
            text_embeddings = text_embeddings.repeat(config.num_images, 1, 1)
        
        # Handle classifier-free guidance
        if config.guidance_scale > 1.0:
            uncond_embeddings = self.text_encoder_wrapper.encode(
                config.negative_prompt if config.negative_prompt else ""
            )
            if config.num_images > 1:
                uncond_embeddings = uncond_embeddings.repeat(config.num_images, 1, 1)
            # Concatenate: [uncond_batch, cond_batch]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Get initial latents
        latents = self.latent_processor.get_initial_latents(
            config.num_images,
            config.height,
            config.width,
            generator
        )
        
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
                encoder_hidden_states=text_embeddings
            )
            
            # Apply classifier-free guidance
            if config.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Call callback if provided
            if callback is not None:
                callback(i, t, latents)
        
        # Decode to images
        images = self.latent_processor.decode_latents(latents)
        
        return images


def create_pipeline(
    model_id: str = "stabilityai/stable-diffusion-2-1-base",
    device: str = None,
    dtype: torch.dtype = None,
    scheduler_type: str = "ddpm"
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_loader = ModelLoader(model_id)
    pipeline = DiffusionPipeline(model_loader, device, dtype, scheduler_type)
    
    return pipeline


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using latent diffusion models")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--model", type=str, default="sd-legacy/stable-diffusion-v1-5", help="Model ID")
    # parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3-medium_amdgpu", help="Model ID")
    # parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1-base", help="Model ID")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
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
    pipeline = create_pipeline(
        model_id=args.model,
        device=device,
        scheduler_type=args.scheduler
    )
    
    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images,
        seed=args.seed
    )
    
    print(f"Generating image with prompt: {args.prompt}")
    
    def progress_callback(step, timestep, latents):
        print(f"Step {step + 1}/{args.steps}")
    
    images = pipeline.generate(config, callback=progress_callback)
    
    # Save images
    if len(images) == 1:
        images[0].save(args.output)
        print(f"Saved image to {args.output}")
    else:
        base_name = args.output.rsplit(".", 1)
        for i, img in enumerate(images):
            if len(base_name) > 1:
                output_path = f"{base_name[0]}_{i}.{base_name[1]}"
            else:
                output_path = f"{args.output}_{i}"
            img.save(output_path)
            print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
