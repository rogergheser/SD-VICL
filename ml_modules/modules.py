from typing import Any
import torch
from PIL import Image
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ml_modules.attention import SD_VICL_AttnProcessor


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

        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

        image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)  # type: ignore

        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents: torch.Tensor) -> list[Image.Image]:
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

    def encode(self, prompt: str | list[str], max_length: int = 77) -> torch.Tensor:
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
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]

        return text_embeddings


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

    def __init__(
        self,
        unet: UNet2DConditionModel,
        attn_temperature: float = 0.4,
        attn_contrast_strength: float = 1.67,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
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
        self._attention_hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.temperature = attn_temperature
        self.contrast_strength = attn_contrast_strength
        self.attention_hooks_info = self.register_attention_hook()
        with open("attention_hooks_info.json", "w") as f:
            import json

            json.dump(self.attention_hooks_info, f, indent=4)

    def register_attention_hook(self) -> dict[str, Any]:
        """
        Register forward hooks on the Q/K/V linear submodules inside attention blocks.
        After calling this, ATTN_OUTPUTS will have entries like:
        ATTN_OUTPUTS['unet.down_blocks.3.attn_1.to_q']['outputs'] = [tensor, ...]
        """
        # clear existing hooks first
        self.clear_attention_hooks()

        attn_layers = self.get_attention_layers()  # list of (full_name, module)

        def make_hook(full_child_name, which):
            def replace_attention(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                    assert isinstance(
                        inp, torch.Tensor
                    )  # stack to single tensor for easier indexing
                # 0->Q, 1->K, 2->V
                B = inp.shape[0]
                assert B % 4 == 0, "Batch must contain ABCD in order"

                # We only apply to the first 4, the rest will remain normal for swap-guidance
                group = 0
                Q_idx = group * 4 + 0  # query
                K_idx = group * 4 + 1  # key
                V_idx = group * 4 + 2  # value
                D_idx = group * 4 + 3  # to modify

                if which == "q":
                    out[D_idx] = out[Q_idx]
                elif which == "k":
                    out[D_idx] = out[K_idx]
                elif which == "v":
                    out[D_idx] = out[V_idx]
                elif which == "":
                    pass
                else:
                    raise ValueError(
                        f"Unknown which value: {which} found at {full_child_name}"
                    )
                if isinstance(out, torch.Tensor):
                    # custom attention logic here
                    out = out  # modify as needed
                else:
                    raise ValueError(
                        f"Expected tensor output from attention module {full_child_name}"
                    )
                return out

            return replace_attention

        skip_list = []
        skip_modules = []
        hooked_layers = []
        hooked_modules = []
        for parent_name, attn_module in attn_layers:
            if (
                hasattr(attn_module, "is_cross_attention")
                and attn_module.is_cross_attention
            ):
                # skip cross-attention for now
                skip_list.append(parent_name)
                skip_modules.append(attn_module.__class__.__name__)
                continue
            if isinstance(attn_module, Attention):
                attn_module.set_processor(
                    SD_VICL_AttnProcessor(
                        temperature=self.temperature,
                        contrast_strength=self.contrast_strength,
                    )
                )
                hooked_layers.append(parent_name)
                hooked_modules.append(attn_module.__class__.__name__)

        return {
            "hooked_layers": hooked_layers,
            "hooked_modules": hooked_modules,
            "skipped_layers": skip_list,
            "skipped_modules": skip_modules,
        }

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
            if "attn" in name.lower():
                attention_layers.append((name, module))

        return attention_layers

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        warmup: bool = False,
        **kwargs,
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
        B, _, _, _ = latents.shape
        assert B == 8, "Batch size must be 8"
        # Forward pass with current forward hooks to extract QKV values
        noise_pred = self.unet(
            latents, timestep, encoder_hidden_states, **kwargs
        ).sample

        return noise_pred
