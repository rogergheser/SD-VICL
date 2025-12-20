import torch
from PIL import Image
from diffusers.models.attention import Attention
from diffusers.models.unets import UNet2DConditionModel
from models.attention import SD_VICL_AttnProcessor


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
        
        image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=self.dtype) # type: ignore
        
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
        dtype: torch.dtype = torch.float16
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
        self._attention_hooks = {}
        self.temperature = attn_temperature
        self.contrast_strength = attn_contrast_strength
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
            def replace_attention(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                    assert isinstance(inp, torch.Tensor)  # stack to single tensor for easier indexing
                print(f"{full_child_name}: {inp.size(1)}")
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
                    elif which == '':
                        pass
                    else:
                        raise ValueError(f"Unknown which value: {which} found at {full_child_name}")
                if isinstance(out, torch.Tensor):
                    # custom attention logic here
                    out = out  # modify as needed
                else:
                    raise ValueError(f"Expected tensor output from attention module {full_child_name}")
                return out
            return replace_attention
        
        skip_list = []
        for parent_name, attn_module in attn_layers:
            if hasattr(attn_module, 'is_cross_attention') and attn_module.is_cross_attention:
                # skip cross-attention for now
                skip_list.append(parent_name)
                continue
            if isinstance(attn_module, Attention):
                attn_module.set_processor(SD_VICL_AttnProcessor(temperature=self.temperature, contrast_strength=self.contrast_strength))
                print(f"Set SD_VICL_AttnProcessor for {parent_name}")
            skip = False
            for layer in skip_list:
                if parent_name.startswith(layer):
                    # print(f"Skipping attention module {parent_name} due to parent skip")
                    skip = True
                    break
            if skip:
                continue
            # iterate child modules to find to_q/to_k/to_v (attribute names vary by implementation)
            for child_name, child_mod in attn_module.named_modules():
                if isinstance(child_mod, Attention):
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'q'))
                    self._attention_hooks[full_child_name] = handle
                elif not isinstance(child_mod, torch.nn.Linear):
                    continue
                if child_name.lower().endswith('to_q'):
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'q'))
                    self._attention_hooks[full_child_name] = handle
                    # ATTN_OUTPUTS[full_child_name]  # ensure key exists
                elif child_name.lower().endswith('to_k'):
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'k'))
                    self._attention_hooks[full_child_name] = handle
                    # ATTN_OUTPUTS[full_child_name]  # ensure key exists
                elif child_name.lower().endswith('to_v'):
                    full_child_name = f"{parent_name}.{child_name}"
                    handle = child_mod.register_forward_hook(make_hook(full_child_name, 'v'))
                    self._attention_hooks[full_child_name] = handle
                    # ATTN_OUTPUTS[full_child_name]
                elif child_name.lower().endswith(''):
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
        current_timestep: int,
        total_timesteps: int,
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
        B, _, _, _ = latents.shape
        # kwargs['attention_scale'] = kwargs.pop('attention_temperature', 0.4)
        assert B == 8, "Batch size must be 8"
        # Forward pass with current forward hooks to extract QKV values
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        ).sample
    
        return noise_pred