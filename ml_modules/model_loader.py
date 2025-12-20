import torch


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
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=dtype, device_map=device
            )
        return self._vae

    def load_unet(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        """
        Load the U-Net model.

        This is the component to modify for custom attention mechanisms.
        The attention layers are accessible via unet.down_blocks, unet.mid_block, and unet.up_blocks
        """
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

        if self._unet is None:
            self._unet = UNet2DConditionModel.from_pretrained(
                self.model_id, subfolder="unet", torch_dtype=dtype, device_map=device
            )
        return self._unet

    def load_text_encoder(
        self, device: str = "cuda", dtype: torch.dtype = torch.float16
    ):
        """Load the text encoder (CLIP)."""
        from transformers import CLIPTextModel

        if self._text_encoder is None:
            self._text_encoder = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=dtype,
                device_map=device,
            )
        return self._text_encoder

    def load_tokenizer(self):
        """Load the tokenizer."""
        from transformers import CLIPTokenizer

        if self._tokenizer is None:
            self._tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
        return self._tokenizer

    def load_scheduler(self, scheduler_type: str = "ddpm"):
        """
        Load the noise scheduler.

        Args:
            scheduler_type: Type of scheduler ('ddpm', 'ddim', 'euler', 'pndm')
        """
        import diffusers.schedulers as sched

        scheduler_map = {
            "ddpm": sched.scheduling_ddpm.DDPMScheduler,
            "ddim": sched.scheduling_ddim.DDIMScheduler,
            "euler": sched.scheduling_euler_discrete.EulerDiscreteScheduler,
            "pndm": sched.scheduling_pndm.PNDMScheduler,
        }

        if scheduler_type not in scheduler_map:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_type}. Choose from {list(scheduler_map.keys())}"
            )

        self._scheduler = scheduler_map[scheduler_type].from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        return self._scheduler

    def load_all(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        scheduler_type: str = "ddpm",
    ):
        """Load all model components."""
        return {
            "vae": self.load_vae(device, dtype),
            "unet": self.load_unet(device, dtype),
            "text_encoder": self.load_text_encoder(device, dtype),
            "tokenizer": self.load_tokenizer(),
            "scheduler": self.load_scheduler(scheduler_type),
        }
