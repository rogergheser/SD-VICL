import torch


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
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings
