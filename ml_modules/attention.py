from diffusers.models.attention_processor import AttnProcessor, Attention
import torch


# class SD_VICL_Attention(Attention):
#     def __init__(self, temperature: float, contrast_strength: float, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.temperature = temperature
#         self.contrast_strength = contrast_strength
#         self.set_processor(
#             AttnProcessor()
#         )
class SD_VICL_AttnProcessor(AttnProcessor):
    def __init__(self, temperature: float, contrast_strength: float):
        super().__init__()
        self.temperature = temperature
        self.contrast_strength = contrast_strength

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask,  # type: ignore[arg-type]
            sequence_length,
            batch_size,
        )  # type: ignore[assignment]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(attn, query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def get_attention_scores(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        default_attention_scores = attn.get_attention_scores(
            query, key, attention_mask=attention_mask
        )
        attention_probs = self.apply_attention_map_contrasting_and_temperature(
            default_attention_scores,
        )

        return attention_probs

    def apply_attention_map_contrasting_and_temperature(self, default_attention_scores):
        """The purpose of this function is to apply the attention map contrasting described
        in the paper. To apply this we recontrast"""
        default_attention_scores = default_attention_scores
        x = default_attention_scores[3, ...]  # We take the D value
        mu = x.mean()
        default_attention_scores[3, ...] = mu + self.contrast_strength * (x - mu)

        return default_attention_scores
