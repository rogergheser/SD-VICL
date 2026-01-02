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
        assert attn.to_k is not None
        assert attn.to_v is not None
        assert attn.to_out is not None
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        query, key, value = self.adjust_d_qkv(query, key, value, attn.heads)

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

    def adjust_d_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adjust the D by integrating information from self attention of A, B, C samples according
        to the paper implementation.
        Sample image is Q
        Guidance image is K
        Guidance ground truth image is V
        Target image to modify is D
        """
        Q_slice = slice(heads * 0, heads * 1)  # query
        K_slice = slice(heads * 1, heads * 2)  # key
        V_slice = slice(heads * 2, heads * 3)  # value
        D_slice = slice(heads * 3, heads * 4)  # to-modify

        query[D_slice] = query[Q_slice].clone()
        key[D_slice] = key[K_slice].clone()
        value[D_slice] = value[V_slice].clone()
        return query, key, value

    def get_attention_scores(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention scores with attention map contrasting for D sample."""
        default_attention_scores = self.standard_attention_scores(
            query,
            key,
            attention_mask=attention_mask,
            temperature=self.temperature,
            heads=attn.heads,
        )
        D_slice = slice(attn.heads * 3, attn.heads * 4)
        default_attention_scores[D_slice] = torch.stack(
            [
                self.apply_attention_map_contrasting(
                    default_attention_scores[head_idx],
                )
                for head_idx in range(D_slice.start, D_slice.stop)
            ]
        )
        return default_attention_scores

    def standard_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        heads: int,
    ) -> torch.Tensor:
        """Compute standard attention scores without any modification."""

        # Not doing multihead attention
        dim = query.shape[-1]
        attention_scores = torch.bmm(query, key.transpose(-1, -2))
        attention_scores = attention_scores * (dim**-0.5) / temperature

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        return attention_scores.softmax(dim=-1)

    def apply_attention_map_contrasting(self, x: torch.Tensor) -> torch.Tensor:
        """The purpose of this function is to apply the attention map contrasting described
        in the paper. To apply this we recontrast"""
        mu = x.mean(dim=-1, keepdim=True)
        x = mu + self.contrast_strength * (x - mu)

        return x
