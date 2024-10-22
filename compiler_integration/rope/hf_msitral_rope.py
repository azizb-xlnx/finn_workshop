# Description: This file contains the MistralRotaryEmbedding class which is used to create the rotary embeddings for the Mistral model.
# The code is copied from the Hugging Face repository and modified to work with the Mistral model.
# The original code can be found at: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L81-L107

import torch
import torch.nn as nn
def export_to_onnx(model, file_path, input_shape=(1, 12, 2048, 64)):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    position_ids = torch.arange(input_shape[2]).expand((input_shape[0], input_shape[2]))
    torch.onnx.export(
        model,
        (dummy_input, position_ids),
        file_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['dummy_input', 'position_ids'],
        output_names=['cos', 'sin'],
        dynamic_axes={
            'dummy_input': {0: 'batch_size', 2: 'seq_len'},
            'position_ids': {0: 'batch_size', 1: 'seq_len'},
            'cos': {0: 'batch_size', 2: 'seq_len'},
            'sin': {0: 'batch_size', 2: 'seq_len'}
        }
    )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def main():
        
    # Example usage:
    model_orig = MistralRotaryEmbedding(dim=64)
    model_orig.eval()

    # Generate random inputs
    batch_size = 1
    num_attention_heads = 12
    seq_len = 2048
    head_size = 64

    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, num_attention_heads, seq_len, head_size)
    position_ids = torch.arange(seq_len).expand((batch_size, seq_len))
    #random q and k valuyes
    query_states = torch.randn(batch_size, num_attention_heads, seq_len, head_size)
    key_states = torch.randn(batch_size, num_attention_heads, seq_len, head_size)

    # Run the model_orig
    cos, sin = model_orig(dummy_input, position_ids)

    # apply the rotary position embedding
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    print("Cosine output shape:", cos.shape)
    print("Sine output shape:", sin.shape)

    print("Query states shape:", query_states.shape)
    print("Key states shape:", key_states.shape)

    # Example usage:
    model = MistralRotaryEmbedding(dim=64)
    export_to_onnx(model, "mistral_rope.onnx")

if __name__ == "__main__":
    main()