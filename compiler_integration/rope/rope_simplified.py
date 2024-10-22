# Description: This file contains the MistralRotaryEmbedding class which is used to create the rotary embeddings for the Mistral model.
# The code is copied from the Hugging Face repository and modified to work with the Mistral model.
# The original code can be found at: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L81-L107

import torch
import torch.nn as nn
from hf_msitral_rope import MistralRotaryEmbedding
from hf_msitral_rope import apply_rotary_pos_emb as orig_apply_rotary_pos_emb

import onnxruntime as ort

def export_to_onnx(model, file_path, input_shape=(1, 12, 2048, 64)):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    position_ids = torch.arange(input_shape[2]).expand((input_shape[0], input_shape[2])).float()
    torch.onnx.export(
        model,
        (position_ids),
        file_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[ 'position_ids'],
        output_names=['cos', 'sin'],
        dynamic_axes={
            'position_ids': {0: 'batch_size', 1: 'seq_len'},
            'cos': {0: 'batch_size', 2: 'seq_len'},
            'sin': {0: 'batch_size', 2: 'seq_len'}
        }
    )
def rotate_half(x):
    """Rotates half the hidden dims of the input using transposes."""
    x1, x2 = x.chunk(2, dim=-1)
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


class MistralRotaryEmbeddingSimple(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        self.inv_freq = self.inv_freq[None, :, None] 
    
    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # ## Reshape the inv_freq tensor to match the shape of the position_ids tensor
        inv_freq_expanded = self.inv_freq 
        # Use broadcasting to match the shape instead of the expand function
        position_ids_expanded = position_ids[:, None, :] 

        # Calculate the frequency embeddings
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    

def main():

    # Validate that the simplification did not change the output
    model_orig = MistralRotaryEmbedding(dim=64)
    model_orig.eval()

    model_simple = MistralRotaryEmbeddingSimple(dim=64)
    model_simple.eval()

    # Generate random inputs
    batch_size = 1
    num_attention_heads = 12
    seq_len = 2048
    head_size = 64

    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, num_attention_heads, seq_len, head_size)
    position_ids = torch.arange(seq_len).expand((batch_size, seq_len)).float()

    query_states = torch.randn(batch_size, num_attention_heads, seq_len, head_size)
    key_states = torch.randn(batch_size, num_attention_heads, seq_len, head_size)

    # Run models
    cos_orig, sin_orig = model_orig(dummy_input, position_ids)
    cos_simple, sin_simple = model_simple( position_ids)

    # apply the rotary position embedding to output of both models
    q_state_orig, k_state_orig = orig_apply_rotary_pos_emb(query_states, key_states, cos_orig, sin_orig)
    q_state_simple, k_state_simple = apply_rotary_pos_emb(query_states, key_states, cos_simple, sin_simple)

    # Check if the outputs are equal
    print("Cosine output is equal:", torch.equal(cos_orig, cos_simple))
    print("Sine output is equal:", torch.equal(sin_orig, sin_simple))

    print("Query state output is equal:", torch.equal(q_state_orig, q_state_simple))
    print("Key state output is equal:", torch.equal(k_state_orig, k_state_simple))

    export_to_onnx(model_simple, "mistral_rope_simple.onnx")


if __name__ == "__main__":
    main()