import torch
import torch.nn as nn

class VanillaSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.T
        attn_wts = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vecs = attn_wts @ values
        return context_vecs
    
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout_prob, qkv_bias=False) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )
    
    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
        attn_wts = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_wts = self.dropout_layer(attn_wts)

        context_vecs = attn_wts @ values
        return context_vecs
