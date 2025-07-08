import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Dropout, Linear, LayerNorm

class Attention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(self.all_head_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        return attention_output

class eca_block(nn.Module):
    def __init__(self, in_channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs).view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x).view([b, c, 1, 1])
        outputs = x * inputs
        return outputs

class EPA(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super(EPA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.head_size * self.num_heads

        self.qk_linear = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.v_linear = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.out_linear = nn.Linear(self.all_head_size, hidden_size, bias=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.E = nn.Linear(self.head_size, self.head_size // 3)
        self.F = nn.Linear(self.head_size, self.head_size // 3)

    def forward(self, x):
        B, N, C = x.size()

        k = self.qk_linear(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        q = self.qk_linear(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))

        k_proj = self.E(k.transpose(1, 2))
        v_proj = self.F(v.transpose(1, 2))

        attn_SA = (k_proj.permute(0, 1, 3, 2) @ q.transpose(1, 2))
        attn_SA = self.attn_drop(F.softmax(attn_SA, dim=-1))

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_linear(out)

        x_CA = (v_proj @ attn_SA).contiguous().view(B, N, self.all_head_size)
        x_CA = self.out_linear(x_CA)

        out = 0.5 * (out + x_CA)
        return out

class MSAMLP_Block(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(MSAMLP_Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)

        self.epa = EPA(num_attention_heads, hidden_size)
        self.eca = eca_block(3)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.epa(x)
        x = (x + h).unsqueeze(2).permute([1, 0, 2, 3])

        h = x
        x = self.ffn_norm(x)
        x = self.eca(x)
        x = (x + h).permute([1, 0, 2, 3]).squeeze(2)

        return x  # 直接返回x，不要进行mean操作
