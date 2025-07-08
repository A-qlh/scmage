import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
# 添加多尺度注意力导入
from MSA2 import MSAMLP_Block


class AutoEncoder(nn.Module):
    def __init__(
            self,
            num_genes,
            hidden_size=128,
            dropout=0,
            masked_data_weight=0.75,
            mask_loss_weight=0.7,
            noise_sd=0.1,
            num_attention_heads=4,  # 新增参数设置 attention heads
            attention_hidden_size=128  # 新增参数设置 attention hidden size
    ):
        super().__init__()

        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight
        self.noise_sd = noise_sd

        # Encoder layers (without the last embedding linear layer)
        self.encoder_initial = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True)
        )

        # 加入MSA模块
        self.msa_block = MSAMLP_Block(num_attention_heads=num_attention_heads, hidden_size=attention_hidden_size)

        # latent 表征层（从 attention_hidden_size 到最终 hidden_size 的映射）
        self.latent_linear = nn.Linear(attention_hidden_size, hidden_size)

        # Mask predictor and decoder
        self.mask_predictor = nn.Linear(hidden_size, num_genes)
        self.decoder = nn.Linear(
            in_features=hidden_size + num_genes, out_features=num_genes)

    def add_noise(self, x):
        if self.training and self.noise_sd > 0:
            noise = torch.normal(mean=0, std=self.noise_sd, size=x.size(), device=x.device)
            return x + noise
        return x

    def forward_mask(self, x, add_noise=True):
        x_noisy = self.add_noise(x) if add_noise else x
        x_encoded = self.encoder_initial(x_noisy)

        # MSA attention expects input as [batch, seq_len, dim]; add a pseudo sequence length
        x_encoded = x_encoded.unsqueeze(1)  # [B, 1, hidden_dim]

        # MSA block
        x_attention = self.msa_block(x_encoded)

        # 去除伪序列维度
        x_attention = x_attention.squeeze(1)  # [B, hidden_dim]

        latent = self.latent_linear(x_attention)

        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(torch.cat([latent, predicted_mask], dim=1))

        return latent, predicted_mask, reconstruction

    def loss_mask(self, x, y, mask, add_noise=True):
        latent, predicted_mask, reconstruction = self.forward_mask(x, add_noise=add_noise)

        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
        reconstruction_loss = (1 - self.mask_loss_weight) * torch.mul(
            w_nums, mse(reconstruction, y, reduction='none'))

        mask_loss = self.mask_loss_weight * \
                    bce_logits(predicted_mask, mask, reduction="mean")
        reconstruction_loss = reconstruction_loss.mean()

        loss = reconstruction_loss + mask_loss
        return latent, loss

    def feature(self, x, add_noise=False):
        x_encoded = self.encoder_initial(x)
        x_encoded = x_encoded.unsqueeze(1)  # [B, 1, hidden_dim]

        x_attention = self.msa_block(x_encoded)
        x_attention = x_attention.squeeze(1)

        latent = self.latent_linear(x_attention)
        return latent
