import os
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from .Embed import DataEmbedding

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        hidden = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class Model(nn.Module):
    """
    Adapted GPT4CP model for narrowband stream-wise prediction.

    Input:
        x_enc: [B, prev_len, 2]
    Output:
        [B, pred_len, 2]
    """

    def __init__(
        self,
        gpt_type='gpt2',
        d_ff=768,
        d_model=768,
        gpt_layers=6,
        pred_len=4,
        prev_len=16,
        use_gpu=1,
        gpu_id=0,
        mlp=0,
        res_layers=4,
        patch_size=4,
        res_dim=64,
        embed='timeF',
        freq='h',
        dropout=0.1
    ):
        super(Model, self).__init__()

        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.d_ff = d_ff
        self.d_model = d_model

        # For your adapted data:
        # each stream is one complex scalar => real/imag => D=2
        self.enc_in_complex = 1
        self.c_out_complex = 1
        self.enc_in_real = 2 * self.enc_in_complex   # = 2
        self.c_out_real = 2 * self.c_out_complex     # = 2

        # Embedding block
        self.enc_embedding1 = DataEmbedding(
            self.enc_in_real, self.d_model, embed, freq, dropout
        )

        # GPT-2 backbone
        if gpt_type == 'gpt2-medium':
            self.gpt2 = GPT2Model.from_pretrained(
                'gpt2-medium',
                output_attentions=True,
                output_hidden_states=True
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1024
        elif gpt_type == 'gpt2-large':
            self.gpt2 = GPT2Model.from_pretrained(
                'gpt2-large',
                output_attentions=True,
                output_hidden_states=True
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1280
        elif gpt_type == 'gpt2-xl':
            self.gpt2 = GPT2Model.from_pretrained(
                'gpt2-xl',
                output_attentions=True,
                output_hidden_states=True
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 1600
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                'gpt2',
                output_attentions=True,
                output_hidden_states=True
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]
            self.gpt_dim = 768

        # Freeze most GPT parameters, fine-tune LN / positional embeddings
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if use_gpu and torch.cuda.is_available():
            self.gpt2.to(device=self.device)

        # Patch layers along time
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_delay = nn.Linear(self.patch_size, self.patch_size)

        # Temporal projection before GPT
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)

        # Output heads
        self.out_layer_dim = nn.Linear(d_ff, self.c_out_real)
        self.output_layer_time = nn.Sequential(
            nn.Linear(self.prev_len, self.pred_len)
        )

        # Residual blocks for frequency/spatial branch and delay branch
        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, kernel_size=3, stride=1, padding=1))
        self.RB_d = nn.Sequential(nn.Conv2d(2, res_dim, kernel_size=3, stride=1, padding=1))

        for _ in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
            self.RB_d.append(Res_block(res_dim))

        self.RB_e.append(nn.Conv2d(res_dim, 2, kernel_size=3, stride=1, padding=1))
        self.RB_d.append(nn.Conv2d(res_dim, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, prev_len, 2]
        """
        # Normalize per batch
        mean = torch.mean(x_enc)
        std = torch.std(x_enc, unbiased=False) + 1e-12
        x_enc = (x_enc - mean) / std

        B, L, enc_in = x_enc.shape
        assert L == self.prev_len, f"Expected prev_len={self.prev_len}, got {L}"
        assert enc_in == self.enc_in_real, f"Expected input dim={self.enc_in_real}, got {enc_in}"
        assert L % self.patch_size == 0, f"prev_len={L} must be divisible by patch_size={self.patch_size}"

        # ---------------- Delay-domain branch ----------------
        # [B, L, 2] -> [B, L, 1, 2]
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)

        # complex representation: [B, L, 1]
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])

        # ifft over "k" dimension; here k=1, so this branch is mostly formal
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)

        # back to real: [B, L, 2]
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)

        # patch over time
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_delay = self.patch_layer_delay(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)

        # [B, L, 2] -> [B, 2, L, 1]
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)
        x_enc_delay = self.RB_d(x_enc_delay)

        # ---------------- Original branch ----------------
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)

        # [B, L, 2] -> [B, 2, L, 1]
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc_fre = self.RB_e(x_enc_fre)

        # Fuse branches
        x_enc = x_enc_fre + x_enc_delay

        # [B, 2, L, 1] -> [B, L, 2]
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)

        # Embedding
        enc_out = self.enc_embedding1(x_enc, x_mark_enc)   # [B, L, d_model]

        # Temporal linear projection
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Pad to GPT hidden size if needed
        if enc_out.shape[-1] < self.gpt_dim:
            enc_out = torch.nn.functional.pad(enc_out, (0, self.gpt_dim - enc_out.shape[-1]))
        elif enc_out.shape[-1] > self.gpt_dim:
            enc_out = enc_out[:, :, :self.gpt_dim]

        # GPT backbone
        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state   # [B, L, gpt_dim]
        dec_out = dec_out[:, :, :self.d_ff]                            # [B, L, d_ff]

        # Output projection
        dec_out = self.out_layer_dim(dec_out)                          # [B, L, 2]
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred_len, 2]

        # De-normalize
        dec_out = dec_out * std + mean

        return dec_out[:, -self.pred_len:, :]   # [B, pred_len, 2]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(pred_len=4, prev_len=16, patch_size=4).to(device)

    inputs = torch.rand(3, 16, 2).to(device)
    out = model(inputs, None, None, None)

    print("input shape :", inputs.shape)
    print("output shape:", out.shape)

    total = sum(param.nelement() for param in model.parameters())
    print("Number of parameter: %.5fM" % (total / 1e6))

    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))