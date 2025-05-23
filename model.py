import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

class SwinEmbedding(nn.Module):
    def __init__(self, patch_size=4, emb_size=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.rearrange(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.linear = nn.Linear(4 * emb_size, 2 * emb_size)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        x = self.linear(x)
        return x

class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=8, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        height = width = int(np.sqrt(x.shape[1]))
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)

        if self.shifted:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding

        if self.shifted:
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2))
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)',
                                    w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask

        wei = torch.softmax(wei, dim=-1) @ V
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.linear2(x)

class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
        )

    def forward(self, x):
        return self.ff(x)

class SwinEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=8):
        super().__init__()
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=False)
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size)
        self.MLP = MLP(emb_size)

    def forward(self, x):
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        return x

class Swin(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        for i in range(3):
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2

        self.stage1 = SwinEncoder(96, 6)
        self.stage2 = SwinEncoder(192, 12)
        self.stage3 = nn.ModuleList([
            SwinEncoder(384, 24),
            SwinEncoder(384, 24),
            SwinEncoder(384, 24),
            SwinEncoder(384, 24),
            SwinEncoder(384, 24),
            SwinEncoder(384, 24),
        ])
        self.stage4 = SwinEncoder(768, 48)
        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=1)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.Embedding(x)
        x = self.stage1(x)
        x = self.PatchMerging[0](x)
        x = self.stage2(x)
        x = self.PatchMerging[1](x)
        for stage in self.stage3:
            x = stage(x)
        x = self.PatchMerging[2](x)
        x = self.stage4(x)
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        bbox_pred = self.bbox_regressor(x)
        class_pred = self.classifier(x)
        bbox_pred = torch.sigmoid(bbox_pred) * 512
        bbox_pred = torch.clamp(bbox_pred, 0, 512)
        return bbox_pred, class_pred