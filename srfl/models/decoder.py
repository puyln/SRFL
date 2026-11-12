import math
import torch
from torch import nn
import torch.nn.functional as F


def position_encoding(height, width, channels, device, dtype):
    if channels % 4:
        raise ValueError('decoder channels must be divisible by 4')
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    frequency = torch.exp(-math.log(10000.) * torch.arange(channels // 4, device=device) / max(channels // 4 - 1, 1))
    xx = x.flatten()[:, None] * frequency
    yy = y.flatten()[:, None] * frequency
    return torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=1).to(dtype)[None]


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))

    def forward(self, query, memory, query_pos, memory_pos):
        normalized = self.norms[0](query)
        qk = normalized + query_pos
        query = query + self.self_attention(qk, qk, normalized, need_weights=False)[0]
        normalized = self.norms[1](query)
        query = query + self.cross_attention(normalized + query_pos, memory + memory_pos,
                                               memory, need_weights=False)[0]
        return query + self.mlp(self.norms[2](query))


class DualScaleDecoder(nn.Module):
    def __init__(self, input_channels, channels, num_classes, heads=4, depth=2, query_grid=16, fusion='multiply'):
        super().__init__()
        if channels % heads or channels % 8:
            raise ValueError('decoder channels must be divisible by attention heads and 8')
        self.query_grid = query_grid
        self.fusion_mode = fusion
        self.projection = nn.Conv2d(input_channels, channels, 1)
        self.attention = nn.ModuleList([AttentionBlock(channels, heads) for _ in range(depth)])
        self.pixel = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, stride=2), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU(),
            nn.ConvTranspose2d(channels, channels, 2, stride=2), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d((2 if fusion == 'concat' else 1) * channels, channels, 3, padding=1),
                                    nn.GroupNorm(8, channels), nn.GELU())
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, encoded, output_size):
        projected = self.projection(encoded)
        b, c, h, w = projected.shape
        qh, qw = min(h, self.query_grid), min(w, self.query_grid)

        query = F.adaptive_avg_pool2d(projected, (qh, qw)).flatten(2).transpose(1, 2)
        memory = projected.flatten(2).transpose(1, 2)
        qpos = position_encoding(qh, qw, c, encoded.device, encoded.dtype)
        mpos = position_encoding(h, w, c, encoded.device, encoded.dtype)
        for layer in self.attention:
            query = layer(query, memory, qpos, mpos)
        global_map = query.transpose(1, 2).reshape(b, c, qh, qw)
        pixel_map = self.pixel(encoded)
        global_map = F.interpolate(global_map, size=pixel_map.shape[-2:], mode='bilinear', align_corners=False)
        combined = global_map * pixel_map if self.fusion_mode == 'multiply' else torch.cat([global_map, pixel_map], dim=1)
        fused = self.fusion(combined)
        logits = F.interpolate(self.classifier(fused), size=output_size, mode='bilinear', align_corners=False)
        return logits, fused
