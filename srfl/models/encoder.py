from pathlib import Path
from functools import partial
import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(torch.relu(self.down(x)))


class AdaptedOperation(nn.Module):
    def __init__(self, original: nn.Module, channels: int, bottleneck: int):
        super().__init__()
        self.original = original
        self.adapter = BottleneckAdapter(channels, bottleneck)

    def forward(self, x):
        value = self.original(x)
        return value + self.adapter(value)


def attach_adapters(encoder: nn.Module, bottleneck: int, enabled: bool = True):

    encoder.requires_grad_(False)
    if enabled:
        for block in encoder.blocks:
            channels = block.norm1.normalized_shape[0]
            block.attn = AdaptedOperation(block.attn, channels, bottleneck)
            block.mlp = AdaptedOperation(block.mlp, channels, bottleneck)
    return encoder


class TinyAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, h, w, c = x.shape
        tokens = x.reshape(b, h * w, c)
        return self.attention(tokens, tokens, tokens, need_weights=False)[0].reshape(b, h, w, c)


class TinyBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = TinyAttention(channels, heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(),
                                 nn.Linear(channels * 4, channels))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyEncoder(nn.Module):

    def __init__(self, image_size=32, channels=32, depth=2):
        super().__init__()
        self.img_size = image_size
        self.patch_embed = nn.Conv2d(3, channels, 4, stride=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, image_size // 4, image_size // 4, channels))
        self.blocks = nn.ModuleList([TinyBlock(channels, 4) for _ in range(depth)])
        self.neck = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.neck(x.permute(0, 3, 1, 2))


def build_encoder(config, *, load_pretrained=True):
    name = config['backbone']
    if name == 'tiny':
        encoder = TinyEncoder(config['image_size'], config['encoder_channels'], config['encoder_depth'])
        channels = config['encoder_channels']
    else:
        if name not in ('vit_b', 'vit_l'):
            raise ValueError('backbone must be tiny, vit_b or vit_l')
        if config['image_size'] != 1024:
            raise ValueError('The official SAM ViT-B/L checkpoint configuration requires image_size=1024')
        try:
            from segment_anything.modeling import ImageEncoderViT
        except ImportError as exc:
            raise ImportError('Install the pinned SAM dependency from requirements-sam.txt') from exc
        embed, depth, heads, global_layers = {
            'vit_b': (768, 12, 12, (2, 5, 8, 11)),
            'vit_l': (1024, 24, 16, (5, 11, 17, 23)),
        }[name]


        encoder = ImageEncoderViT(img_size=1024, patch_size=16, embed_dim=embed,
            depth=depth, num_heads=heads, out_chans=256, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), use_rel_pos=True,
            window_size=14, global_attn_indexes=global_layers)
        channels = 256
        if load_pretrained:
            checkpoint = Path(config['sam_checkpoint'])
            if not checkpoint.is_file():
                raise FileNotFoundError(f'SAM checkpoint not found: {checkpoint}')
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if 'state_dict' in weights:
                weights = weights['state_dict']
            prefix = 'image_encoder.'
            selected = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

            encoder.load_state_dict(selected if selected else weights, strict=True)
    if name!='tiny' and config.get('attention_backend','original')=='sdpa_global':
        import types
        from .attention import sdpa_attention_forward
        for block in encoder.blocks:
            if block.window_size==0:
                block.attn.forward=types.MethodType(sdpa_attention_forward,block.attn)
    return attach_adapters(encoder, config['adapter_dim'], config['adapters']), channels
