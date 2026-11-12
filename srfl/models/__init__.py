from dataclasses import dataclass
import torch
from torch import nn
from .encoder import build_encoder
from .decoder import DualScaleDecoder


@dataclass
class ModelOutput:
    logits: torch.Tensor
    usgr_features: torch.Tensor


class SRFL(nn.Module):
    def __init__(self, config, *, load_pretrained=True):
        super().__init__()
        self.config = dict(config)
        self.encoder, channels = build_encoder(config, load_pretrained=load_pretrained)
        self.decoder = DualScaleDecoder(channels, config['decoder_channels'], config['num_classes'],
            config['decoder_heads'], config['decoder_depth'], config['query_grid'], config['fusion'])
        self.register_buffer('pixel_mean', torch.tensor([123.675,116.28,103.53]).reshape(1,3,1,1))
        self.register_buffer('pixel_std', torch.tensor([58.395,57.12,57.375]).reshape(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)

        self.encoder.eval()
        return self

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('SRFL expects [B,3,H,W] images scaled to [0,1]')
        if image.shape[-2:] != (self.config['image_size'], self.config['image_size']):
            raise ValueError('Input spatial shape must match configured image_size')
        if self.config['sam_normalization']:
            image = (image * 255. - self.pixel_mean) / self.pixel_std

        encoded = self.encoder(image)
        logits, fused = self.decoder(encoded, image.shape[-2:])
        feature = encoded if self.config['usgr_feature'] == 'encoder' else fused
        return ModelOutput(logits, feature)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_counts(self):
        return {'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'adapters': sum(p.numel() for n,p in self.named_parameters() if '.adapter.' in n),
                'decoder': sum(p.numel() for p in self.decoder.parameters())}
