"""Zero-DCE objective adapted to LiteIE's single shared RGB curve.

Equations/reductions follow Li-Chongyi/Zero-DCE, commit
 e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd, Zero-DCE_code/Myloss.py
and lowlight_train.py. This is NOT the unpublished LiteIE paper loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ZeroDCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('directions', torch.tensor([
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        ], dtype=torch.float32).unsqueeze(1))

    def forward(self, low, enhanced, curves):
        if min(enhanced.shape[-2:]) < 16:
            raise ValueError('Zero-DCE exposure pooling requires image dimensions >= 16')
        # Original TV sums channels, normalizes by spatial counts and batch.
        # LiteIE provides one 3-channel curve, unlike Zero-DCE's 24 channels.
        dh = curves[:, :, 1:, :] - curves[:, :, :-1, :]
        dw = curves[:, :, :, 1:] - curves[:, :, :, :-1]
        tv = 2 * (dh.square().sum() / (dh.shape[-2]*dh.shape[-1])
                  + dw.square().sum() / (dw.shape[-2]*dw.shape[-1])) / len(curves)
        original = F.avg_pool2d(low.mean(1, keepdim=True), 4)
        output = F.avg_pool2d(enhanced.mean(1, keepdim=True), 4)
        differences = (F.conv2d(original, self.directions, padding=1)
                       - F.conv2d(output, self.directions, padding=1))
        spatial = differences.square().sum(1).mean()
        means = enhanced.mean((2, 3))
        differences = torch.stack((means[:, 0]-means[:, 1], means[:, 0]-means[:, 2],
                                   means[:, 2]-means[:, 1]), dim=1)
        # Norm of squared differences equals sqrt(sum(channel_difference**4)).
        # vector_norm also gives finite gradients at exact gray (all zero).
        color = torch.linalg.vector_norm(differences.square(), dim=1).mean()
        exposure = (F.avg_pool2d(enhanced.mean(1, keepdim=True), 16) - .6).square().mean()
        components = {'tv': 200*tv, 'spatial': spatial, 'color': 5*color, 'exposure': 10*exposure}
        return sum(components.values()), components
