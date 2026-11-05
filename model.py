import torch
import torch.nn as nn

coeffs = [10.6301,-49.3563,193.9553,-507.2211,582.7439]
class LiteIE(nn.Module):
    def __init__(self):
        super(LiteIE, self).__init__()

        self.relu = nn.LeakyReLU(0.01)

        bottleneck_channels = 1
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, bottleneck_channels, 3, padding=1,bias=True),
            nn.Conv2d(bottleneck_channels, 3, 3, padding=1, bias=True)
        )

    def forward(self, x, return_curves=False):

        x1 = self.relu(self.shared_conv(x))
        x2 = self.relu(self.shared_conv(x1))
        curves = torch.tanh((self.shared_conv(x2)))

        # Keep the original brightness rule, independently for each image.
        values = x.detach().mean(dim=(1, 2, 3)).double()
        polynomial = sum(c * values**i for i, c in enumerate(coeffs))
        steps = torch.where(values < 0.1, 8, polynomial.round().clamp(1, 15)).long()
        for iteration in range(int(steps.max().item())):
            updated = x + curves * (x*x - x)
            x = torch.where((steps > iteration)[:, None, None, None], updated, x)
        return (x, curves) if return_curves else x
