import torch
import torch.nn as nn


class CBAMLayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.combine = nn.Conv2d(channel, int(channel / 2), kernel_size=1)
        self.assemble = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self._forward_channel(x)
        x = self._forward_spatial(x)
        return x

    def _forward_channel(self, x):
        # Channel attention module (SE with max-pool and average-pool)
        b, c, _, _ = x.size()
        x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        y = torch.sigmoid(x_avg + x_max)
        return self.combine(x * y)

    def _forward_spatial(self, x):
        # Spatial attention module
        x_avg = torch.mean(x, 1, True)
        x_max, _ = torch.max(x, 1, True)
        y = torch.cat((x_avg, x_max), 1)
        y = torch.sigmoid(self.assemble(y))
        return x * y
