"""
U-Net generator for cloud removal (supervised baseline for the CycleGAN comparison).

Input / output: [B, C, H, W] in [-1, 1] (same convention as models.py / evaluate.py).
H and W must be multiples of 32 (256 is fine).

Bilinear upsample + conv in the decoder (no checkerboard artefacts), InstanceNorm
(works with batch size 4), skip connections at every scale.
"""
import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.InstanceNorm2d(cout, affine=True),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            _block(cout, cout),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _block(cin, cout),
        )
        self.fuse = _block(cout + skip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        b = base
        self.inc = nn.Sequential(_block(in_ch, b), _block(b, b))      # 256
        self.d1 = Down(b, 2 * b)                                       # 128
        self.d2 = Down(2 * b, 4 * b)                                   # 64
        self.d3 = Down(4 * b, 8 * b)                                   # 32
        self.d4 = Down(8 * b, 8 * b)                                   # 16
        self.d5 = Down(8 * b, 8 * b)                                   # 8
        self.u1 = Up(8 * b, 8 * b, 8 * b)                              # 16
        self.u2 = Up(8 * b, 8 * b, 4 * b)                              # 32
        self.u3 = Up(4 * b, 4 * b, 2 * b)                              # 64
        self.u4 = Up(2 * b, 2 * b, b)                                  # 128
        self.u5 = Up(b, b, b)                                          # 256
        self.outc = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0, "H and W must be multiples of 32"
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        y = self.u1(x5, x4)
        y = self.u2(y, x3)
        y = self.u3(y, x2)
        y = self.u4(y, x1)
        y = self.u5(y, x0)
        return torch.tanh(self.outc(y))


if __name__ == "__main__":
    g = UNetGenerator()
    n = sum(p.numel() for p in g.parameters()) / 1e6
    out = g(torch.randn(2, 3, 256, 256))
    print(f"params: {n:.1f}M | output {tuple(out.shape)} range [{out.min():.2f}, {out.max():.2f}]")