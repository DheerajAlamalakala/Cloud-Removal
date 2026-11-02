"""
CycleGAN model definitions for cloud removal.

- ResnetGenerator: translates between cloudy <-> cloud-free domains
- PatchDiscriminator: 70x70 PatchGAN discriminator

Supports arbitrary input channel counts, so it works with:
  - RGB-only datasets (e.g. RICE, 3 channels)
  - Multispectral Sentinel-2 datasets (e.g. SEN12MS-CR, 13 bands) if you
    preprocess the .tif stacks into multi-channel tensors.
"""

import functools
import torch
import torch.nn as nn


def weights_init(m):
    """Standard CycleGAN weight initialization (Normal, mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find("Conv") != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (V2 only)."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResnetBlock(nn.Module):
    """A single residual block with reflection padding + InstanceNorm."""

    def __init__(self, dim, use_dropout=False, use_bias=True, se=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        ]
        if se:
            layers.append(SEBlock(dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based generator, the standard CycleGAN architecture
    (Johnson et al. style encoder-resblocks-decoder).

    Default: 9 residual blocks (recommended for 256x256 inputs).
    Use n_blocks=6 for smaller (e.g. 128x128) patches.
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9, use_dropout=False):
        super().__init__()
        use_bias = True  # InstanceNorm -> use bias in convs

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]

        # Residual blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2), affine=False, track_running_stats=False),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
        ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


DEFAULT_ARCH = dict(upsample="deconv", skip=False, se=False, residual_out=False)


def _norm(c):
    return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)


class ResnetGeneratorV2(nn.Module):
    """
    Upgraded generator for cloud removal. Same overall shape as ResnetGenerator but with
    optional upgrades (each switchable, so you can ablate them):

      upsample="resize"  nearest upsample + conv instead of ConvTranspose2d
                         (removes the checkerboard artifacts seen in your fake_A row)
      skip=True          U-Net style skips from encoder to decoder, so high-resolution
                         detail (piers, coastlines, roads) bypasses the ResNet bottleneck
      se=True            squeeze-and-excitation channel attention in every residual block
      residual_out=True  output = clamp(input + delta): the network only predicts the
                         correction, so clear land is copied through almost for free
                         (needs in_channels == out_channels)
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9, use_dropout=False,
                 upsample="resize", skip=True, se=True, residual_out=True):
        super().__init__()
        assert upsample in ("deconv", "resize")
        if residual_out:
            assert in_channels == out_channels, "residual_out needs in_channels == out_channels"
        self.skip, self.residual_out = skip, residual_out

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, 7), _norm(ngf), nn.ReLU(True))
        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1), _norm(ngf * 2), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), _norm(ngf * 4), nn.ReLU(True))
        self.blocks = nn.Sequential(
            *[ResnetBlock(ngf * 4, use_dropout=use_dropout, se=se) for _ in range(n_blocks)])

        self.up1 = self._up(ngf * 4, ngf * 2, upsample)
        self.up2 = self._up(ngf * 2, ngf, upsample)
        if skip:
            self.fuse1 = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(ngf * 4, ngf * 2, 3), _norm(ngf * 2), nn.ReLU(True))
            self.fuse2 = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(ngf * 2, ngf, 3), _norm(ngf), nn.ReLU(True))
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_channels, 7))

    @staticmethod
    def _up(cin, cout, mode):
        if mode == "resize":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3), _norm(cout), nn.ReLU(True))
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1, output_padding=1),
            _norm(cout), nn.ReLU(True))

    def forward(self, x):
        s = self.stem(x)
        d1 = self.down1(s)
        d2 = self.down2(d1)
        h = self.up1(self.blocks(d2))
        if self.skip:
            h = self.fuse1(torch.cat([h, d1], 1))
        h = self.up2(h)
        if self.skip:
            h = self.fuse2(torch.cat([h, s], 1))
        out = self.out(h)
        if self.residual_out:
            return torch.clamp(x + torch.tanh(out), -1.0, 1.0)
        return torch.tanh(out)


def make_generator(in_ch, out_ch, ngf=64, n_blocks=9, arch=None):
    """Old ResnetGenerator when every arch option is at its default (so old checkpoints
    load unchanged); ResnetGeneratorV2 otherwise."""
    a = dict(DEFAULT_ARCH)
    a.update(arch or {})
    if a == DEFAULT_ARCH:
        return ResnetGenerator(in_ch, out_ch, ngf=ngf, n_blocks=n_blocks)
    return ResnetGeneratorV2(in_ch, out_ch, ngf=ngf, n_blocks=n_blocks, **a)


def build_models(in_ch_A, in_ch_B, ngf=64, ndf=64, n_blocks=9, device="cpu", arch=None):
    """
    2 generators + 2 discriminators for CycleGAN, with weight initialization.

    Domain A = cloudy images, Domain B = cloud-free images.
    G_A2B: cloudy -> cloud-free   (the model you actually deploy for inference)
    G_B2A: cloud-free -> cloudy   (needed only for cycle-consistency during training)
    """
    G_A2B = make_generator(in_ch_A, in_ch_B, ngf, n_blocks, arch).to(device)
    G_B2A = make_generator(in_ch_B, in_ch_A, ngf, n_blocks, arch).to(device)
    D_A = PatchDiscriminator(in_ch_A, ndf=ndf).to(device)
    D_B = PatchDiscriminator(in_ch_B, ndf=ndf).to(device)

    for net in (G_A2B, G_B2A, D_A, D_B):
        net.apply(weights_init)

    return G_A2B, G_B2A, D_A, D_B
