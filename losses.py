"""
Extra generator losses for cloud removal (all take tensors in [-1, 1], shape B x C x H x W).

  edge_loss   - L1 between Sobel gradients of prediction and ground truth.
                Punishes lost coastlines / roads / piers (the "flat dark output" failure).
  ssim_loss   - 1 - SSIM, differentiable. Directly optimises the SSIM metric you report.
  VGGPerceptual - L1 between frozen VGG16 features. Pushes towards realistic texture.

All of them run in float32 even when training with mixed precision (--amp 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fp32_ctx(x):
    return torch.autocast(device_type=x.device.type, enabled=False)


# ------------------------------------------------------------------ edge loss
def _sobel(x):
    c = x.shape[1]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype) / 8.0
    ky = kx.t().contiguous()
    kx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def edge_loss(pred, target, weight_map=None):
    """
    L1 distance between Sobel gradients. weight_map (B x 1 x H x W) lets you weight some
    pixels more (e.g. 1 + boost * cloud_mask).
    """
    with _fp32_ctx(pred):
        pred, target = pred.float(), target.float()
        gxp, gyp = _sobel(pred)
        gxt, gyt = _sobel(target)
        diff = (gxp - gxt).abs() + (gyp - gyt).abs()          # B x C x H x W
        if weight_map is None:
            return diff.mean()
        w = weight_map.float()
        return (w * diff).sum() / (w.sum() * diff.shape[1] + 1e-6)


# ------------------------------------------------------------------ SSIM loss
def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    """1 - mean SSIM (Gaussian window), computed on images mapped to [0, 1]."""
    with _fp32_ctx(pred):
        x = (pred.float() + 1.0) / 2.0
        y = (target.float() + 1.0) / 2.0
        c = x.shape[1]
        w = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
        pad = window_size // 2

        mu_x = F.conv2d(x, w, padding=pad, groups=c)
        mu_y = F.conv2d(y, w, padding=pad, groups=c)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
        sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
        sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
        return 1.0 - ssim_map.mean()


# ------------------------------------------------------------------ perceptual loss
class VGGPerceptual(nn.Module):
    """
    L1 between VGG16 features (relu1_2, relu2_2, relu3_3). Needs 3-channel input.
    Downloads ImageNet weights on first use (Colab has internet access).
    """

    FEATURE_IDS = (3, 8, 15)

    def __init__(self):
        super().__init__()
        from torchvision import models
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:                                   # older torchvision
            vgg = models.vgg16(pretrained=True)
        self.features = vgg.features[: max(self.FEATURE_IDS) + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract(self, x):
        x = ((x + 1.0) / 2.0 - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.FEATURE_IDS:
                feats.append(x)
        return feats

    def forward(self, pred, target):
        n = pred.shape[0]
        feats = self._extract(torch.cat([pred, target.detach()], dim=0))
        loss = 0.0
        for f in feats:
            loss = loss + F.l1_loss(f[:n].float(), f[n:].float())
        return loss / len(feats)
