import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from .dataset import RGB


def _gauss(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y):
    """Per-pixel SSIM averaged over channels -> (B,1,H,W). Inputs in [0,1]."""
    C = x.shape[1]
    w = _gauss(device=x.device).expand(C, 1, 11, 11).to(x.dtype)
    conv = lambda t: F.conv2d(t, w, padding=5, groups=C)
    mx, my = conv(x), conv(y)
    sxx, syy, sxy = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return m.mean(1, keepdim=True)


def edge_loss(p, t):
    gx = lambda a: a[..., :, 1:] - a[..., :, :-1]
    gy = lambda a: a[..., 1:, :] - a[..., :-1, :]
    return F.l1_loss(gx(p), gx(t)) + F.l1_loss(gy(p), gy(t))


def sam_map(p, t, eps=1e-8):
    """Spectral angle in degrees per pixel -> (B,1,H,W)."""
    cos = (p * t).sum(1, keepdim=True) / (p.norm(dim=1, keepdim=True) * t.norm(dim=1, keepdim=True) + eps)
    return torch.rad2deg(torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))


def cloud_mask(cloudy, clear, thr=0.05):
    """Pseudo-mask from |cloudy-clear| on RGB bands; not a ground-truth cloud label."""
    d = (cloudy - clear).abs()[:, RGB].mean(1, keepdim=True)
    return (F.avg_pool2d(d, 5, 1, 2) > thr).float()


class LPIPSLoss(torch.nn.Module):
    """LPIPS on RGB bands (B4,B3,B2), brightened for display and scaled to [-1,1]."""

    def __init__(self, net="vgg", spatial=False):
        super().__init__()
        import lpips
        self.m = lpips.LPIPS(net=net, verbose=False, spatial=spatial).eval()
        for p in self.m.parameters():
            p.requires_grad = False

    @staticmethod
    def prep(x):
        return (x[:, RGB].float() * 2.5).clamp(0, 1) * 2 - 1

    def forward(self, p, t):
        context = torch.autocast("cuda", enabled=False) if p.is_cuda else nullcontext()
        with context:
            return self.m(self.prep(p), self.prep(t)).mean()


class CRLoss(torch.nn.Module):
    def __init__(self, l_ssim=0.5, l_edge=2.0, l_lpips=0.1,
                 cloud_w=4.0, lpips_net="vgg"):
        super().__init__()

        self.l_ssim = l_ssim
        self.l_edge = l_edge
        self.l_lpips = l_lpips
        self.cloud_w = cloud_w

        # Sentinel-2 spectral weights: B01 through B12.
        weights = torch.tensor([
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.5, 1.5, 1.5, 1.5,
            1.0, 1.0,
            1.5, 1.5
        ], dtype=torch.float32)

        weights = weights / weights.mean()
        self.register_buffer("band_weights", weights.view(1, 13, 1, 1))
        self.lp = LPIPSLoss(lpips_net) if l_lpips > 0 else None

    def forward(self, pred, clear, cloudy):
        pred = pred.float()
        m = cloud_mask(cloudy, clear)
        l1 = (
            (pred - clear).abs()
            * self.band_weights
            * (1 + self.cloud_w * m)
        ).mean()
        pc = pred.clamp(0, 1)
        parts = dict(l1=l1, ssim=1 - ssim_map(pc, clear).mean(), edge=edge_loss(pred, clear))
        total = l1 + self.l_ssim * parts["ssim"] + self.l_edge * parts["edge"]
        if self.lp is not None:
            parts["lpips"] = self.lp(pc, clear)
            total = total + self.l_lpips * parts["lpips"]
        return total, parts


@torch.no_grad()
def per_image_metrics(pred, clear, mask, lp_model=None):
    """All tensors (B,C,H,W)/(B,1,H,W) in [0,1]. Returns dict of (B,) tensors; *_cloud is NaN if mask empty."""
    pred = pred.clamp(0, 1)
    B = pred.shape[0]
    se = ((pred - clear) ** 2).mean(1, keepdim=True)          # (B,1,H,W)
    sm, am = ssim_map(pred, clear), sam_map(pred, clear)
    cnt = mask.flatten(1).sum(1)
    wmean = lambda t: (t * mask).flatten(1).sum(1) / cnt.clamp(min=1)
    nan = torch.full((B,), float("nan"), device=pred.device)
    has = cnt > 0
    mse_f, mse_c = se.flatten(1).mean(1), wmean(se)
    out = dict(
        PSNR_full=-10 * torch.log10(mse_f.clamp(min=1e-10)),
        PSNR_cloud=torch.where(has, -10 * torch.log10(mse_c.clamp(min=1e-10)), nan),
        RMSE_full=mse_f.sqrt(), RMSE_cloud=torch.where(has, mse_c.sqrt(), nan),
        SSIM_full=sm.flatten(1).mean(1), SSIM_cloud=torch.where(has, wmean(sm), nan),
        SAM_full=am.flatten(1).mean(1), SAM_cloud=torch.where(has, wmean(am), nan),
        mask_cov=mask.flatten(1).mean(1),
    )
    if lp_model is not None:
        lm = lp_model.m(lp_model.prep(pred), lp_model.prep(clear))  # spatial=True -> (B,1,H,W)
        out["LPIPS_full"] = lm.flatten(1).mean(1)
        out["LPIPS_cloud"] = torch.where(has, wmean(lm), nan)
    return out
