import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttn(nn.Module):
    def __init__(self, c, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        t = self.attn(t, t, t, need_weights=False)[0]
        return x + t.transpose(1, 2).reshape(B, C, H, W)


class ResUNet(nn.Module):
    """Residual U-Net. Input: cloudy S2 (13) [+ S1 (2)]. Output = cloudy + predicted correction."""

    def __init__(self, s2_ch=13, s1_ch=2, base=64, mults=(1, 2, 4, 8), use_sar=True):
        super().__init__()
        self.use_sar = use_sar
        chs = [base * m for m in mults]
        self.inc = nn.Conv2d(s2_ch + (s1_ch if use_sar else 0), chs[0], 3, padding=1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, ch in enumerate(chs):
            self.enc.append(nn.Sequential(ResBlock(c, ch), ResBlock(ch, ch)))
            c = ch
            if i < len(chs) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.mid = nn.Sequential(ResBlock(c, c), SelfAttn(c), ResBlock(c, c))
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                         nn.Conv2d(c, chs[i], 3, padding=1)))
            self.dec.append(nn.Sequential(ResBlock(chs[i] * 2, chs[i]), ResBlock(chs[i], chs[i])))
            c = chs[i]
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, s2_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, cloudy, s1=None):
        x = torch.cat([cloudy, s1], 1) if (self.use_sar and s1 is not None) else cloudy
        h = self.inc(x)
        skips = []
        for i, e in enumerate(self.enc):
            h = e(h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        h = self.mid(h)
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        return cloudy + self.out(h)