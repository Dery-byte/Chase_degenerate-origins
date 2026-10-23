"""
fixed_origin.py -- baselines for the CAD "is CHASE a fixed-joint recentring?" test.
Copy to CHASE/NeurIPS24/model/Wrapper/fixed_origin.py.

mode='joint' : origin := x[:, :, t, v, m] for every sample (the point the released
               CHASE checkpoint selects in 761/761 CAD test clips: t=3, v=4, m=0).
               Padding is handled exactly as in CHASWrapper (x - p for all points),
               so the ONLY difference from CHASE is that the origin is fixed, not learned.
mode='none'  : no shift (vanilla backbone), same wrapper so eval returns logits.
"""
import torch.nn as nn


class FixedOriginWrapper(nn.Module):
    def __init__(self, backbone, mode="joint", t=3, v=4, m=0, **unused):
        super().__init__()
        assert mode in ("joint", "none")
        self.backbone, self.mode, self.t, self.v, self.m = backbone, mode, t, v, m

    def forward(self, x):
        if self.mode == "joint":
            p = x[:, :, self.t, self.v, self.m]                     # (N, C)
            x = x - p[:, :, None, None, None]
        out = self.backbone(x)
        if not self.training and isinstance(out, tuple):
            return out[0]
        return out
