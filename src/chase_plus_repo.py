"""
chase_plus_repo.py -- adapter that plugs CHASE+ (chase_plus.py) into the CHASE
repository's training loop (main.py / main_group.py) without changing it.

Copy BOTH files to CHASE/NeurIPS24/model/Wrapper/ and add to its __init__.py:
    from . import chase_plus, chase_plus_repo

Train mode : returns backbone outputs + (aux_loss,)   where
             aux_loss = lam_rbr * L_rbr + lam_sd * L_sd     (shape (1,), DataParallel-safe)
Eval mode  : returns the class logits only.

Use with the loss 'LSCE_AUX_GROUP' (CAD/VD) or 'CE_AUX' (NTU), which add
x_tuple[-1].mean() to the usual classification loss (see patch_loss.py text in the notebook).
"""
import torch
import torch.nn as nn

from .chase_plus import CHASEPlus, SDMPDLoss


class CHASEPlusRepo(nn.Module):
    def __init__(self, backbone, in_channels=2, num_frame=10, num_point=17, num_entity=6,
                 c1=64, c2=8, centred=True, masked=True, hierarchical=True, tau=1.0,
                 kappa_lo=10.0, kappa_hi=None, kappa_floor=None, lam_rbr=1.0, lam_sd=0.1, eps=None,
                 sd_points=8, sd_pairs=4, **unused):
        super().__init__()
        self.core = CHASEPlus(backbone, in_channels, num_frame, num_point, num_entity, c1, c2,
                              centred=centred, masked=masked, hierarchical=hierarchical, tau=tau,
                              stream="joint", kappa_lo=kappa_lo, kappa_hi=kappa_hi,
                              kappa_floor=kappa_floor)
        self.lam_rbr, self.lam_sd = float(lam_rbr), float(lam_sd)
        if self.lam_sd > 0:
            assert eps is not None and eps > 0, "pass a FROZEN eps (calibrated once on training data)"
            self.sd = SDMPDLoss(eps=float(eps), n_per_sample=sd_points, num_pairs=sd_pairs)
        else:
            self.sd = None

    def forward(self, x):
        if not self.training:
            return self.core(x)
        out, aux = self.core(x)
        loss = self.lam_rbr * aux["l_rbr"]
        if self.sd is not None:
            l_sd, _ = self.sd(aux["x_shift"], aux["pmask"])
            loss = loss + self.lam_sd * l_sd
        out = out if isinstance(out, tuple) else (out,)
        return out + (loss.reshape(1),)
