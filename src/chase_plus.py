"""
chase_plus.py -- reference implementation of the Paper C corrections to CHASE
(Wen et al., NeurIPS 2024; https://github.com/Necolizer/CHASE, NeurIPS24/).

  C1  PBR    : Plucker bone stream computed on CHASE-shifted joints
  C2  RBR    : centred + masked CLB (exact translation equivariance, padding-safe),
               two-sided Renyi-2 support band, optional entity-permutation-invariant
               hierarchical weights
  C3  SD-MPD : debiased Sinkhorn divergence between batch-pooled entity marginals,
               with the shift-invariant mean floor reported separately

Drop-in for the original `CHASWrapper` (model/Wrapper/chas.py).  Input layout
is the CHASE/CTR-GCN layout  x : (N, C, T, V, M)  with C in {2, 3}.

    python chase_plus.py          # runs the self-tests (CPU, ~10 s)

Requires: torch >= 1.9, einops is NOT required.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Skeleton parent tables (0-based).  The root points to itself.
# VERIFY against the joint order of your own data loader before use.
# ----------------------------------------------------------------------------
# NTU RGB+D 25 joints; root = 20 (spine centre), same as CTR-GCN's ntu_pairs.
NTU25_PARENTS = [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14,
                 0, 16, 17, 18, 20, 22, 7, 24, 11]
# COCO-17 (CAD / VD 2-D poses); root = 0 (nose).
COCO17_PARENTS = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
# 21-joint hand (H2O / Assembly101); root = 0 (wrist), 4 joints per finger.
HAND21_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


# ----------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------
def point_mask(x: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """(N,C,T,V,M) -> bool (N,T,V,M).  A point is padding iff all its
    coordinates are exactly zero (CHASE CAD/VD feeders pad absent persons and
    agent-dropout persons with zeros; NTU pads absent bodies with zeros)."""
    return x.abs().amax(dim=1) > tol


def entity_mask(pm: torch.Tensor) -> torch.Tensor:
    """(N,T,V,M) -> bool (N,M): entity has at least one valid point."""
    return pm.flatten(1, 2).any(dim=1)


def masked_mean(x: torch.Tensor, m: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Mean of x over dims, counting only positions where m is True.
    x : (N,C,...) ; m broadcastable to x without the channel axis."""
    w = m.unsqueeze(1).to(x.dtype)
    return (x * w).sum(dim=dims) / w.sum(dim=dims).clamp_min(1.0)


# ----------------------------------------------------------------------------
# C1  Plucker bones
# ----------------------------------------------------------------------------
def vector_bones(x: torch.Tensor, parents: Sequence[int]) -> torch.Tensor:
    """CHASE Def. 4 bone b_ij = p_i - p_j.  Shift-blind (CHASE App. A.2)."""
    return x - x[:, :, :, list(parents)]


def khop_bones(x: torch.Tensor, parents: Sequence[int]) -> torch.Tensor:
    """CHASE Def. 5 with k = 1 (InfoGCN input): like vector_bones but the root
    keeps its joint coordinate, so it is shift-sensitive at ONE joint only."""
    b = vector_bones(x, parents)
    roots = [j for j, p in enumerate(parents) if p == j]
    b[:, :, :, roots] = x[:, :, :, roots]
    return b


def plucker_bones(x: torch.Tensor, parents: Sequence[int], normalise: bool = True,
                  eps: float = 1e-6, pm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Plucker line of every bone, Eq. (18):
         d_ij = p_i - p_j ,   m_ij = p_j x d_ij       (C = 3  -> 6 channels)
         m_ij = p_j,x d_y - p_j,y d_x                 (C = 2  -> 3 channels)
    normalise=True returns m / ||d|| (distance-scaled moment, units of length),
    whose norm is the distance from the origin to the bone line and is bounded
    by diam(X) when the origin lies in conv(X) (Prop. 8)."""
    par = list(parents)
    pj = x[:, :, :, par]
    d = x - pj
    C = x.shape[1]
    if C == 3:
        m = torch.cross(pj, d, dim=1)
    elif C == 2:
        m = (pj[:, 0] * d[:, 1] - pj[:, 1] * d[:, 0]).unsqueeze(1)
    else:
        raise ValueError("C must be 2 or 3")
    if normalise:
        # smooth norm: no NaN gradient at d = 0 (root self-bone, padding)
        m = m / torch.sqrt((d ** 2).sum(dim=1, keepdim=True) + eps ** 2)
    out = torch.cat([d, m], dim=1)
    if pm is not None:  # a bone is valid only if both endpoints are valid
        valid = pm & pm[:, :, par]
        out = out * valid.unsqueeze(1).to(out.dtype)
    return out


# ----------------------------------------------------------------------------
# C2  Renyi-2 support band
# ----------------------------------------------------------------------------
def renyi2(alpha: torch.Tensor) -> torch.Tensor:
    """H2(alpha) = -log sum alpha^2  over the last axis; N_eff = exp(H2)."""
    return -torch.log((alpha ** 2).sum(dim=-1).clamp_min(1e-30))


def neff_floor(n: int, c: float) -> float:
    """Exact minimum of N_eff = (sum e^l)^2 / sum e^{2l} over logits l in [-c, c]^n.
    The minimum is attained at a vertex (k logits at +c, n-k at -c), Prop. 17."""
    hi, lo, best = math.exp(c), math.exp(-c), float(n)
    for k in range(1, n):
        z = k * hi + (n - k) * lo
        best = min(best, z * z / (k * hi * hi + (n - k) * lo * lo))
    return best


def logit_bound_for_neff(n: int, kappa: float) -> float:
    """Smallest-range bound c such that every softmax over n logits in [-c, c] has N_eff >= kappa."""
    assert 1 < kappa < n
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if neff_floor(n, mid) > kappa:
            lo = mid
        else:
            hi = mid
    return lo


def rbr_loss(alpha: torch.Tensor, kappa_lo: float, kappa_hi: Optional[float]) -> torch.Tensor:
    """Eq. (19'):  ReLU(log k_lo - H2) + ReLU(H2 - log k_hi), batch mean."""
    h2 = renyi2(alpha)
    loss = F.relu(math.log(kappa_lo) - h2)
    if kappa_hi is not None:
        loss = loss + F.relu(h2 - math.log(kappa_hi))
    return loss.mean()


# ----------------------------------------------------------------------------
# CHASE+ wrapper
# ----------------------------------------------------------------------------
class CHASEPlus(nn.Module):
    """CHASE with the C1/C2 corrections.

    Args
      backbone      single-entity backbone taking (N, C_out, T, V, M)
      in_channels   C (2 or 3)
      num_frame, num_point, num_entity   T, V, M
      c1, c2        CLB widths (CHASE default 64, 8)
      centred       feed the CLB with x - mean(x)   (Prop. 10; exact
                    translation equivariance of p*).  False = original CHASE.
      masked        exclude padded points from the hull / pooling (Prop. 13)
      hierarchical  alpha_{e,u} = alpha_e * alpha_{u|e} with a head shared across
                    entities (Prop. 12; entity-permutation invariance).  Use for
                    unordered entities (NTU, CAD); keep False for ordered ones
                    (H2O, ASB101, VD) to preserve CHASE's orderliness.
      tau           softmax temperature
      stream        'joint' | 'bone' | 'khop' | 'plucker'  -- transform applied
                    AFTER the shift; 'plucker' is C1.
      parents       parent table for the bone streams
      kappa_lo/hi   Renyi-2 band (C2) as a soft hinge.  kappa_lo=1, kappa_hi=None disables it.
      kappa_floor   HARD lower bound N_eff >= kappa_floor (per entity if hierarchical) via
                    bounded logits c*tanh(z/c) (Prop. 17).  Use this, not kappa_lo, against
                    saturation: the hinge cannot act once the softmax is one-hot.

    forward(x) -> logits                         (eval)
               -> (logits, aux: dict)            (train)
      aux['alpha'] (N,U), aux['pstar'] (N,C), aux['x_shift'] (N,C,T,V,M) joints,
      aux['pmask'] (N,T,V,M), aux['emask'] (N,M), aux['neff'] (N,), aux['l_rbr'].
    """

    def __init__(self, backbone: nn.Module, in_channels: int = 3, num_frame: int = 64,
                 num_point: int = 25, num_entity: int = 2, c1: int = 64, c2: int = 8,
                 centred: bool = True, masked: bool = True, hierarchical: bool = False,
                 tau: float = 1.0, stream: str = "joint", parents: Optional[Sequence[int]] = None,
                 kappa_lo: float = 1.0, kappa_hi: Optional[float] = None,
                 kappa_floor: Optional[float] = None):
        super().__init__()
        assert stream in ("joint", "bone", "khop", "plucker")
        if stream != "joint" and parents is None:
            raise ValueError("bone streams need a parent table")
        self.C, self.T, self.V, self.M = in_channels, num_frame, num_point, num_entity
        self.centred, self.masked, self.hier = centred, masked, hierarchical
        self.tau, self.stream, self.parents = tau, stream, parents
        self.kappa_lo, self.kappa_hi = kappa_lo, kappa_hi
        # Hard Renyi floor by construction (Prop. 17): bounded logits c*tanh(z/c).
        # A soft hinge on H2 cannot undo saturation: the softmax gradient vanishes at a one-hot alpha.
        n_soft = num_frame * num_point if hierarchical else num_frame * num_point * num_entity
        self.logit_c = logit_bound_for_neff(n_soft, kappa_floor) if kappa_floor else None
        self.backbone = backbone

        self.embed = nn.Conv3d(in_channels, c1, kernel_size=1)          # W1, b
        if hierarchical:
            # shared per-entity head: point logits (T*V) and one entity logit
            self.point_head = nn.Sequential(nn.Linear(c1, c2, bias=False), nn.ReLU(inplace=True),
                                            nn.Linear(c2, num_frame * num_point, bias=False))
            self.entity_head = nn.Linear(c1, 1)
        else:
            # CHASE Eq. 8:  W3 delta(W2 phi(W1 X + b)),  U = T*V*M outputs
            self.point_head = nn.Sequential(nn.Linear(c1, c2, bias=False), nn.ReLU(inplace=True),
                                            nn.Linear(c2, num_frame * num_point * num_entity, bias=False))

    # -- p* -------------------------------------------------------------------
    def _weights(self, x: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Returns alpha over U = T*V*M points in (t v m) order, shape (N, U)."""
        N, C, T, V, M = x.shape
        if self.centred:
            mu = masked_mean(x, pm, dims=(2, 3, 4)) if self.masked else x.mean(dim=(2, 3, 4))
            xin = x - mu.view(N, C, 1, 1, 1)
        else:
            xin = x
        h = self.embed(xin)                                              # (N,c1,T,V,M)
        neg = torch.finfo(h.dtype).min

        if not self.hier:
            g = masked_mean(h, pm, (2, 3, 4)) if self.masked else h.mean(dim=(2, 3, 4))
            logits = self.point_head(g).view(N, T, V, M) / self.tau
            if self.logit_c is not None:
                logits = self.logit_c * torch.tanh(logits / self.logit_c)
            if self.masked:
                logits = logits.masked_fill(~pm, neg)
            return logits.reshape(N, -1).softmax(dim=-1)

        # hierarchical: per-entity features with shared heads
        em = entity_mask(pm)                                             # (N,M)
        ge = (h * pm.unsqueeze(1)).sum((2, 3)) / pm.sum((1, 2)).clamp_min(1).unsqueeze(1)  # (N,c1,M)
        ge = ge.transpose(1, 2)                                          # (N,M,c1)
        lp = self.point_head(ge).view(N, M, T, V) / self.tau             # (N,M,T,V)
        if self.logit_c is not None:
            lp = self.logit_c * torch.tanh(lp / self.logit_c)
        lp = lp.masked_fill(~pm.permute(0, 3, 1, 2), neg)
        a_u = lp.reshape(N, M, -1).softmax(-1).view(N, M, T, V)
        le = self.entity_head(ge).squeeze(-1).masked_fill(~em, neg)      # (N,M)
        a_e = le.softmax(-1)
        alpha = a_e.view(N, M, 1, 1) * a_u                               # (N,M,T,V)
        return alpha.permute(0, 2, 3, 1).reshape(N, -1)                  # (t v m)

    def shift(self, x: torch.Tensor):
        N, C, T, V, M = x.shape
        pm = point_mask(x) if self.masked else torch.ones(N, T, V, M, dtype=torch.bool, device=x.device)
        alpha = self._weights(x, pm)                                     # (N,U)
        pstar = torch.einsum("nu,ncu->nc", alpha, x.reshape(N, C, -1))   # Eq. 3
        xs = x - pstar.view(N, C, 1, 1, 1)                               # Eq. 1
        if self.masked:
            xs = xs * pm.unsqueeze(1).to(xs.dtype)                       # padding stays padding
        return xs, pstar, alpha, pm

    def transform(self, xs: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        if self.stream == "joint":
            return xs
        if self.stream == "bone":
            return vector_bones(xs, self.parents) * pm.unsqueeze(1)
        if self.stream == "khop":
            return khop_bones(xs, self.parents) * pm.unsqueeze(1)
        return plucker_bones(xs, self.parents, normalise=True, pm=pm)

    def forward(self, x: torch.Tensor):
        xs, pstar, alpha, pm = self.shift(x)
        out = self.backbone(self.transform(xs, pm))
        if not self.training:
            return out[0] if isinstance(out, tuple) else out
        aux = dict(alpha=alpha, pstar=pstar, x_shift=xs, pmask=pm, emask=entity_mask(pm),
                   neff=torch.exp(renyi2(alpha)).detach(),
                   l_rbr=rbr_loss(alpha, self.kappa_lo, self.kappa_hi))
        return out, aux


# ----------------------------------------------------------------------------
# C3  Debiased Sinkhorn divergence
# ----------------------------------------------------------------------------
def _sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2).sum(-1)       # c = 1/2 ||x-y||^2


def _softmin(eps: float, C: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """-eps * log sum_j exp(h_j - C_ij / eps)  over the last axis."""
    return -eps * torch.logsumexp(h.unsqueeze(-2) - C / eps, dim=-1)


def ot_eps(a: torch.Tensor, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor,
           eps: float, n_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """Entropic OT_eps(a,x; b,y) with KL(Pi || a x b), Eq. (20), log domain.
    Potentials are solved without autograd; the value is then re-evaluated with
    the dual objective at the frozen potentials, whose gradient w.r.t. x, y is
    exactly sum_j Pi_ij grad C_ij (envelope theorem)."""
    la = torch.log(a.clamp_min(1e-38)).masked_fill(a <= 0, -1e30)
    lb = torch.log(b.clamp_min(1e-38)).masked_fill(b <= 0, -1e30)
    with torch.no_grad():
        Cxy = _sqdist(x, y)
        f = torch.zeros_like(a); g = torch.zeros_like(b)
        for _ in range(n_iter):
            f_new = _softmin(eps, Cxy, lb + g / eps)
            g = _softmin(eps, Cxy.transpose(-1, -2), la + f_new / eps)
            if (f_new - f).abs().max() < tol:
                f = f_new
                break
            f = f_new
    Cxy = _sqdist(x, y)                                                   # with grad
    f, g = f.detach(), g.detach()
    plan = torch.exp((f.unsqueeze(-1) + g.unsqueeze(-2) - Cxy) / eps + la.unsqueeze(-1) + lb.unsqueeze(-2))
    af = torch.where(a > 0, a * f, torch.zeros_like(f))
    bg = torch.where(b > 0, b * g, torch.zeros_like(g))
    return af.sum(-1) + bg.sum(-1) - eps * (plan.sum((-1, -2)) - 1.0)


def sinkhorn_divergence(a, x, b, y, eps: float, n_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """Eq. (21): S_eps = OT(a,b) - 1/2 OT(a,a) - 1/2 OT(b,b)  (Feydy et al. 2019)."""
    return ot_eps(a, x, b, y, eps, n_iter, tol) - 0.5 * ot_eps(a, x, a, x, eps, n_iter, tol) \
        - 0.5 * ot_eps(b, y, b, y, eps, n_iter, tol)


def calibrate_eps(x_shift: torch.Tensor, pm: torch.Tensor, frac: float = 0.05,
                  n: int = 2048) -> float:
    """Pick eps ONCE (e.g. on a few training batches) as frac * median cost of
    the centred pooled cloud, then freeze it.  eps must not depend on the
    current batch, or Prop. 15 (mean/shape split) no longer holds."""
    N, C = x_shift.shape[:2]
    pts = x_shift.permute(0, 2, 3, 4, 1)[pm]                              # (P,C)
    pts = pts[torch.randperm(len(pts))[:n]]
    pts = pts - pts.mean(0)
    return float(frac * _sqdist(pts, pts).median())


def _pooled_measure(xs: torch.Tensor, pm: torch.Tensor, e: int, n_per_sample: int,
                    support: str, generator: Optional[torch.Generator] = None):
    """Batch-pooled empirical marginal of entity e:  (1/N n) sum_s sum_k delta(x_e,k^(s) - p*^(s)).
    support='points'  : n random valid points per sample   (default)
    support='corners' : per-sample channel max & min over (T,V), as in the
                        released CHASE code -- for a kernel-vs-OT ablation."""
    N, C, T, V, M = xs.shape
    pts_e = xs[..., e].reshape(N, C, T * V).transpose(1, 2)               # (N,TV,C)
    m_e = pm[..., e].reshape(N, T * V)                                    # (N,TV)
    valid_s = m_e.any(1)                                                  # sample has entity e
    if support == "corners":
        big = torch.finfo(xs.dtype).max
        mx = pts_e.masked_fill(~m_e.unsqueeze(-1), -big).amax(1)
        mn = pts_e.masked_fill(~m_e.unsqueeze(-1), big).amin(1)
        P = torch.stack([mx, mn], 1)                                      # (N,2,C)
    else:
        prob = m_e.to(xs.dtype) + (~valid_s).unsqueeze(1).to(xs.dtype)    # avoid all-zero rows
        idx = torch.multinomial(prob, n_per_sample, replacement=True, generator=generator)
        P = torch.gather(pts_e, 1, idx.unsqueeze(-1).expand(-1, -1, C))   # (N,n,C)
    P = torch.where(valid_s.view(N, 1, 1), P, torch.zeros_like(P))      # keep padding finite
    return P, valid_s                                                     # (N,k,C), (N,)


def _pair_weights(vi: torch.Tensor, vj: torch.Tensor, k: int, dtype) -> torch.Tensor:
    """Uniform weight on samples where BOTH entities exist.  Equal per-sample
    weight for the two entities is what makes p*^(s) cancel in the mean gap
    (Prop. 15 corollary); samples with only one entity are dropped."""
    v = (vi & vj).unsqueeze(1).expand(-1, k).to(dtype).reshape(-1)
    return v / v.sum().clamp_min(1e-12)


class SDMPDLoss(nn.Module):
    """Eq. (22): mean debiased Sinkhorn divergence over entity pairs of the
    batch-pooled shifted marginals.  E <= all_pairs_max: all valid pairs;
    otherwise num_pairs random pairs.  pair_mask (M,M) bool excludes
    heterogeneous pairs (e.g. ball-player in VD) if desired.

    Returns (loss, stats) where stats holds the shift-invariant mean floor
    1/2 ||mu_a - mu_b||^2 (Prop. 15) and the reducible part S - floor."""

    def __init__(self, eps: float, n_per_sample: int = 8, support: str = "points",
                 all_pairs_max: int = 4, num_pairs: int = 4, n_iter: int = 100,
                 pair_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps, self.n, self.support = eps, n_per_sample, support
        self.all_pairs_max, self.num_pairs, self.n_iter = all_pairs_max, num_pairs, n_iter
        self.pair_mask = pair_mask

    def forward(self, x_shift: torch.Tensor, pm: torch.Tensor):
        M = x_shift.shape[-1]
        pairs = [(i, j) for i in range(M) for j in range(i + 1, M)
                 if self.pair_mask is None or bool(self.pair_mask[i, j])]
        if M > self.all_pairs_max and len(pairs) > self.num_pairs:
            sel = torch.randperm(len(pairs))[: self.num_pairs].tolist()
            pairs = [pairs[k] for k in sel]
        meas = {}
        losses, floors = [], []
        for i, j in pairs:
            for e in (i, j):
                if e not in meas:
                    meas[e] = _pooled_measure(x_shift, pm, e, self.n, self.support)
            (Pi, vi), (Pj, vj) = meas[i], meas[j]
            if not bool((vi & vj).any()):
                continue
            k = Pi.shape[1]
            wi = wj = _pair_weights(vi, vj, k, Pi.dtype)
            Pi, Pj = Pi.reshape(-1, Pi.shape[-1]), Pj.reshape(-1, Pj.shape[-1])
            s = sinkhorn_divergence(wi, Pi, wj, Pj, self.eps, self.n_iter)
            losses.append(s)
            with torch.no_grad():
                floors.append(0.5 * ((wi @ Pi - wj @ Pj) ** 2).sum())
        if not losses:
            z = x_shift.sum() * 0.0
            return z, dict(floor=z.detach(), reducible=z.detach(), n_pairs=0)
        loss = torch.stack(losses).mean()
        floor = torch.stack(floors).mean()
        return loss, dict(floor=floor, reducible=(loss.detach() - floor), n_pairs=len(losses))


# ----------------------------------------------------------------------------
# Faithful re-implementation of the released MPMMD (baseline for ablations)
# NeurIPS24/model/Wrapper/chas.py + utils/loss.py
# ----------------------------------------------------------------------------
def repo_mmd(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0,
             kernel_num: int = 5) -> torch.Tensor:
    total = torch.cat([source, target], 0)
    n = total.shape[0]
    L2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bw = L2.detach().sum() / (n * n - n) / kernel_mul ** (kernel_num // 2)
    K = sum(torch.exp(-L2 / (bw * kernel_mul ** i)) for i in range(kernel_num))
    k = source.shape[0]
    return K[:k, :k].mean() + K[k:, k:].mean() - K[:k, k:].mean() - K[k:, :k].mean()


def repo_mpmmd(x_shift: torch.Tensor) -> torch.Tensor:
    """One random entity pair per batch; per-sample max/min corners; RBF-MMD."""
    N, C, T, V, M = x_shift.shape
    i, j = sorted(torch.randperm(M)[:2].tolist())
    xt = x_shift.permute(0, 2, 3, 1, 4).reshape(N, T * V, C, M)
    corners = torch.cat([xt.amax(1, keepdim=True), xt.amin(1, keepdim=True)], 1)  # (N,2,C,M)
    return repo_mmd(corners[..., i].reshape(-1, C), corners[..., j].reshape(-1, C))


# ----------------------------------------------------------------------------
# Training-step helper
# ----------------------------------------------------------------------------
def chase_plus_loss(logits, target, aux, sd_loss: Optional[SDMPDLoss],
                    lam_sd: float = 0.1, lam_rbr: float = 0.1,
                    cls_loss: Optional[nn.Module] = None):
    """L = L_cls + lam_sd * L_sd + lam_rbr * L_rbr   (lam_sd = 0.1 as CHASE's lambda)."""
    cls_loss = cls_loss or nn.CrossEntropyLoss()
    L = cls_loss(logits, target)
    logs = dict(cls=L.detach(), neff_median=aux["neff"].median())
    if sd_loss is not None and lam_sd > 0:
        l_sd, st = sd_loss(aux["x_shift"], aux["pmask"])
        L = L + lam_sd * l_sd
        logs.update(sd=l_sd.detach(), sd_floor=st["floor"], sd_reducible=st["reducible"])
    if lam_rbr > 0:
        L = L + lam_rbr * aux["l_rbr"]
        logs.update(rbr=aux["l_rbr"].detach())
    return L, logs


# ----------------------------------------------------------------------------
# Self-tests (mirror verify_numpy.py on the actual torch code)
# ----------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    dt = torch.float64
    ok_all = []

    def check(name, ok, info=""):
        ok_all.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")

    class Mean(nn.Module):          # trivial backbone: (N,C,T,V,M) -> (N,K)
        def __init__(s, c, k=5): super().__init__(); s.fc = nn.Linear(c, k)
        def forward(s, z): return s.fc(z.mean((2, 3, 4)))

    N, C, T, V, M = 4, 3, 16, 25, 2
    x = torch.randn(N, C, T, V, M, dtype=dt) + 5 * torch.randn(N, C, 1, 1, M, dtype=dt)
    t = torch.tensor([3.0, -7.0, 11.0], dtype=dt).view(1, 3, 1, 1, 1)

    for hier in (False, True):
        for centred in (False, True):
            net = CHASEPlus(Mean(3), 3, T, V, M, centred=centred, hierarchical=hier).to(dt).eval()
            xs1, p1, *_ = net.shift(x)
            xs2, p2, *_ = net.shift(x + t)
            err = (xs1 - xs2).abs().max().item()
            if centred:
                check(f"Prop 10  centred CLB => shifted input translation-invariant (hier={hier})", err < 1e-9, f"err={err:.1e}")
            else:
                check(f"Prop 10  original CLB is not translation-invariant (hier={hier})", err > 1e-4, f"err={err:.2e}")

    # Prop 12 entity permutation invariance
    net = CHASEPlus(Mean(3), 3, T, V, M, hierarchical=True).to(dt).eval()
    _, pa, *_ = net.shift(x); _, pb, *_ = net.shift(x.flip(-1))
    check("Prop 12  hierarchical p* invariant to entity order", torch.allclose(pa, pb))
    net0 = CHASEPlus(Mean(3), 3, T, V, M, hierarchical=False).to(dt).eval()
    _, pa, *_ = net0.shift(x); _, pb, *_ = net0.shift(x.flip(-1))
    check("Prop 12  flat CHASE p* depends on entity order", not torch.allclose(pa, pb))

    # p* inside the hull: convex weights
    _, p, alpha, _ = net.shift(x)
    check("Prop 1   alpha is a probability vector", torch.allclose(alpha.sum(-1), torch.ones(N, dtype=dt)) and (alpha >= 0).all())

    # Prop 13 masking: padded entity gets zero weight and stays zero
    xp = x.clone(); xp[1, ..., 1] = 0
    netm = CHASEPlus(Mean(3), 3, T, V, M, masked=True).to(dt).eval()
    xs, p, alpha, pm = netm.shift(xp)
    a1 = alpha[1].view(T, V, M)[..., 1].sum().item()
    check("Prop 13  padded entity has zero hull weight", a1 == 0.0, f"w={a1}")
    check("Prop 13  padded points remain zero after shift", xs[1, ..., 1].abs().max().item() == 0.0)
    netu = CHASEPlus(Mean(3), 3, T, V, M, masked=False).to(dt).eval()
    xs_u, *_ = netu.shift(xp)
    check("Prop 13  unmasked CHASE turns padding into fake joints at -p*", xs_u[1, ..., 1].abs().max().item() > 0)

    # C1 Plucker
    par = NTU25_PARENTS
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=dt)); R = q * torch.sign(torch.diag(r))
    if torch.det(R) < 0: R[:, 0] *= -1
    pb = plucker_bones(x, par, normalise=False)
    xr = torch.einsum("ij,njtvm->nitvm", R, x) + t
    pb2 = plucker_bones(xr, par, normalise=False)
    d, m = pb[:, :3], pb[:, 3:]
    d2, m2 = pb2[:, :3], pb2[:, 3:]
    Rd = torch.einsum("ij,njtvm->nitvm", R, d)
    pred = torch.einsum("ij,njtvm->nitvm", R, m) + torch.cross(t.expand_as(Rd), Rd, dim=1)
    check("Prop 7   d'=Rd, m'=Rm + t x Rd", torch.allclose(d2, Rd) and torch.allclose(m2, pred))
    check("Prop 7   d.m = 0", (d2 * m2).sum(1).abs().max().item() < 1e-8)
    check("Prop 7   vector bones shift-blind", torch.allclose(vector_bones(x + t, par), vector_bones(x, par)))
    kh = khop_bones(x + t, par) - khop_bones(x, par)
    check("Def 5    1-hop bones shift-sensitive only at the root", (kh.abs().amax((0, 1, 2, 4)) > 0).sum().item() == 1)
    xs, p, alpha, pm = netm.shift(x)
    diam = torch.cdist(x[0].reshape(3, -1).T, x[0].reshape(3, -1).T).max()
    mh = plucker_bones(xs, par)[0, 3:].norm(dim=0).max()
    check("Prop 8   ||m/||d|| || <= diam under convex-hull origin", mh <= diam + 1e-9, f"{mh:.2f} <= {diam:.2f}")
    x2d = torch.randn(N, 2, T, 17, M, dtype=dt)
    check("C1 2-D   Plucker has 3 channels", plucker_bones(x2d, COCO17_PARENTS).shape[1] == 3)

    # C2 RBR
    a = torch.softmax(torch.randn(1000, 300, dtype=dt) * torch.rand(1000, 1, dtype=dt) * 4, -1)
    ne = torch.exp(renyi2(a)); k = 20.0
    sel = ne >= k
    assert sel.any() and (~sel).any()
    check("Prop 9   N_eff >= k => max alpha <= 1/sqrt(k)", (a[sel].amax(-1) <= 1 / math.sqrt(k) + 1e-12).all())
    check("Prop 9   L_rbr = 0 iff k_lo <= N_eff <= k_hi",
          rbr_loss(torch.full((1, 100), 0.01, dtype=dt), 10, 200).item() == 0 and
          rbr_loss(torch.full((1, 100), 0.01, dtype=dt), 10, 50).item() > 0)

    # Prop 17: bounded logits give a hard N_eff floor even for exploding logits
    netb = CHASEPlus(Mean(3), 3, T, V, M, hierarchical=True, kappa_floor=8.0).to(dt).eval()
    with torch.no_grad():
        for prm in netb.point_head.parameters():
            prm.mul_(1e4)                                        # force saturation
    _, _, ab, _ = netb.shift(x)
    ne_b = torch.exp(renyi2(ab)).min().item()
    check("Prop 17  bounded logits: N_eff >= kappa_floor under exploding logits", ne_b >= 8.0 - 1e-6, f"min N_eff={ne_b:.2f}")
    check("Prop 17  vertex formula matches brute force", abs(neff_floor(5, 1.0) - min(
        (k * math.e + (5 - k) / math.e) ** 2 / (k * math.e ** 2 + (5 - k) / math.e ** 2) for k in range(1, 5))) < 1e-12)

    # C3 Sinkhorn divergence
    eps = 0.1
    xa = torch.randn(60, 3, dtype=dt); yb = torch.randn(50, 3, dtype=dt) * 1.3 + torch.tensor([2.0, -1, .5], dtype=dt)
    wa = torch.rand(60, dtype=dt); wa /= wa.sum(); wb = torch.rand(50, dtype=dt); wb /= wb.sum()
    S = sinkhorn_divergence(wa, xa, wb, yb, eps, 3000, 1e-14)
    check("Prop 14  S >= 0, S(P,P)=0", S.item() >= -1e-8 and abs(sinkhorn_divergence(wa, xa, wa, xa, eps, 3000, 1e-14).item()) < 1e-7)
    mua, mub = wa @ xa, wb @ yb
    Sc = sinkhorn_divergence(wa, xa - mua, wb, yb - mub, eps, 3000, 1e-14)
    check("Prop 15  S = S(centred) + 1/2||dmu||^2", abs(S - Sc - 0.5 * ((mua - mub) ** 2).sum()).item() < 1e-6)
    ps = torch.randn(3, dtype=dt) * 4
    check("Prop 16  per-sample S blind to a common shift",
          abs(sinkhorn_divergence(wa, xa - ps, wb, yb - ps, eps, 3000, 1e-14) - S).item() < 1e-7)
    # gradient = finite differences
    xg = xa.clone().requires_grad_(True)
    sinkhorn_divergence(wa, xg, wb, yb, eps, 3000, 1e-14).backward()
    h = 1e-5; E_ = torch.zeros_like(xa); E_[3, 1] = h
    fd = (sinkhorn_divergence(wa, xa + E_, wb, yb, eps, 3000, 1e-14) - sinkhorn_divergence(wa, xa - E_, wb, yb, eps, 3000, 1e-14)) / (2 * h)
    check("Grad     envelope gradient matches finite differences", abs(fd - xg.grad[3, 1]).item() < 1e-5,
          f"fd={fd.item():.6f} ad={xg.grad[3,1].item():.6f}")

    # end-to-end training step
    net = CHASEPlus(Mean(6), 3, T, V, M, stream="plucker", parents=par, hierarchical=True,
                    kappa_lo=10, kappa_hi=400).to(dt).train()
    xs, *_ , pm = net.shift(x)
    eps_c = calibrate_eps(xs.detach(), pm)
    sd = SDMPDLoss(eps=eps_c, n_per_sample=8)
    out, aux = net(x)
    L, logs = chase_plus_loss(out, torch.randint(0, 5, (N,)), aux, sd)
    L.backward()
    gnorm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    check("E2E      loss finite, gradients reach the CLB", torch.isfinite(L).item() and gnorm > 0,
          f"L={L.item():.3f} floor={logs['sd_floor'].item():.3f} reducible={logs['sd_reducible'].item():.3f}")
    # padded entity + both supports: finite loss, mean floor invariant to p*
    xp = x.clone(); xp[2, ..., 1] = 0
    xs_p, _, _, pm_p = netm.shift(xp)
    for sup in ("points", "corners"):
        l, st = SDMPDLoss(eps=eps_c, support=sup)(xs_p, pm_p)
        check(f"C3       SD-MPD finite with padding (support={sup})", torch.isfinite(l).item(),
              f"S={l.item():.3f} floor={st['floor'].item():.3f}")
    g = torch.Generator().manual_seed(1)
    Pa, va = _pooled_measure(xp, pm_p, 0, 8, "points", g)
    g = torch.Generator().manual_seed(2)
    Pb, vb = _pooled_measure(xp, pm_p, 1, 8, "points", g)
    w = _pair_weights(va, vb, 8, dt)
    shift = torch.randn(N, 1, 1, dtype=dt) * 9 * torch.ones(1, 1, 3, dtype=dt)
    gap0 = w @ Pa.reshape(-1, 3) - w @ Pb.reshape(-1, 3)
    gap1 = w @ (Pa - shift).reshape(-1, 3) - w @ (Pb - shift).reshape(-1, 3)
    check("Prop 15  pooled mean gap invariant to per-sample p* (with padding)", torch.allclose(gap0, gap1))
    rm = repo_mpmmd(aux["x_shift"])
    check("Repo     MPMMD baseline runs", torch.isfinite(rm).item(), f"mpmmd={rm.item():.4f}")

    print(f"\n{sum(ok_all)}/{len(ok_all)} self-tests passed")
    return all(ok_all)


if __name__ == "__main__":
    _selftest()
