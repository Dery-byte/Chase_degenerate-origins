"""
probe_chase_v2.py -- follow-up controls after the first CAD probe.
Needs probe_chase.py in the same folder (reuses its loader).

  A. Which point is the origin?   (N_eff = 1 => alpha is one-hot)
     frame / joint / slot of argmax alpha, whether it is padding, whether it is
     an extreme point (min or max of x or y over the valid points).
  B. Translation control           (is the accuracy drop caused by padding, G3?)
     valid-only  : translate real joints, padding stays 0          (as in v1)
     all-points  : translate real joints AND padding together
     If CHASE's origin is equivariant, all-points must give ~no change.
  C. Permutation decomposition     (CHASE vs backbone order dependence, G4)
     full          : model(x permuted)
     backbone_only : same origin as clean, entities permuted after the shift
     chase_only    : origin from the permuted input, entities kept in clean order
  D. Origin replacement            (how much does the trained model rely on the
     selected point?)  origin := S2CoM centroid of the valid points.

  python probe_chase_v2.py --config cad_probe.yaml --weights cad_chas.pt --out probe_cad_v2
"""
import argparse
import json
import os

import numpy as np
import torch

from probe_chase import hull, load, logits_of, valid_mask


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="probe_v2")
    ap.add_argument("--sigmas", default="0.5,1,2")
    ap.add_argument("--perm_trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, loader = load(a.config, a.weights, dev, 0)
    bb = model.backbone
    sig = [float(s) for s in a.sigmas.split(",")]

    n = 0
    clean_ok = 0
    sel_t, sel_v, sel_m, sel_pad, sel_ext = [], [], [], [], []
    tr = {s: {k: [0, 0] for k in ("valid_only", "all_points")} for s in sig}   # [correct, flips]
    pe = {k: [0, 0] for k in ("full", "backbone_only", "chase_only")}
    pe_n = 0
    rep = [0, 0]

    for batch in loader:
        x = batch[0].float().to(dev); y = batch[1].long().to(dev)
        N, C, T, V, M = x.shape
        keep = valid_mask(x).flatten(1).any(1)                       # drop all-empty samples
        x, y = x[keep], y[keep]
        N = x.shape[0]
        if N == 0:
            continue
        pm = valid_mask(x)
        alpha, ps = hull(model, x)
        pred = logits_of(model, x).argmax(1)
        clean_ok += (pred == y).sum().item()

        # ---- A. selected point
        u = alpha.argmax(1)
        t_i, rem = u // (V * M), u % (V * M)
        v_i, m_i = rem // M, rem % M
        sel_t += t_i.tolist(); sel_v += v_i.tolist(); sel_m += m_i.tolist()
        sel_pad += (~pm.flatten(1).gather(1, u[:, None])[:, 0]).tolist()
        big = torch.finfo(x.dtype).max
        xf = x.flatten(2)                                           # (N,C,U)
        vm = pm.flatten(1).unsqueeze(1)
        mx = xf.masked_fill(~vm, -big).amax(2); mn = xf.masked_fill(~vm, big).amin(2)
        sp = xf.gather(2, u.view(N, 1, 1).expand(N, C, 1))[..., 0]
        sel_ext += ((sp == mx) | (sp == mn)).any(1).tolist()

        # ---- B. translation control
        cnt = pm.flatten(1).sum(1).clamp_min(1)
        cen = (x * pm.unsqueeze(1)).flatten(2).sum(2) / cnt.unsqueeze(1)
        scale = torch.sqrt((((x - cen.view(N, C, 1, 1, 1)) ** 2).sum(1) * pm).flatten(1).sum(1) / cnt)
        for s in sig:
            t = (torch.randn(N, C, device=dev) * scale.view(N, 1) * s).view(N, C, 1, 1, 1)
            for k, xt in (("valid_only", x + t * pm.unsqueeze(1)), ("all_points", x + t)):
                p = logits_of(model, xt).argmax(1)
                tr[s][k][0] += (p == y).sum().item(); tr[s][k][1] += (p != pred).sum().item()

        # ---- C. permutation decomposition
        xs = x - ps.view(N, C, 1, 1, 1)                              # wrapper's input to backbone
        for _ in range(a.perm_trials):
            perm = torch.arange(M - 1, -1, -1) if M == 2 else torch.randperm(M)
            if M > 2 and torch.equal(perm, torch.arange(M)):
                perm = torch.roll(perm, 1)
            perm = perm.to(dev); inv = torch.argsort(perm)
            xp = x[..., perm]
            _, pp = hull(model, xp)
            outs = {
                "full": logits_of(model, xp),
                "backbone_only": logits_of(bb, xs[..., perm]),
                "chase_only": logits_of(bb, (xp - pp.view(N, C, 1, 1, 1))[..., inv]),
            }
            for k, o in outs.items():
                p = o.argmax(1)
                pe[k][0] += (p == y).sum().item(); pe[k][1] += (p != pred).sum().item()
            pe_n += N

        # ---- D. origin := S2CoM
        p = logits_of(bb, x - cen.view(N, C, 1, 1, 1)).argmax(1)
        rep[0] += (p == y).sum().item(); rep[1] += (p != pred).sum().item()
        n += N

    pct = lambda c: round(100 * c / n, 2)
    hist = lambda v, K: np.bincount(np.array(v), minlength=K).tolist()
    s = dict(
        n_samples=n, acc_clean=pct(clean_ok),
        A_selected_point=dict(
            frac_on_padding=round(float(np.mean(sel_pad)), 4),
            frac_extreme_xy=round(float(np.mean(sel_ext)), 4),
            joint_hist=hist(sel_v, V), frame_hist=hist(sel_t, T), slot_hist=hist(sel_m, M)),
        B_translation={str(k): {m: dict(acc=pct(v[0]), flip_rate=pct(v[1])) for m, v in d.items()}
                       for k, d in tr.items()},
        C_permutation={k: dict(acc=round(100 * v[0] / pe_n, 2), flip_rate=round(100 * v[1] / pe_n, 2))
                       for k, v in pe.items()},
        D_origin_s2com=dict(acc=pct(rep[0]), flip_rate=pct(rep[1])))
    json.dump(s, open(os.path.join(a.out, "summary_v2.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
