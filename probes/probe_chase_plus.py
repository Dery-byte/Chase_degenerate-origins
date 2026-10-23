"""
probe_chase_plus.py -- the E1/E2 diagnostics for a trained CHASE+ checkpoint.
Needs probe_chase.py (for the loader) in the same folder.

  python probe_chase_plus.py --config cad_chaseplus.yaml --weights runs/.../x.pt --out probe_plus_s1
"""
import argparse, json, os
import numpy as np
import torch
from probe_chase import load, logits_of, valid_mask


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="probe_plus"); ap.add_argument("--perm_trials", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, loader = load(a.config, a.weights, dev, 0)
    core = model.core
    n = ok = 0
    neff, padm = [], []
    tr = {s: {"valid_only": [0, 0], "all_points": [0, 0]} for s in (1.0, 2.0)}
    eq = []
    pe = [0, 0, 0]
    for batch in loader:
        x = batch[0].float().to(dev); y = batch[1].long().to(dev)
        keep = valid_mask(x).flatten(1).any(1); x, y = x[keep], y[keep]
        N, C, T, V, M = x.shape
        if N == 0:
            continue
        pm = valid_mask(x)
        _, ps, alpha, _ = core.shift(x)
        pred = logits_of(model, x).argmax(1); ok += (pred == y).sum().item()
        neff += (1.0 / (alpha ** 2).sum(1)).tolist()
        padm += (alpha * (~pm.flatten(1))).sum(1).tolist()
        cnt = pm.flatten(1).sum(1).clamp_min(1)
        cen = (x * pm.unsqueeze(1)).flatten(2).sum(2) / cnt.unsqueeze(1)
        s = torch.sqrt((((x - cen.view(N, C, 1, 1, 1)) ** 2).sum(1) * pm).flatten(1).sum(1) / cnt)
        for sg in tr:
            t = (torch.randn(N, C, device=dev) * s.view(N, 1) * sg)
            xv = x + t.view(N, C, 1, 1, 1) * pm.unsqueeze(1)
            _, pst, _, _ = core.shift(xv)
            eq += ((pst - ps - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-8)).tolist()
            for k, xt in (("valid_only", xv), ("all_points", x + t.view(N, C, 1, 1, 1))):
                p = logits_of(model, xt).argmax(1)
                tr[sg][k][0] += (p == y).sum().item(); tr[sg][k][1] += (p != pred).sum().item()
        for _ in range(a.perm_trials):
            perm = torch.arange(M - 1, -1, -1) if M == 2 else torch.randperm(M)
            p = logits_of(model, x[..., perm.to(dev)]).argmax(1)
            pe[0] += (p == y).sum().item(); pe[1] += (p != pred).sum().item(); pe[2] += N
        n += N
    neff = np.array(neff); padm = np.array(padm)
    q = lambda v: {k: float(np.quantile(v, p)) for k, p in (("p05", .05), ("median", .5), ("p95", .95))}
    out = dict(n_samples=n, acc_clean=round(100 * ok / n, 2), U=int(alpha.shape[1]),
               neff=q(neff), frac_neff_eq_1=float((neff < 1.5).mean()),
               frac_neff_ge_half_U=float((neff > 0.5 * alpha.shape[1]).mean()),
               pad_mass_max=float(padm.max()),
               equiv_err_median=float(np.median(eq)), equiv_err_p95=float(np.quantile(eq, .95)),
               translation={str(k): {m: dict(acc=round(100 * v[0] / n, 2), flip_rate=round(100 * v[1] / n, 2))
                                     for m, v in d.items()} for k, d in tr.items()},
               permutation=dict(acc=round(100 * pe[0] / pe[2], 2), flip_rate=round(100 * pe[1] / pe[2], 2)))
    json.dump(out, open(os.path.join(a.out, "summary_plus.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
