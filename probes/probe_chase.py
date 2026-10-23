"""
probe_chase.py -- the two cheap Paper C experiments, run on the RELEASED CHASE
checkpoints. No training, inference only.

  Exp 1  Diagnostic of the hull weights alpha (RBR decision rule, issues a/b, G3)
         N_eff = 1/sum(alpha^2), max alpha, alpha mass on zero-padded points,
         distance of p* from the S2CoM centroid, between-sample spread of p*.
  Exp 2  Exactness tests (Props. 10 and 12)
         (a) test-time global translation of the valid joints: equivariance error
             ||p*(x+t) - p*(x) - t|| / ||t||, accuracy and prediction flips
         (b) test-time permutation of the entity axis: ||p*(x_pi) - p*(x)||,
             accuracy and prediction flips
         CHASE+ (chase_plus.py) gives exactly zero for (a) and, if hierarchical,
         for (b); this probe measures how far the released CHASE is from that.

Run from inside the CHASE repo folder NeurIPS24/ :

  python probe_chase.py --config config/CTRGCN/cad/cad_chase.yaml \
                        --weights cad_chas.pt --out probe_cad
  python probe_chase.py --config config/CTRGCN/ntu11/ntu11_xview_joint_chase.yaml \
                        --weights ntu11_xview_chas.pt --out probe_ntu11

Outputs in --out: summary.json, per_sample.csv, neff_hist.png, translation.csv,
permutation.csv.
"""
import argparse
import csv
import importlib
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def import_class(name):
    mod, cls = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def load(cfg_path, weights, device, num_workers=0):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = import_class(cfg["model"])(**cfg["model_args"])
    if cfg.get("wrapper"):
        model = import_class(cfg["wrapper"])(model, **cfg["wrapper_args"])
    else:
        raise SystemExit("config has no CHASE wrapper")
    if list(cfg["wrapper_args"].get("pooling_seg", [1, 1, 1])) != [1, 1, 1]:
        raise SystemExit("probe assumes pooling_seg [1,1,1] (the CHASE default)")
    sd = torch.load(weights, map_location="cpu")
    sd = OrderedDict((k.split("module.")[-1], v) for k, v in sd.items())
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    feeder = import_class(cfg["feeder"])(**cfg["test_feeder_args"])
    loader = DataLoader(feeder, batch_size=cfg.get("test_batch_size", 64),
                        shuffle=False, num_workers=num_workers, drop_last=False)
    return cfg, model, loader


@torch.no_grad()
def hull(wrapper, x):
    """Replicates CHASWrapper.forward lines 19-21: alpha (N,U) and p* (N,C)."""
    N, C, T, V, M = x.shape
    alpha = wrapper.shift(x).view(N, T * M * V, -1)[..., 0].softmax(dim=1)
    tx = x.reshape(N, C, T * V * M)                              # (t v m), as rearrange
    pstar = torch.einsum("ncu,nu->nc", tx, alpha)
    return alpha, pstar


def valid_mask(x):
    return x.abs().amax(dim=1) > 0                                  # (N,T,V,M)


def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--sigmas", default="0.1,0.5,1,2,5",
                    help="translation std, in units of the data scale s")
    ap.add_argument("--perm_trials", type=int, default=3)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, loader = load(a.config, a.weights, dev, a.num_workers)
    sigmas = [float(s) for s in a.sigmas.split(",")]

    rows = []
    n_tot = 0
    correct = {"clean": 0}
    tr = {s: dict(correct=0, flips=0, eqerr=[], dvalid=[], dpad=[]) for s in sigmas}
    pe = dict(correct=0, flips=0, n=0, dp=[])

    for b, batch in enumerate(loader):
        if a.max_batches and b >= a.max_batches:
            break
        x = batch[0].float().to(dev)
        y = batch[1].long().to(dev)
        N, C, T, V, M = x.shape
        pm = valid_mask(x)                                          # (N,T,V,M)
        pmf = pm.flatten(1)                                         # (N,U) in (t v m)
        alpha, ps = hull(model, x)
        pred = logits_of(model, x).argmax(1)
        correct["clean"] += (pred == y).sum().item()

        # ---- Exp 1: weight diagnostics
        cnt = pm.flatten(1).sum(1).clamp_min(1)
        cen = (x * pm.unsqueeze(1)).flatten(2).sum(2) / cnt.unsqueeze(1)         # S2CoM (valid pts)
        dev2 = ((x - cen.view(N, C, 1, 1, 1)) ** 2).sum(1)                       # (N,T,V,M)
        scale = torch.sqrt((dev2 * pm).flatten(1).sum(1) / cnt)                  # RMS radius s
        neff = 1.0 / (alpha ** 2).sum(1)
        pad_mass = (alpha * (~pmf)).sum(1)
        for i in range(N):
            rows.append(dict(
                label=int(y[i]), correct=int(pred[i] == y[i]),
                U=int(alpha.shape[1]), U_valid=int(pmf[i].sum()),
                neff=float(neff[i]), max_alpha=float(alpha[i].max()),
                pad_mass=float(pad_mass[i]),
                dist_to_s2com_over_s=float((ps[i] - cen[i]).norm() / scale[i].clamp_min(1e-8)),
                scale=float(scale[i]),
                **{f"pstar_{c}": float(ps[i, c]) for c in range(C)}))

        # ---- Exp 2a: global translation of the valid joints only
        for s in sigmas:
            t = torch.randn(N, C, device=dev) * scale.view(N, 1) * s
            xt = x + t.view(N, C, 1, 1, 1) * pm.unsqueeze(1)
            _, pst = hull(model, xt)
            tn = t.norm(dim=1).clamp_min(1e-8)
            tr[s]["eqerr"] += ((pst - ps - t).norm(dim=1) / tn).tolist()
            # change of the backbone input after the CHASE shift
            xs0 = x - ps.view(N, C, 1, 1, 1); xs1 = xt - pst.view(N, C, 1, 1, 1)
            d = (xs1 - xs0).norm(dim=1) / scale.view(N, 1, 1, 1).clamp_min(1e-8)
            tr[s]["dvalid"] += [float(d[i][pm[i]].mean()) for i in range(N)]
            tr[s]["dpad"] += [float(d[i][~pm[i]].mean()) if (~pm[i]).any() else 0.0 for i in range(N)]
            pt = logits_of(model, xt).argmax(1)
            tr[s]["correct"] += (pt == y).sum().item()
            tr[s]["flips"] += (pt != pred).sum().item()

        # ---- Exp 2b: entity permutation
        if M > 1:
            for _ in range(a.perm_trials):
                perm = torch.arange(M - 1, -1, -1) if M == 2 else torch.randperm(M)
                if M > 2 and torch.equal(perm, torch.arange(M)):
                    perm = torch.roll(perm, 1)
                xp = x[..., perm.to(dev)]
                _, pp = hull(model, xp)
                pe["dp"] += ((pp - ps).norm(dim=1) / scale.clamp_min(1e-8)).tolist()
                pp_pred = logits_of(model, xp).argmax(1)
                pe["correct"] += (pp_pred == y).sum().item()
                pe["flips"] += (pp_pred != pred).sum().item()
                pe["n"] += N
        n_tot += N
        print(f"batch {b}: {n_tot} samples", flush=True)

    # ---------------------------------------------------------------- report
    neff = np.array([r["neff"] for r in rows]); uv = np.array([r["U_valid"] for r in rows])
    U = rows[0]["U"]
    P = np.array([[r[f"pstar_{c}"] for c in range(C)] for r in rows])
    S = np.array([r["scale"] for r in rows])
    q = lambda v: {k: float(np.quantile(v, p)) for k, p in
                   [("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95)]}
    summary = dict(
        config=a.config, weights=a.weights, n_samples=n_tot, U=U,
        acc_clean=100 * correct["clean"] / n_tot,
        exp1=dict(
            neff=q(neff), neff_over_Uvalid=q(neff / np.maximum(uv, 1)),
            frac_saturated_neff_lt_10=float((neff < 10).mean()),
            frac_overdispersed_neff_gt_half_Uvalid=float((neff > 0.5 * uv).mean()),
            max_alpha=q(np.array([r["max_alpha"] for r in rows])),
            pad_mass=q(np.array([r["pad_mass"] for r in rows])),
            frac_samples_with_padding=float((uv < U).mean()),
            pad_mass_mean_on_padded_samples=float(np.mean([r["pad_mass"] for r in rows if r["U_valid"] < U]))
                if (uv < U).any() else 0.0,
            dist_to_s2com_over_s=q(np.array([r["dist_to_s2com_over_s"] for r in rows])),
            pstar_between_sample_std_over_mean_s=(P.std(0) / S.mean()).tolist()),
        exp2_translation={str(s): dict(
            acc=100 * v["correct"] / n_tot, flip_rate=100 * v["flips"] / n_tot,
            equiv_err_median=float(np.median(v["eqerr"])), equiv_err_p95=float(np.quantile(v["eqerr"], .95)),
            input_change_valid_mean=float(np.mean(v["dvalid"])),
            input_change_padding_mean=float(np.mean(v["dpad"]))) for s, v in tr.items()},
        exp2_permutation=dict(
            acc=100 * pe["correct"] / max(pe["n"], 1), flip_rate=100 * pe["flips"] / max(pe["n"], 1),
            dpstar_over_s=q(np.array(pe["dp"])) if pe["dp"] else None),
        notes=["CHASE+ predicts equiv_err = 0, input_change = 0 and flip_rate = 0 for translation; "
               "hierarchical CHASE+ predicts dpstar = 0 and flip_rate = 0 for permutation.",
               "input_change_padding > 0 even for an exactly equivariant p*: zero padding becomes -p* "
               "in the released wrapper (G3); read equiv_err for the CLB itself."])

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(a.out, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "translation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma_over_s", "acc", "flip_rate", "equiv_err_median",
                                       "equiv_err_p95", "input_change_valid", "input_change_padding"])
        for s, v in summary["exp2_translation"].items():
            w.writerow([s, v["acc"], v["flip_rate"], v["equiv_err_median"], v["equiv_err_p95"],
                        v["input_change_valid_mean"], v["input_change_padding_mean"]])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].hist(neff, bins=50, color="#4C72B0")
        ax[0].set_xlabel("N_eff = 1 / sum alpha^2"); ax[0].set_ylabel("test samples")
        ax[0].set_title(f"U = {U}, median U_valid = {int(np.median(uv))}")
        ax[1].hist(neff / np.maximum(uv, 1), bins=50, color="#55A868")
        ax[1].set_xlabel("N_eff / U_valid  (1 = uniform = S2CoM)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "neff_hist.png"), dpi=150)
    except Exception as e:                                         # plotting is optional
        print("plot skipped:", e)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
