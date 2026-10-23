"""Calibrate the Sinkhorn epsilon ONCE on CAD training data, then freeze it in the config.

Exactly the procedure used for CHASE+ / CHASE+B (notebooks 06 and 07): 256 random training
clips (numpy seed 0), valid points centred per clip, 2048 points sub-sampled, and
eps = 0.05 * median(0.5 * ||x - y||^2).  Because the training feeder applies random
augmentation, the value varies slightly between runs; the values actually used are
eps = 681.95 (CHASE+, configs/cad_chaseplus.yaml) and eps = 665.36 (CHASE+B, configs/cad_chaseplusB.yaml).

Run from the CHASE repository folder NeurIPS24/:
    python calibrate_eps.py --config cad_chaseplus.yaml
"""
import argparse

import numpy as np
import yaml
from feeders.feeder_cad import Collective

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
ds = Collective(**cfg["train_feeder_args"])
rng = np.random.default_rng(0)
pts = []
for i in rng.choice(len(ds), 256, replace=False):
    x = ds[int(i)][0].numpy()                      # (C, T, V, M)
    v = np.abs(x).max(0) > 0
    p = x.transpose(1, 2, 3, 0)[v]                 # valid points (P, C)
    if len(p):
        pts.append(p - p.mean(0))
P = np.concatenate(pts)
P = P[rng.choice(len(P), min(2048, len(P)), replace=False)]
eps = float(0.05 * np.median(0.5 * ((P[:, None] - P[None]) ** 2).sum(-1)))
print("frozen eps =", eps)
