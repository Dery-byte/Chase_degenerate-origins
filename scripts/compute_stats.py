"""Reproduce Tables 4 and 5 (CAD accuracy per arm, paired differences) from the run CSVs.

    python scripts/compute_stats.py            # uses results/merged/cad_all_arms_envA.csv

Paired statistics are over seeds common to both arms: mean difference, wins,
95% t-interval, paired t-test p-value and exact Wilcoxon signed-rank p-value.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["vanilla", "chase", "fixed", "chaseplus", "chaseplusB"]
NAMES = {"vanilla": "Vanilla", "chase": "CHASE", "fixed": "Fixed", "chaseplus": "CHASE+", "chaseplusB": "CHASE+B"}
PAIRS = [("fixed", "chase"), ("chaseplus", "chase"), ("chaseplusB", "chase"), ("chaseplus", "fixed"),
         ("chaseplusB", "fixed"), ("chaseplusB", "chaseplus"), ("chaseplus", "vanilla"),
         ("chaseplusB", "vanilla"), ("fixed", "vanilla")]


def paired(p, a, b):
    d = (p[a] - p[b]).dropna()
    n = len(d)
    if n < 2:
        return None
    m, s = d.mean(), d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    t = stats.ttest_rel(p.loc[d.index, a], p.loc[d.index, b]).pvalue
    try:
        w = stats.wilcoxon(d, mode="exact").pvalue
    except TypeError:                       # older scipy
        w = stats.wilcoxon(d).pvalue
    return dict(comparison=f"{NAMES[a]} - {NAMES[b]}", n=n, wins=f"{int((d > 0).sum())}/{n}",
                mean=round(m, 2), ci95=f"[{m - h:.2f}, {m + h:.2f}]", p_t=round(t, 3), p_wilcoxon=round(w, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results/merged/cad_all_arms_envA.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    for metric in ("best_acc", "last_acc"):
        p = df.pivot(index="seed", columns="arm", values=metric)[[c for c in ORDER if c in set(df.arm)]]
        print(f"\n=== {metric}: per seed ===")
        print(p.to_string())
        print(f"\n=== {metric}: mean +- s.d. ===")
        for c in p.columns:
            v = p[c].dropna()
            print(f"{NAMES[c]:8s}  {v.mean():6.2f} +- {v.std(ddof=1):5.2f}   (n={len(v)})")
        print(f"\n=== {metric}: paired differences ===")
        rows = [r for r in (paired(p, x, y) for x, y in PAIRS) if r]
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
