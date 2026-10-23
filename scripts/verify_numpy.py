"""
NumPy verification of every formal claim in the Paper C (CHASE) methodology.
Runs on CPU in seconds; no deep-learning framework needed.

    python verify_numpy.py

Each check prints PASS/FAIL with the numerical residual.
"""
import numpy as np
from numpy.linalg import norm

rng = np.random.default_rng(0)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def lse(z, axis):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rand_rot():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ---------------------------------------------------------------- C1: PBR
def plucker(P, parents, normalise=True, eps=1e-6):
    """P: (J,3) joints. Returns d (J,3), m (J,3) for bone j -> parent(j)."""
    d = P - P[parents]                      # d_ij = p_i - p_j with j = parent
    m = np.cross(P[parents], d)             # m_ij = p_j x d_ij
    if normalise:
        m = m / np.maximum(norm(d, axis=1, keepdims=True), eps)
    return d, m


J = 25
parents = np.r_[0, rng.integers(0, np.arange(1, J))]   # random tree, root 0 is self-loop
P = rng.normal(size=(J, 3))
R, t = rand_rot(), rng.normal(size=3) * 5

d, m = plucker(P, parents, normalise=False)
d2, m2 = plucker(P @ R.T + t, parents, normalise=False)
check("Prop 7a  d' = R d", np.allclose(d2, d @ R.T), f"res={abs(d2 - d @ R.T).max():.2e}")
pred = m @ R.T + np.cross(t, d @ R.T)
check("Prop 7b  m' = R m + t x R d", np.allclose(m2, pred), f"res={abs(m2 - pred).max():.2e}")
check("Prop 7c  d . m = 0 preserved", abs((d2 * m2).sum(1)).max() < 1e-9)
# Jacobian of m' wrt t is -[Rd]_x, rank 2 for d != 0
u = d[3] @ R.T
Jt = np.stack([np.cross(np.eye(3)[k], u) for k in range(3)], 1)
check("Prop 7d  rank dm'/dt = 2", np.linalg.matrix_rank(Jt) == 2)
# vector bones are blind to any origin shift; Plucker moments are not
ps = rng.normal(size=3)
dv, mv = plucker(P - ps, parents, normalise=False)
check("Prop 7e  vector bone shift-blind, moment shift-sensitive",
      np.allclose(dv, d) and not np.allclose(mv, m))

# Prop 8: p* in conv hull => ||m_hat|| <= diam (normalised) and ||m|| <= ||d|| diam
X = rng.normal(size=(200, 3)) * 3 + 10                   # all points of a sequence
alpha = rng.dirichlet(np.ones(200))
pstar = alpha @ X
Pj = X[:J]
diam = max(norm(X[i] - X[k]) for i in range(200) for k in range(200))
dh, mh = plucker(Pj - pstar, parents, normalise=True)
dr, mr = plucker(Pj - pstar, parents, normalise=False)
ok8 = (norm(mh, axis=1) <= diam + 1e-9).all() and \
      (norm(mr, axis=1) <= norm(dr, axis=1) * diam + 1e-9).all()
check("Prop 8   moment bound inside convex hull", ok8,
      f"max||m_hat||={norm(mh,axis=1).max():.2f} <= diam={diam:.2f}")
far = pstar + 1e3
_, mfar = plucker(Pj - far, parents, normalise=True)
check("Prop 8   arbitrary origin breaks bound", norm(mfar, axis=1).max() > diam)

# 2-D case (CAD / VD): scalar moment, same shift sensitivity
P2 = rng.normal(size=(17, 2)); par2 = np.r_[0, rng.integers(0, np.arange(1, 17))]
d2d = P2 - P2[par2]; m2d = P2[par2, 0] * d2d[:, 1] - P2[par2, 1] * d2d[:, 0]
t2 = rng.normal(size=2)
Q2 = P2 + t2; dq = Q2 - Q2[par2]; mq = Q2[par2, 0] * dq[:, 1] - Q2[par2, 1] * dq[:, 0]
check("Prop 7(2D) m' = m + t x d (scalar)",
      np.allclose(mq, m2d + t2[0] * d2d[:, 1] - t2[1] * d2d[:, 0]))

# ------------------------------------------- C1/C2: exact translation equivariance
W1 = rng.normal(size=(16, 3)); b1 = rng.normal(size=16); W3 = rng.normal(size=(200, 16))


def psi(Xin):                       # CHASE CLB, Eq. 8 (squeeze = mean over points)
    h = np.tanh((Xin @ W1.T + b1).mean(0))
    return W3 @ h


def pstar_chase(Xin, centred):
    z = psi(Xin - Xin.mean(0)) if centred else psi(Xin)
    return softmax(z) @ Xin


tv = np.array([3.0, -2.0, 7.0])
e_orig = norm(pstar_chase(X + tv, False) - (pstar_chase(X, False) + tv))
e_cent = norm(pstar_chase(X + tv, True) - (pstar_chase(X, True) + tv))
check("Prop 10  original CLB is NOT translation-equivariant", e_orig > 1e-3, f"err={e_orig:.3f}")
check("Prop 10  centred CLB: p*(X+t) = p*(X) + t", e_cent < 1e-10, f"err={e_cent:.1e}")

# ------------------------------------------- C2: RBR (Renyi-2 band)
for _ in range(1000):
    a = softmax(rng.normal(size=200) * rng.uniform(0.1, 8))
    H2 = -np.log((a ** 2).sum()); Neff = np.exp(H2); kappa = rng.uniform(1, 150)
    if Neff >= kappa and a.max() > 1 / np.sqrt(kappa) + 1e-12:
        check("Prop 9   N_eff>=kappa => max a <= 1/sqrt(kappa)", False); break
else:
    check("Prop 9   N_eff>=kappa => max a <= 1/sqrt(kappa)  (1000 draws)", True)
a = np.r_[0.5, np.full(10**4, 0.5 / 10**4)]
Hs = -(a * np.log(a)).sum(); H2 = -np.log((a ** 2).sum())
check("Prop 9   Shannon gives no max-weight bound",
      np.exp(Hs) > 100 and a.max() > 1 / np.sqrt(100) and np.exp(H2) < 4.01,
      f"exp(H)={np.exp(Hs):.0f}, exp(H2)={np.exp(H2):.2f}, max={a.max()}")
# upper band: N_eff <= kappa_hi keeps alpha away from uniform (S2CoM)
U = 200; khi = 50
a = softmax(rng.normal(size=U) * 2)
gap = ((a - 1 / U) ** 2).sum(); bound = 1 / np.exp(-np.log((a ** 2).sum())) - 1 / U
check("Prop 11  ||a - 1/U||^2 = 1/N_eff - 1/U", np.isclose(gap, bound))
# sensitivity of p* to one joint (alpha fixed) is alpha_k
k = 7; dlt = rng.normal(size=3)
Xp = X.copy(); Xp[k] += dlt
a = softmax(rng.normal(size=200))
check("Prop 11  ||dp*|| = alpha_k ||delta|| <= ||delta||/sqrt(N_eff)",
      np.isclose(norm(a @ Xp - a @ X), a[k] * norm(dlt)) and a[k] <= 1 / np.sqrt(np.exp(-np.log((a**2).sum()))) + 1e-12)

# entity-permutation invariance of the hierarchical form
E, n = 3, 40
Xe = rng.normal(size=(E, n, 3)) + rng.normal(size=(E, 1, 3)) * 4
Wg = rng.normal(size=(n, 3)); wh = rng.normal(size=3)


def pstar_hier(Xe):
    p = 0
    ent_logits = np.array([wh @ Xe[e].mean(0) for e in range(len(Xe))])
    ae = softmax(ent_logits)
    for e in range(len(Xe)):
        c = Xe[e] - Xe[e].mean(0)
        au = softmax((Wg * c).sum(1))       # shared per-entity head
        p = p + ae[e] * (au @ Xe[e])
    return p


perm = [2, 0, 1]
check("Prop 12  hierarchical p* is entity-permutation invariant",
      np.allclose(pstar_hier(Xe), pstar_hier(Xe[perm])))


# ------------------------------------------- C3: SD-MPD
def ot_eps(a, x, b, y, eps, n_it=2000):
    C = 0.5 * ((x[:, None] - y[None]) ** 2).sum(-1)
    f = np.zeros(len(a)); g = np.zeros(len(b))
    la, lb = np.log(a), np.log(b)
    for _ in range(n_it):
        f = -eps * lse(lb[None] + (g[None] - C) / eps, 1)
        g = -eps * lse(la[:, None] + (f[:, None] - C) / eps, 0)
    return a @ f + b @ g


def sink_div(a, x, b, y, eps):
    return ot_eps(a, x, b, y, eps) - 0.5 * ot_eps(a, x, a, x, eps) - 0.5 * ot_eps(b, y, b, y, eps)


nP, nQ = 60, 50
x = rng.normal(size=(nP, 3)); y = rng.normal(size=(nQ, 3)) * 1.3 + np.array([2.0, -1, 0.5])
a = rng.dirichlet(np.ones(nP)); b = rng.dirichlet(np.ones(nQ))
eps = 0.1
S = sink_div(a, x, b, y, eps)
check("Prop 14  S_eps >= 0", S >= -1e-8, f"S={S:.4f}")
check("Prop 14  S_eps(P,P) = 0", abs(sink_div(a, x, a, x, eps)) < 1e-8)
mu_x, mu_y = a @ x, b @ y
Sc = sink_div(a, x - mu_x, b, y - mu_y, eps)
dec = Sc + 0.5 * norm(mu_x - mu_y) ** 2
check("Prop 15  S(P,Q) = S(P~,Q~) + 1/2||mu_P - mu_Q||^2", abs(S - dec) < 1e-6,
      f"S={S:.6f} vs {dec:.6f}")

# Prop 16: per-sample discrepancy is blind to the common shift (vacuous gradient)
xs = rng.normal(size=(40, 3)); ys = rng.normal(size=(40, 3)) + 1
u40 = np.full(40, 1 / 40); ps = rng.normal(size=3) * 3
check("Prop 16  per-sample S(Pa-p*, Pb-p*) independent of p*",
      abs(sink_div(u40, xs, u40, ys, eps) - sink_div(u40, xs - ps, u40, ys - ps, eps)) < 1e-8)

# Prop 15 corollary on batch-pooled marginals: mean gap invariant to sample-adaptive p*
Nb, n = 16, 10
Xa = rng.normal(size=(Nb, n, 3)) + rng.normal(size=(Nb, 1, 3)) * 3
Xb = Xa + rng.normal(size=(Nb, n, 3)) * 0.3 + np.array([1.0, 0, 0])
pst = rng.normal(size=(Nb, 1, 3)) * 5
gap0 = Xa.reshape(-1, 3).mean(0) - Xb.reshape(-1, 3).mean(0)
gap1 = (Xa - pst).reshape(-1, 3).mean(0) - (Xb - pst).reshape(-1, 3).mean(0)
check("Prop 15  pooled mean gap invariant to sample-adaptive p*", np.allclose(gap0, gap1))
u = np.full(Nb * n, 1 / (Nb * n))
S_before = sink_div(u, Xa.reshape(-1, 3), u, Xb.reshape(-1, 3), 0.5)
pst_good = Xa.mean(1, keepdims=True)                   # re-centre each sample
S_after = sink_div(u, (Xa - pst_good).reshape(-1, 3), u, (Xb - pst_good).reshape(-1, 3), 0.5)
floor = 0.5 * norm(gap0) ** 2
check("Prop 15  pooled S reducible by p*, but never below the floor",
      S_after < S_before and S_after >= floor - 1e-6,
      f"before={S_before:.3f} after={S_after:.3f} floor={floor:.3f}")

# Masking: zero weight on padded points == dropping them
xpad = np.vstack([x, np.zeros((10, 3))]); apad = np.r_[a, np.full(10, 1e-30)]; apad /= apad.sum()
check("Mask     zero-weighted padding == removed padding",
      abs(sink_div(apad, xpad, b, y, eps) - S) < 1e-5)

# Repo MMD (CHASE utils/loss.py) bandwidth is data-adaptive -> scale-free
def repo_mmd(s, t, mul=2.0, num=5):
    tot = np.vstack([s, t]); n = len(tot)
    L2 = ((tot[:, None] - tot[None]) ** 2).sum(-1)
    bw = L2.sum() / (n * n - n) / mul ** (num // 2)
    K = sum(np.exp(-L2 / (bw * mul ** i)) for i in range(num))
    k = len(s)
    return K[:k, :k].mean() + K[k:, k:].mean() - 2 * K[:k, k:].mean()


s1 = rng.normal(size=(64, 3)); t1 = rng.normal(size=(64, 3)) + 2
check("Repo MMD is scale-invariant (adaptive bandwidth, not fixed)",
      np.isclose(repo_mmd(s1, t1), repo_mmd(100 * s1, 100 * t1)))
check("Repo MMD (V-statistic) is biased: > 0 for P = Q samples",
      np.mean([repo_mmd(rng.normal(size=(64, 3)), rng.normal(size=(64, 3))) for _ in range(50)]) > 0)

# Envelope gradient used in chase_plus.ot_eps: dOT/dx_i = sum_j Pi_ij (x_i - y_j)
def ot_grad_x(a, x, b, y, eps, n_it=3000, self_term=False):
    C = 0.5 * ((x[:, None] - y[None]) ** 2).sum(-1)
    f = np.zeros(len(a)); g = np.zeros(len(b)); la, lb = np.log(a), np.log(b)
    for _ in range(n_it):
        f = -eps * lse(lb[None] + (g[None] - C) / eps, 1)
        g = -eps * lse(la[:, None] + (f[:, None] - C) / eps, 0)
    Pi = np.exp((f[:, None] + g[None] - C) / eps + la[:, None] + lb[None])
    gx = (Pi.sum(1)[:, None] * x - Pi @ y)
    if self_term:                       # x on both sides: add the column contribution
        gx = gx + (Pi.sum(0)[:, None] * x - Pi.T @ x)
    return gx


G = ot_grad_x(a, x, b, y, eps) - 0.5 * ot_grad_x(a, x, a, x, eps, self_term=True)
h = 1e-5; E_ = np.zeros_like(x); E_[3, 1] = h
fd = (sink_div(a, x + E_, b, y, eps) - sink_div(a, x - E_, b, y, eps)) / (2 * h)
check("Grad     envelope gradient of S_eps == finite differences", abs(fd - G[3, 1]) < 1e-5,
      f"fd={fd:.6f} env={G[3,1]:.6f}")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
