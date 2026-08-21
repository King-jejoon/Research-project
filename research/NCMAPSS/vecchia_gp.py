"""
vecchia_gp.py  —  Scalable Gaussian-process healthy baseline via the
Vecchia approximation (Guinness, Technometrics 2018).

Why:  the existing baseline (gp_residuals.MultitaskGPModel, ExactGP) is O(n^3)
and is therefore fit on only ~180 sub-sampled normal points.  Vecchia's
approximation replaces each conditional density by conditioning on a small set
of *nearest previously-ordered* neighbours, giving an O(n m^3) likelihood that
can be trained on thousands of normal points -> a sharper, less biased healthy
baseline  y_hat = f(W)  and cleaner residuals  r = y - y_hat.

Faithful to the paper's two contributions:
  * permutation / ordering  -> maximum-minimum-distance (max-min) ordering
  * conditioning sets        -> m nearest previously-ordered neighbours
(Grouping is approximated by the shared-neighbour batched solve.)

This is an independent-per-sensor (single-output) Vecchia GP: 14 sensors share
the same input locations W and ordering, but each has its own RBF-ARD
hyper-parameters.  Cross-task ICM coupling is dropped for scalability (residuals
are computed per sensor anyway); a multitask-Vecchia extension is possible.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

JIT = 1e-4


# ----------------------------------------------------------------------
#  ordering + neighbours  (Guinness 2018, Sec. 3)
# ----------------------------------------------------------------------
def maxmin_order(X):
    """Maximum-minimum-distance ordering (vectorised, O(n^2)).
    Start at the point closest to the centroid; each subsequent point
    maximises its minimum distance to all already-ordered points."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    order = np.empty(n, dtype=np.int64)
    c = X.mean(0)
    first = int(((X - c) ** 2).sum(1).argmin())
    order[0] = first
    mind = ((X - X[first]) ** 2).sum(1)
    mind[first] = -1.0
    for k in range(1, n):
        j = int(mind.argmax())
        order[k] = j
        dj = ((X - X[j]) ** 2).sum(1)
        mind = np.minimum(mind, dj)
        mind[j] = -1.0
    return order


def build_neighbours(Xo, m):
    """For each ordered point, the m nearest neighbours among the *previous*
    points (NNGP-style).  Returns NB (n,m) int and MASK (n,m) bool."""
    n = Xo.shape[0]
    tree = cKDTree(Xo)
    kq = min(n, m * 4 + 1)
    _, knn = tree.query(Xo, k=kq)
    knn = np.atleast_2d(knn)
    NB = np.zeros((n, m), dtype=np.int64)
    MASK = np.zeros((n, m), dtype=bool)
    for i in range(n):
        prev = knn[i][knn[i] < i]      # keep only earlier-ordered points
        prev = prev[:m]
        NB[i, :len(prev)] = prev
        MASK[i, :len(prev)] = True
    return NB, MASK


# ----------------------------------------------------------------------
#  Vecchia GP (independent per sensor, RBF-ARD)
# ----------------------------------------------------------------------
class VecchiaGP:
    def __init__(self, m=15, device='cpu'):
        self.m = m
        self.device = device

    # ---- standardisation ----
    def _fit_scalers(self, W, Y):
        self.x_mean = W.mean(0); self.x_std = W.std(0) + 1e-8
        self.y_mean = Y.mean(0); self.y_std = Y.std(0) + 1e-8

    def _sx(self, W):  return (W - self.x_mean) / self.x_std
    def _sy(self, Y):  return (Y - self.y_mean) / self.y_std

    # ---- kernel pieces (batched over sensors S) ----
    @staticmethod
    def _knn_kqn(Dnb, Dself, ell2, sf2):
        """Dnb (N,m,m,d), Dself (N,m,d); ell2 (S,d), sf2 (S,)
        -> Knn (S,N,m,m), Kqn (S,N,m)  (no noise/mask yet)."""
        e_nb = -0.5 * (Dnb[None] / ell2[:, None, None, None, :]).sum(-1)   # (S,N,m,m)
        Knn = sf2[:, None, None, None] * torch.exp(e_nb)
        e_q = -0.5 * (Dself[None] / ell2[:, None, None, :]).sum(-1)        # (S,N,m)
        Kqn = sf2[:, None, None] * torch.exp(e_q)
        return Knn, Kqn

    def fit(self, W, Y, iters=150, lr=0.05, seed=0, verbose=True):
        torch.manual_seed(seed); np.random.seed(seed)
        W = np.asarray(W, np.float64); Y = np.asarray(Y, np.float64)
        self._fit_scalers(W, Y)
        Xs = self._sx(W); Ys = self._sy(Y)
        S = Ys.shape[1]

        # ordering + neighbours (shared across sensors)
        order = maxmin_order(Xs)
        Xo = Xs[order]; Yo = Ys[order]
        NB, MASK = build_neighbours(Xo, self.m)
        self._order = order; self._Xo = Xo; self._Yo = Yo

        dev = self.device
        Xo_t = torch.tensor(Xo, dtype=torch.float32, device=dev)
        Yo_t = torch.tensor(Yo, dtype=torch.float32, device=dev)
        NB_t = torch.tensor(NB, dtype=torch.long, device=dev)
        validf = torch.tensor(MASK, dtype=torch.float32, device=dev)        # (N,m)

        # precompute per-dim squared diffs (locations are fixed)
        Xnb = Xo_t[NB_t]                                                    # (N,m,d)
        Dself = (Xo_t[:, None, :] - Xnb) ** 2                              # (N,m,d)
        Dnb = (Xnb[:, :, None, :] - Xnb[:, None, :, :]) ** 2               # (N,m,m,d)
        Ynb = Yo_t[NB_t]                                                    # (N,m,S)
        Ynb = Ynb.permute(2, 0, 1)                                          # (S,N,m)
        yo = Yo_t.t().contiguous()                                          # (S,N)
        N, m = NB.shape
        eye = torch.eye(m, device=dev)
        pair = (validf[:, :, None] * validf[:, None, :])[None]            # (1,N,m,m)
        d = Xs.shape[1]

        # parameters per sensor
        log_ell = torch.zeros(S, d, device=dev, requires_grad=True)
        log_sf = torch.zeros(S, device=dev, requires_grad=True)
        log_noise = torch.full((S,), -2.0, device=dev, requires_grad=True)
        opt = torch.optim.Adam([log_ell, log_sf, log_noise], lr=lr)

        Ynb_m = Ynb * validf[None]                                          # mask invalid y

        for it in range(iters):
            opt.zero_grad()
            ell2 = torch.exp(2 * log_ell)
            sf2 = torch.exp(2 * log_sf)
            noise = torch.exp(log_noise) + 1e-6
            Knn, Kqn = self._knn_kqn(Dnb, Dself, ell2, sf2)
            Kqn = Kqn * validf[None]
            Knn = Knn * pair
            diagval = validf * (noise[:, None, None] + JIT) + (1 - validf) * 1.0   # (S,N,m)
            Knn = Knn + diagval[..., None] * eye
            kii = sf2 + noise                                              # (S,)
            rhs = torch.stack([Kqn, Ynb_m], dim=-1)                        # (S,N,m,2)
            sol = torch.linalg.solve(Knn, rhs)                            # (S,N,m,2)
            a = sol[..., 0]; b = sol[..., 1]
            cond_mean = (Kqn * b).sum(-1)                                  # (S,N)
            cond_var = (kii[:, None] - (Kqn * a).sum(-1)).clamp_min(1e-6)
            nll = 0.5 * (torch.log(2 * np.pi * cond_var) + (yo - cond_mean) ** 2 / cond_var)
            loss = nll.mean()
            loss.backward(); opt.step()
            if verbose and (it + 1) % 25 == 0:
                print(f'    vecchia iter {it+1}/{iters}  nll={loss.item():.4f}')

        self.log_ell = log_ell.detach()
        self.log_sf = log_sf.detach()
        self.log_noise = log_noise.detach()
        self._tree = cKDTree(Xo)
        return self

    @torch.no_grad()
    def predict(self, Wq):
        """Return (mean, std) in original sensor units, shape (len(Wq), S)."""
        dev = self.device
        Xq = self._sx(np.asarray(Wq, np.float64))
        m = self.m
        kq = min(self._Xo.shape[0], m)
        _, NBq = self._tree.query(Xq, k=kq)
        NBq = np.atleast_2d(NBq)[:, :m]

        Xo_t = torch.tensor(self._Xo, dtype=torch.float32, device=dev)
        Yo_t = torch.tensor(self._Yo, dtype=torch.float32, device=dev)
        Xq_t = torch.tensor(Xq, dtype=torch.float32, device=dev)
        NBq_t = torch.tensor(NBq, dtype=torch.long, device=dev)

        Xnb = Xo_t[NBq_t]                                                  # (Nq,m,d)
        Dself = (Xq_t[:, None, :] - Xnb) ** 2                             # (Nq,m,d)
        Dnb = (Xnb[:, :, None, :] - Xnb[:, None, :, :]) ** 2              # (Nq,m,m,d)
        Ynb = Yo_t[NBq_t].permute(2, 0, 1)                                # (S,Nq,m)

        ell2 = torch.exp(2 * self.log_ell); sf2 = torch.exp(2 * self.log_sf)
        noise = torch.exp(self.log_noise) + 1e-6
        Knn, Kqn = self._knn_kqn(Dnb, Dself, ell2, sf2)
        mm = Knn.shape[-1]
        Knn = Knn + (noise[:, None, None] + JIT)[..., None] * torch.eye(mm, device=dev)
        kii = sf2 + noise
        rhs = torch.stack([Kqn, Ynb], dim=-1)
        sol = torch.linalg.solve(Knn, rhs)
        a = sol[..., 0]; b = sol[..., 1]
        mean_s = (Kqn * b).sum(-1)                                         # (S,Nq)
        var_s = (kii[:, None] - (Kqn * a).sum(-1)).clamp_min(1e-6)
        mean = mean_s.t().cpu().numpy() * self.y_std + self.y_mean
        std = np.sqrt(var_s.t().cpu().numpy()) * self.y_std
        return mean, std
