"""
Anisotropic Harmonic Flow Matching with Action-Optimal Transport
================================================================

The key missing piece: the COUPLING π(x₀, x₁) determines which source
points are paired with which target points. Standard OT minimizes
½|x₁ - x₀|² (Euclidean). We instead minimize the anisotropic action:

    S(x₀, x₁) = Σₖ (ωₖ/2sinωₖ)[(x̃₀ᵏ² + x̃₁ᵏ²)cosωₖ - 2x̃₀ᵏx̃₁ᵏ]

where x̃ = Rx are coordinates in the eigenbasis of Ω².

This action penalizes transport across the gap (high ω₂) more than
transport along the crescents (low ω₁), so the OT coupling preferentially
pairs source points with nearby target points measured in the *anisotropic
metric*, not the Euclidean one.

Three coupling strategies compared:
  1. Random:      π = ρ₀ ⊗ ρ₁ (independent, no optimization)
  2. Euclidean OT: min E[½|x₁-x₀|²]  (standard)
  3. Action OT:    min E[S(x₀,x₁)]    (anisotropic, ours)
"""

import math
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Dict, Tuple

warnings.filterwarnings("ignore")


# ============================================================================
# 1. Data
# ============================================================================

def make_moons(n=1000, noise=0.07, seed=42):
    rng = np.random.RandomState(seed)
    n1 = n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n - n1)
    upper = np.column_stack([np.cos(t1), np.sin(t1)])
    lower = np.column_stack([1 - np.cos(t2), -np.sin(t2) + 0.5])
    X = np.vstack([upper, lower]) + noise * rng.randn(n, 2)
    X -= X.mean(0)
    return X


# ============================================================================
# 2. Anisotropic Harmonic Oscillator
# ============================================================================

@dataclass
class AnisoParams:
    omega1: float = 1.5
    omega2: float = 4.5
    angle: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def R(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    @property
    def Omega2(self):
        R = self.R
        return R.T @ np.diag([self.omega1**2, self.omega2**2]) @ R

    @classmethod
    def from_data(cls, data, omega_ratio=3.0, omega_base=1.5):
        center = data.mean(0)
        cov = np.cov(data.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        return cls(omega1=omega_base, omega2=omega_base * omega_ratio,
                   angle=angle, center=center)

    def to_tensors(self, device='cpu'):
        """Return (R, w, center) as torch tensors."""
        R = torch.tensor(self.R, dtype=torch.float32, device=device)
        w = torch.tensor([self.omega1, self.omega2], dtype=torch.float32, device=device)
        center = torch.tensor(self.center, dtype=torch.float32, device=device)
        return R, w, center


# ============================================================================
# 3. Analytic Paths, Velocities, and Action
# ============================================================================

def aniso_path(t, x0, x1, p: AnisoParams):
    """ψ_t: analytic sinusoidal interpolation in eigenbasis.

    Args:
        t:  Tensor[N] — time values in [0, 1]
        x0: Tensor[N, 2]
        x1: Tensor[N, 2]
        p:  AnisoParams

    Returns:
        Tensor[N, 2]
    """
    R, w, center = p.to_tensors(x0.device)
    x0t = (x0 - center) @ R.T          # [N, 2]
    x1t = (x1 - center) @ R.T          # [N, 2]
    st = torch.sin(w * (1 - t[:, None])) / torch.sin(w)   # [N, 2]
    at = torch.sin(w * t[:, None])      / torch.sin(w)    # [N, 2]
    return (st * x0t + at * x1t) @ R + center


def aniso_velocity(t, x0, x1, p: AnisoParams):
    """ψ̇_t: analytic velocity.

    Args:
        t:  Tensor[N]
        x0: Tensor[N, 2]
        x1: Tensor[N, 2]

    Returns:
        Tensor[N, 2]
    """
    R, w, center = p.to_tensors(x0.device)
    x0t = (x0 - center) @ R.T
    x1t = (x1 - center) @ R.T
    dst = -w * torch.cos(w * (1 - t[:, None])) / torch.sin(w)
    dat =  w * torch.cos(w * t[:, None])        / torch.sin(w)
    return (dst * x0t + dat * x1t) @ R


def aniso_action(x0, x1, p: AnisoParams):
    """Batched pairwise anisotropic action (Mehler kernel exponent).

        S[i, j] = Σₖ (ωₖ/2sinωₖ)[(x̃₀ᵢᵏ² + x̃₁ⱼᵏ²)cosωₖ - 2x̃₀ᵢᵏx̃₁ⱼᵏ]

    Args:
        x0: Tensor[N0, 2]
        x1: Tensor[N1, 2]

    Returns:
        Tensor[N0, N1]
    """
    R, w, center = p.to_tensors(x0.device)
    x0t = (x0 - center) @ R.T          # [N0, 2]
    x1t = (x1 - center) @ R.T          # [N1, 2]
    a = x0t[:, None, :]                 # [N0, 1, 2]
    b = x1t[None, :, :]                 # [1, N1, 2]
    coeff = w / (2 * torch.sin(w))      # [2]
    return (coeff * ((a**2 + b**2) * torch.cos(w) - 2 * a * b)).sum(-1)  # [N0, N1]


def euclidean_action(x0, x1):
    """Vectorized pairwise Euclidean action: S[i,j] = ½|x1_j - x0_i|².

    Args:
        x0: Tensor[N0, 2]
        x1: Tensor[N1, 2]

    Returns:
        Tensor[N0, N1]
    """
    return 0.5 * torch.cdist(x0, x1, p=2) ** 2


def free_path(t, x0, x1):
    """Straight-line interpolation. t: Tensor[N], x0/x1: Tensor[N,2]."""
    return (1 - t[:, None]) * x0 + t[:, None] * x1


def free_velocity(x0, x1):
    """Constant velocity for free particle. x0/x1: Tensor[N,2]."""
    return x1 - x0


# ============================================================================
# 4. Optimal Transport Solvers
# ============================================================================

def ot_coupling(x0_batch, x1_batch, cost_fn):
    """Exact OT coupling via Hungarian algorithm.

    Args:
        x0_batch: Tensor[N, 2]
        x1_batch: Tensor[N, 2]
        cost_fn:  callable(Tensor[N0,2], Tensor[N1,2]) -> Tensor[N0,N1]

    Returns:
        col_ind: np.ndarray[N] — permutation such that x1_batch[col_ind[i]]
                 is paired with x0_batch[i]
    """
    with torch.no_grad():
        C = cost_fn(x0_batch, x1_batch)
    _, col_ind = linear_sum_assignment(C.cpu().numpy())
    return col_ind


def sinkhorn_coupling(x0_batch, x1_batch, cost_fn, reg=0.1, n_iter=50):
    """Entropic OT via Sinkhorn iterations; returns hard assignment via sampling.

    Returns:
        perm: np.ndarray[N]
    """
    with torch.no_grad():
        C = cost_fn(x0_batch, x1_batch)
    K = torch.exp(-C / reg)
    n = x0_batch.shape[0]
    u = torch.ones(n, device=x0_batch.device) / n
    for _ in range(n_iter):
        v = 1.0 / (K.T @ u + 1e-10)
        u = 1.0 / (K @ v + 1e-10)
    P = (u[:, None] * K) * v[None, :]  # [N, N]

    # Sample hard assignment from soft coupling
    P_np = (P / P.sum(1, keepdim=True)).cpu().numpy()
    perm = np.array([np.random.choice(n, p=P_np[i]) for i in range(n)])
    return perm


# ============================================================================
# 5. Velocity Network
# ============================================================================

class VelocityNet(nn.Module):
    """Random Fourier Feature network for flow matching velocity field.

    Architecture:
        φ(x, t) = cos(W₁ [x; t] + b₁)   (fixed random features)
        v(x, t) = W₂ φ(x, t) + b₂        (trained linear layer)
    """

    def __init__(self, d=2, n_feat=1024, seed=42):
        super().__init__()
        self.d = d
        gen = torch.Generator().manual_seed(seed)
        W1 = torch.randn(n_feat, d + 1, generator=gen) * 3.0
        b1 = torch.rand(n_feat, generator=gen) * 2 * math.pi
        self.register_buffer('W1', W1)
        self.register_buffer('b1', b1)
        self.linear = nn.Linear(n_feat, d)
        nn.init.normal_(self.linear.weight, std=0.001)
        nn.init.zeros_(self.linear.bias)

    def _features(self, X, T):
        """X: [N,2], T: [N] → [N, n_feat]"""
        XT = torch.cat([X, T[:, None]], dim=1)       # [N, d+1]
        return torch.cos(XT @ self.W1.T + self.b1)   # [N, n_feat]

    def forward(self, X, T):
        """X: [N,2], T: [N] → [N, 2]"""
        return self.linear(self._features(X, T))


# ============================================================================
# 6. Flow Matching with Different Couplings
# ============================================================================

def train_fm(
    target_data, params: AnisoParams,
    coupling_mode='action_ot',   # 'random', 'euclidean_ot', 'action_ot'
    path_mode='aniso',           # 'aniso', 'free'
    n_epochs=400, batch_size=128, lr=5e-4, seed=42,
    device='cpu',
):
    """Train flow matching with specified coupling and path type.

    coupling_mode:
      - 'random':        no OT, just random pairing
      - 'euclidean_ot':  minibatch OT with cost = ½|x₁-x₀|²
      - 'action_ot':     minibatch OT with cost = S_aniso(x₀,x₁)

    path_mode:
      - 'aniso': anisotropic harmonic conditional paths
      - 'free':  straight-line (free particle)
    """
    torch.manual_seed(seed)
    target = torch.tensor(target_data, dtype=torch.float32, device=device)
    n_data = len(target)

    net = VelocityNet(d=2, n_feat=1024, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    total_action = []

    for epoch in range(n_epochs):
        # Sample source and target batches
        x0_batch = torch.randn(batch_size, 2, device=device) * 0.5
        idx = torch.randint(0, n_data, (batch_size,), device=device)
        x1_pool = target[idx]

        # Compute coupling
        if coupling_mode == 'random':
            perm = torch.arange(batch_size, device=device)

        elif coupling_mode == 'euclidean_ot':
            perm_np = ot_coupling(x0_batch, x1_pool, euclidean_action)
            perm = torch.from_numpy(perm_np).to(device)

        elif coupling_mode == 'action_ot':
            cost_fn = lambda a, b: aniso_action(a, b, params)
            perm_np = ot_coupling(x0_batch, x1_pool, cost_fn)
            perm = torch.from_numpy(perm_np).to(device)

        else:
            raise ValueError(f"Unknown coupling: {coupling_mode}")

        x1_batch = x1_pool[perm]

        # Record mean action of this coupling (no grad needed)
        with torch.no_grad():
            if path_mode == 'aniso':
                A = aniso_action(x0_batch, x1_batch, params)  # [N, N]
                batch_act = A.diagonal().mean().item()
            else:
                A = euclidean_action(x0_batch, x1_batch)
                batch_act = A.diagonal().mean().item()
        total_action.append(batch_act)

        # Sample time and compute conditional paths/velocities (vectorized)
        t_batch = torch.rand(batch_size, device=device) * 0.9 + 0.05  # [0.05, 0.95]

        if path_mode == 'aniso':
            xt = aniso_path(t_batch, x0_batch, x1_batch, params)
            ut = aniso_velocity(t_batch, x0_batch, x1_batch, params)
        else:
            xt = free_path(t_batch, x0_batch, x1_batch)
            ut = free_velocity(x0_batch, x1_batch)

        # Train step
        opt.zero_grad()
        pred = net(xt, t_batch)
        loss = loss_fn(pred, ut)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"    [{coupling_mode}+{path_mode}] Epoch {epoch+1}/{n_epochs}, "
                  f"Loss: {loss.item():.4f}, Mean action: {np.mean(total_action[-50:]):.4f}")

    return net, losses, total_action


@torch.no_grad()
def sample_ode(net, n_samples=500, n_steps=200, seed=123, device='cpu'):
    torch.manual_seed(seed)
    dt = 1.0 / n_steps
    x = torch.randn(n_samples, 2, device=device) * 0.5
    snapshots = {0.0: x.cpu().numpy().copy()}
    net.eval()
    for step in range(n_steps):
        t_tensor = torch.full((n_samples,), step * dt, device=device)
        v = net(x, t_tensor)
        x = x + v * dt
        t_now = (step + 1) * dt
        for t_save in [0.25, 0.5, 0.75, 1.0]:
            if abs(t_now - t_save) < dt:
                snapshots[round(t_now, 2)] = x.cpu().numpy().copy()
    return x.cpu().numpy(), snapshots


# ============================================================================
# 7. Evaluation Metrics
# ============================================================================

def compute_w2_approx(samples, target, n_subsample=500, seed=0):
    """Approximate W2 distance via OT on subsamples."""
    rng = np.random.RandomState(seed)
    n = min(n_subsample, len(samples), len(target))
    s = torch.tensor(samples[rng.choice(len(samples), n, replace=False)], dtype=torch.float32)
    t = torch.tensor(target[rng.choice(len(target), n, replace=False)], dtype=torch.float32)
    C = torch.cdist(s, t, p=2) ** 2
    row, col = linear_sum_assignment(C.numpy())
    return float(np.sqrt(C.numpy()[row, col].mean()))


def compute_coverage(samples, target, k=5):
    """Fraction of target modes covered (via nearest-neighbor)."""
    s = torch.tensor(samples, dtype=torch.float32)
    t = torch.tensor(target, dtype=torch.float32)
    # Nearest sample for each target point
    dists = torch.cdist(t, s, p=2) ** 2   # [N_target, N_samples]
    nn_dists = dists.min(dim=1).values.numpy()
    threshold = np.percentile(nn_dists, 90)
    return float(np.mean(nn_dists < threshold))


# ============================================================================
# 8. Visualization
# ============================================================================

def plot_full(target, params, results, save_path=None):
    """
    results: dict of {name: (net, losses, actions, samples, snapshots)}
    """

    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor('#07070d')
    gs = GridSpec(6, 3, figure=fig, hspace=0.35, wspace=0.28)

    BG = '#07070d'
    TXT = '#e0ddd5'
    M1 = '#ff5757'
    M2 = '#57b5ff'
    n_half = len(target) // 2

    COLORS = {
        'random+free':        '#777777',
        'euclidean_ot+free':  '#aaaaaa',
        'random+aniso':       '#ff9944',
        'euclidean_ot+aniso': '#44ddaa',
        'action_ot+aniso':    '#ffc857',
    }
    LABELS = {
        'random+free':        'Random + Free Particle',
        'euclidean_ot+free':  'Euclidean OT + Free Particle',
        'random+aniso':       'Random + Anisotropic',
        'euclidean_ot+aniso': 'Euclidean OT + Anisotropic',
        'action_ot+aniso':    'Action OT + Anisotropic (ours)',
    }

    def sty(ax, title=''):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TXT, fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors=TXT, labelsize=8)
        for s in ax.spines.values():
            s.set_color('#2a2a3a')

    # Potential grid
    xr = np.linspace(-2.5, 2.5, 120)
    yr = np.linspace(-2, 2, 120)
    XX, YY = np.meshgrid(xr, yr)
    Om2 = params.Omega2
    ZZ = np.array([[0.5 * (np.array([XX[i,j], YY[i,j]]) - params.center) @
                     Om2 @ (np.array([XX[i,j], YY[i,j]]) - params.center)
                     for j in range(120)] for i in range(120)])

    # ── Row 0: Setup ──

    # 0,0: Target + potential
    ax = fig.add_subplot(gs[0, 0], facecolor=BG)
    sty(ax, 'Target Data + Potential Contours')
    ax.contourf(XX, YY, ZZ, levels=25, cmap='magma', alpha=0.3)
    ax.contour(XX, YY, ZZ, levels=8, colors='white', linewidths=0.3, alpha=0.3)
    ax.scatter(target[:n_half, 0], target[:n_half, 1], c=M1, s=8, alpha=0.5)
    ax.scatter(target[n_half:, 0], target[n_half:, 1], c=M2, s=8, alpha=0.5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2); ax.set_aspect('equal')

    # 0,1: Anisotropic vs Euclidean cost landscapes
    ax = fig.add_subplot(gs[0, 1], facecolor=BG)
    sty(ax, 'Cost from Origin: Aniso Action vs Euclidean')
    x0_ref = torch.zeros(1, 2)
    grid_pts = torch.tensor(
        np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        ZZ_aniso = aniso_action(x0_ref, grid_pts, params).numpy().reshape(120, 120)
    ZZ_euclid = 0.5 * (XX**2 + YY**2)

    ax.contour(XX, YY, ZZ_euclid, levels=10, colors='#777', linewidths=1, alpha=0.6)
    ax.contourf(XX, YY, ZZ_aniso, levels=25, cmap='inferno', alpha=0.7)
    ax.contour(XX, YY, ZZ_aniso, levels=10, colors='white', linewidths=0.4, alpha=0.5)
    ax.scatter(target[:, 0], target[:, 1], c='white', s=3, alpha=0.15)
    ax.plot(0, 0, 'o', color='lime', ms=8, zorder=5)
    ax.text(0.1, 0.15, 'source', color='lime', fontsize=9)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2); ax.set_aspect('equal')

    # 0,2: Coupling comparison (show paired arrows for one batch)
    ax = fig.add_subplot(gs[0, 2], facecolor=BG)
    sty(ax, 'Coupling Visualization (32 pairs)')
    ax.scatter(target[:, 0], target[:, 1], c='white', s=3, alpha=0.1)

    torch.manual_seed(99)
    n_show_pairs = 32
    x0_show = torch.randn(n_show_pairs, 2) * 0.5
    idx_show = torch.randint(0, len(target), (n_show_pairs,))
    x1_pool_show = torch.tensor(target[idx_show.numpy()], dtype=torch.float32)

    perm_euc = ot_coupling(x0_show, x1_pool_show, euclidean_action)
    perm_act = ot_coupling(x0_show, x1_pool_show, lambda a, b: aniso_action(a, b, params))

    x0_np = x0_show.numpy()
    x1_np = x1_pool_show.numpy()
    for i in range(n_show_pairs):
        x1e = x1_np[perm_euc[i]]
        ax.plot([x0_np[i, 0], x1e[0]], [x0_np[i, 1], x1e[1]],
                '-', color='#44ddaa', alpha=0.3, lw=0.8)
        x1a = x1_np[perm_act[i]]
        ax.plot([x0_np[i, 0], x1a[0]], [x0_np[i, 1], x1a[1]],
                '-', color=COLORS['action_ot+aniso'], alpha=0.5, lw=1.2)

    ax.scatter(x0_np[:, 0], x0_np[:, 1], c='white', s=25, zorder=5, marker='o')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#44ddaa', lw=1.5, alpha=0.6, label='Euclidean OT'),
        Line2D([0], [0], color=COLORS['action_ot+aniso'], lw=2, label='Action OT'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, facecolor='#1a1a2e',
              edgecolor='#333', labelcolor=TXT)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2); ax.set_aspect('equal')

    # ── Row 1: Paths for different configs ──

    configs_paths = ['random+free', 'euclidean_ot+aniso', 'action_ot+aniso']

    for col, cfg in enumerate(configs_paths):
        ax = fig.add_subplot(gs[1, col], facecolor=BG)
        sty(ax, f'Paths: {LABELS[cfg]}')
        ax.contourf(XX, YY, ZZ, levels=15, cmap='magma', alpha=0.15)

        torch.manual_seed(7)
        n_path_show = 15
        x0s = torch.randn(n_path_show, 2) * 0.4
        idxs = torch.randint(0, len(target), (n_path_show,))
        x1_pool_p = torch.tensor(target[idxs.numpy()], dtype=torch.float32)

        coupling, pathtype = cfg.split('+')

        if coupling == 'random':
            perm_p = np.arange(n_path_show)
        elif coupling == 'euclidean_ot':
            perm_p = ot_coupling(x0s, x1_pool_p, euclidean_action)
        elif coupling == 'action_ot':
            perm_p = ot_coupling(x0s, x1_pool_p, lambda a, b: aniso_action(a, b, params))

        x1s = x1_pool_p[perm_p]
        t_pts = torch.linspace(0, 1, 80)

        with torch.no_grad():
            for i in range(n_path_show):
                x0_i = x0s[i:i+1].expand(80, -1)
                x1_i = x1s[i:i+1].expand(80, -1)
                if pathtype == 'aniso':
                    path = aniso_path(t_pts, x0_i, x1_i, params).numpy()
                else:
                    path = free_path(t_pts, x0_i, x1_i).numpy()

                mc = M1 if x1s[i, 1].item() > target[:, 1].mean() else M2
                ax.plot(path[:, 0], path[:, 1], '-', color=mc, alpha=0.5, lw=1.3)
                ax.plot(path[0, 0], path[0, 1], '.', color='white', ms=4)
                ax.plot(path[-1, 0], path[-1, 1], '.', color=mc, ms=5)

        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2); ax.set_aspect('equal')

    # ── Row 2: Training losses and action curves ──

    ax = fig.add_subplot(gs[2, 0], facecolor=BG)
    sty(ax, 'Training Loss')
    for name, (net, losses, actions, _, _) in results.items():
        ax.plot(losses, color=COLORS[name], lw=1.2, alpha=0.8, label=LABELS[name])
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', color=TXT)
    ax.set_ylabel('FM Loss', color=TXT)
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT)

    ax = fig.add_subplot(gs[2, 1], facecolor=BG)
    sty(ax, 'Mean Coupling Action (lower = better pairing)')
    window = 20
    for name, (net, losses, actions, _, _) in results.items():
        smoothed = np.convolve(actions, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, color=COLORS[name], lw=1.5, alpha=0.8, label=LABELS[name])
    ax.set_xlabel('Epoch', color=TXT)
    ax.set_ylabel('Mean S[ψ]', color=TXT)
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT)

    ax = fig.add_subplot(gs[2, 2], facecolor=BG)
    sty(ax, 'Action Distribution per Coupling')
    for name, (_, _, actions, _, _) in results.items():
        ax.hist(actions[-200:], bins=30, color=COLORS[name], alpha=0.4,
                density=True, label=LABELS[name])
    ax.set_xlabel('Action S[ψ]', color=TXT)
    ax.set_ylabel('Density', color=TXT)
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT)

    # ── Row 3: Generated samples ──

    show_configs = ['random+free', 'euclidean_ot+aniso', 'action_ot+aniso']

    for col, cfg in enumerate(show_configs):
        ax = fig.add_subplot(gs[3, col], facecolor=BG)
        sty(ax, f'Samples: {LABELS[cfg]}')
        net, losses, actions, samples, snaps = results[cfg]
        ax.scatter(target[:, 0], target[:, 1], c='white', s=4, alpha=0.12)
        ax.scatter(samples[:, 0], samples[:, 1], c=COLORS[cfg], s=10, alpha=0.45)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')

    # ── Row 4: Density evolution + metrics ──

    ax = fig.add_subplot(gs[4, 0], facecolor=BG)
    sty(ax, 'Density Evolution: Action OT + Aniso')
    _, _, _, _, snaps = results['action_ot+aniso']
    evo_colors = ['#2233aa', '#6644cc', '#aa55dd', '#dd77ee', COLORS['action_ot+aniso']]
    for i, (tk, pts) in enumerate(sorted(snaps.items())):
        if i < len(evo_colors):
            ax.scatter(pts[:, 0], pts[:, 1], c=evo_colors[i], s=3, alpha=0.4,
                       label=f't={tk:.2f}')
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT, ncol=2)

    ax = fig.add_subplot(gs[4, 1], facecolor=BG)
    sty(ax, 'Density Evolution: Euclidean OT + Free')
    _, _, _, _, snaps = results.get('euclidean_ot+free', results.get('random+free'))
    evo_colors2 = ['#2233aa', '#4455aa', '#6677bb', '#8899cc', '#aabbdd']
    for i, (tk, pts) in enumerate(sorted(snaps.items())):
        if i < len(evo_colors2):
            ax.scatter(pts[:, 0], pts[:, 1], c=evo_colors2[i], s=3, alpha=0.4,
                       label=f't={tk:.2f}')
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT, ncol=2)

    ax = fig.add_subplot(gs[4, 2], facecolor=BG)
    sty(ax, 'Quantitative Comparison')
    ax.axis('off')

    metrics_text = []
    metrics_text.append(f"{'Configuration':<32} {'W₂↓':>8} {'Cov↑':>8} {'Loss↓':>8}")
    metrics_text.append("─" * 60)

    for name in show_configs:
        net, losses, actions, samples, _ = results[name]
        w2 = compute_w2_approx(samples, target)
        cov = compute_coverage(samples, target)
        final_loss = np.mean(losses[-50:])
        label = LABELS[name][:30]
        metrics_text.append(f"{label:<32} {w2:>8.4f} {cov:>8.3f} {final_loss:>8.4f}")

    for i, line in enumerate(metrics_text):
        color = TXT if i < 2 else COLORS[show_configs[min(i-2, len(show_configs)-1)]]
        ax.text(0.05, 0.85 - i * 0.12, line, transform=ax.transAxes,
                fontsize=10, color=color, fontfamily='monospace',
                fontweight='bold' if i == 0 else 'normal')

    # ── Row 5: Ablation — effect of ω ratio ──

    ax = fig.add_subplot(gs[5, 0:2], facecolor=BG)
    sty(ax, 'Ablation: Effect of Frequency Ratio ω₂/ω₁')

    ratios = [1.0, 2.0, 3.0, 5.0]
    ratio_colors = ['#ff4444', '#ff8844', '#ffcc44', '#44ff88']

    for ri, ratio in enumerate(ratios):
        p_abl = AnisoParams.from_data(target, omega_ratio=ratio, omega_base=1.5)
        torch.manual_seed(7)
        n_abl = 8
        x0_a = torch.randn(n_abl, 2) * 0.4
        abl_idx = torch.randint(0, len(target), (n_abl,))
        x1_a = torch.tensor(target[abl_idx.numpy()], dtype=torch.float32)
        t_pts = torch.linspace(0, 1, 60)
        with torch.no_grad():
            for i in range(n_abl):
                path = aniso_path(
                    t_pts,
                    x0_a[i:i+1].expand(60, -1),
                    x1_a[i:i+1].expand(60, -1),
                    p_abl,
                ).numpy()
                ax.plot(path[:, 0] + ri * 5.5, path[:, 1], '-',
                        color=ratio_colors[ri], alpha=0.5, lw=1)

        ax.text(ri * 5.5, -2.3, f'ω₂/ω₁ = {ratio:.0f}',
                color=ratio_colors[ri], fontsize=11, ha='center', fontweight='bold')

    ax.scatter(target[:, 0], target[:, 1], c='white', s=2, alpha=0.08)
    for ri in range(len(ratios)):
        ax.scatter(target[:, 0] + ri * 5.5, target[:, 1], c='white', s=2, alpha=0.08)
    ax.set_xlim(-3, 5.5 * len(ratios) - 2)
    ax.set_ylim(-2.8, 2.5)

    ax = fig.add_subplot(gs[5, 2], facecolor=BG)
    sty(ax, 'SNR Across Gap (ω₂ direction)')
    t_s = np.linspace(0.01, 0.99, 200)
    for ri, ratio in enumerate(ratios):
        w2 = 1.5 * ratio
        snr2 = np.sin(w2 * t_s) / (np.sin(w2 * (1 - t_s)) + 1e-10)
        ax.plot(t_s, np.clip(snr2, -10, 10), color=ratio_colors[ri], lw=2,
                label=f'ω₂/ω₁={ratio:.0f}')
    free_snr = t_s / (1 - t_s + 1e-10)
    ax.plot(t_s, free_snr, ':', color='#777', lw=1.5, label='free particle')
    ax.set_ylim(-2, 12)
    ax.set_xlabel('t', color=TXT)
    ax.set_ylabel('SNR₂ = α₂/σ₂', color=TXT)
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor=TXT)

    fig.suptitle('Anisotropic Harmonic Flow Matching\nwith Action-Optimal Transport on Double Moons',
                 color=TXT, fontsize=20, fontweight='bold', y=1.0)
    fig.text(0.5, 0.965,
             f'ω₁={params.omega1:.2f} (along crescents)   '
             f'ω₂={params.omega2:.2f} (across gap)   '
             f'Ω²₁₂={params.Omega2[0,1]:.2f} (coupling)',
             ha='center', color='#888', fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n  Saved to {save_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Action-Optimal Transport + Anisotropic Harmonic Flow Matching")
    print("=" * 65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Data
    target = make_moons(1000, noise=0.07, seed=42)
    params = AnisoParams.from_data(target, omega_ratio=3.0, omega_base=1.5)

    print(f"\n  Ω² = \n{params.Omega2.round(3)}")
    print(f"  Off-diagonal: {params.Omega2[0,1]:.3f}")
    print(f"  ω₁={params.omega1:.2f}, ω₂={params.omega2:.2f}, θ={np.degrees(params.angle):.1f}°")

    # ── Train all configurations ──
    configs = [
        ('random+free',        'random',       'free'),
        ('euclidean_ot+free',  'euclidean_ot', 'free'),
        ('random+aniso',       'random',       'aniso'),
        ('euclidean_ot+aniso', 'euclidean_ot', 'aniso'),
        ('action_ot+aniso',    'action_ot',    'aniso'),
    ]

    results = {}
    for name, coupling, pathtype in configs:
        print(f"\n  ── Training: {name} ──")
        t0 = time.time()
        net, losses, actions = train_fm(
            target, params,
            coupling_mode=coupling, path_mode=pathtype,
            n_epochs=400, batch_size=128, lr=5e-4, seed=42,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

        samples, snaps = sample_ode(net, 600, 200, seed=77, device=device)
        results[name] = (net, losses, actions, samples, snaps)

        w2 = compute_w2_approx(samples, target)
        print(f"    W₂ distance: {w2:.4f}")

    # ── Visualize ──
    print("\n  ── Generating visualization ──")
    plot_full(target, params, results,
              save_path="./action_ot_aniso_moons.png")

    print("\n" + "=" * 65)
    print("  Done!")
    print("=" * 65)
