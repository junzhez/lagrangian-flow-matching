"""Implements Conditional Flow Matcher Losses."""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math
import warnings
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch

from .optimal_transport import OTPlanSampler


@dataclass
class AnisoParams:
    """Anisotropic harmonic oscillator parameters (any dimension d).

    Fit per-eigendirection frequencies via PCA: high-variance directions
    receive low ω (gentle paths); low-variance directions receive high ω
    (tighter, more direct paths).  2-D is just d=2.

    Attributes
    ----------
    omegas  : np.ndarray, shape (d,) — per-eigendirection frequencies
    eigvecs : np.ndarray, shape (d, d) — rows are PCA eigenvectors (descending variance)
    center  : np.ndarray, shape (d,)  — data mean (flat)
    """

    omegas:  "np.ndarray"
    eigvecs: "np.ndarray"
    center:  "np.ndarray"

    def __post_init__(self):
        bad = [int(k) for k, w in enumerate(self.omegas) if math.sin(float(w)) <= 0]
        if bad:
            raise ValueError(
                f"omegas at indices {bad} have sin(ω) ≤ 0. "
                "All omegas must satisfy sin(ω) > 0 (ω ∈ (0, π))."
            )

    @classmethod
    def from_data(
        cls,
        data,
        omega_base: float = 0.8,
        omega_ratio: float = 2.0,
        freq_mode: str = "linear",
    ):
        """Fit from data using PCA.

        Parameters
        ----------
        data : array-like, shape (N, *dims)
            Training samples — will be flattened to (N, d).
        omega_base : float
            Lowest frequency, assigned to the 1st PC (highest variance).
            Default 0.8; sin(0.8) ≈ 0.72.
        omega_ratio : float
            Ratio omega_max / omega_base.  omega_max = omega_base * omega_ratio
            is assigned to the last PC (lowest variance).  Must satisfy
            sin(omega_base * omega_ratio) > 0, i.e. omega_base * omega_ratio < π.
            Default 2.0 → omega_max = 1.6; sin(1.6) ≈ 1.0.
        freq_mode : str
            How to map PCA singular values to frequencies.  Options:

            ``'linear'`` (default)
                Uniformly spaced by index: ``np.linspace(omega_base, omega_max, k)``.
                Backward-compatible with existing checkpoints.

            ``'log'`` (recommended for images / embeddings)
                Log-linear in singular-value ratios.  Frequency of PC i is:
                ``omega_base + (omega_max - omega_base) * log(s_max/s_i) / log(s_max/s_min)``
                This distributes frequencies proportionally to information content
                on a log scale, which matches the power-law decay typical of image
                PCA spectra.  Falls back to ``'linear'`` when all singular values
                are equal or the smallest is ≤ 0.

            ``'power'``
                Exponential in the normalized singular-value position:
                ``omega_base * (omega_max / omega_base) ** (1 - (s_i - s_min) / (s_max - s_min))``.
                Anchors both endpoints (``omegas[0] = omega_base``, ``omegas[-1] = omega_max``).
                Falls back to ``'linear'`` when s_max ≤ 0 or all singular values are equal.

            Null-space directions (when N < d) always receive ``omega_max``
            regardless of ``freq_mode``, since no variance evidence exists for them.
        """
        data = np.asarray(data, dtype=float)
        data_flat = data.reshape(len(data), -1)
        N, d = data_flat.shape
        center = data_flat.mean(0)
        centered = data_flat - center
        # Thin SVD: Vt has shape (min(N, d), d).  When N < d (e.g. CIFAR-10 with a
        # small fit batch), this avoids computing the d×d right-singular-vector matrix
        # and is substantially faster than full_matrices=True.
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)  # s: (k,), Vt: (k, d)
        k = Vt.shape[0]
        if k < d:
            # Complete Vt to a full (d, d) orthonormal basis by appending null-space
            # vectors.  Draw random rows, project out the data subspace, then QR.
            rng = np.random.default_rng(0)
            rand = rng.standard_normal((d - k, d)).astype(float)
            rand -= (rand @ Vt.T) @ Vt   # remove data-subspace components
            Q, _ = np.linalg.qr(rand.T)  # Q: (d, d-k) orthonormal columns
            Vt = np.vstack([Vt, Q.T])    # (d, d)
        omega_max = omega_base * omega_ratio
        omegas = np.empty(d)
        if freq_mode == "linear":
            omegas[:k] = np.linspace(omega_base, omega_max, k)
        elif freq_mode == "log":
            s_max, s_min = s[0], s[k - 1]
            if s_min <= 0 or s_max == s_min:
                omegas[:k] = np.linspace(omega_base, omega_max, k)
            else:
                t = np.log(s_max / s) / np.log(s_max / s_min)  # in [0, 1]
                omegas[:k] = omega_base + (omega_max - omega_base) * t
        elif freq_mode == "power":
            s_max, s_min = s[0], s[k - 1]
            if s_max <= 0 or s_max == s_min:
                omegas[:k] = np.linspace(omega_base, omega_max, k)
            else:
                ratio = (s - s_min) / (s_max - s_min)  # in [0, 1], descending across PCs
                omegas[:k] = omega_base * (omega_max / omega_base) ** (1.0 - ratio)
        else:
            raise ValueError(
                f"freq_mode={freq_mode!r} is not recognised. "
                "Choose 'linear', 'log', or 'power'."
            )
        omegas[k:] = omega_max  # null-space directions have no variance ordering
        return cls(omegas=omegas, eigvecs=Vt, center=center)

    def to_tensors(self, device="cpu"):
        """Return ``(R, w, center)`` as float tensors.

        Returns
        -------
        R      : Tensor, shape (d, d) — eigenvector matrix (rows = eigenvectors)
        w      : Tensor, shape (d,)  — per-dimension frequencies
        center : Tensor, shape (d,)
        """
        R = torch.tensor(self.eigvecs, dtype=torch.float, device=device)
        w = torch.tensor(self.omegas,  dtype=torch.float, device=device)
        c = torch.tensor(self.center,  dtype=torch.float, device=device)
        return R, w, c


def _harmonic_action_cost(
    x0: torch.Tensor, x1: torch.Tensor, omega: float
) -> torch.Tensor:
    """Batched pairwise isotropic harmonic-oscillator action cost matrix.

    Computes the Mehler-kernel exponent for a scalar frequency ω applied
    uniformly across all dimensions (identity eigenbasis, zero center):

        S[i,j] = (ω / 2 sinω) [(‖x₀ᵢ‖² + ‖x₁ⱼ‖²) cosω − 2 ⟨x₀ᵢ, x₁ⱼ⟩]

    This is the classical action of a harmonic oscillator with frequency ω
    connecting x₀ at t=0 to x₁ at t=1.

    Parameters
    ----------
    x0 : Tensor, shape (N0, *dim)
    x1 : Tensor, shape (N1, *dim)
    omega : float
        Harmonic frequency in radians. Must satisfy sin(omega) != 0.

    Returns
    -------
    S : Tensor, shape (N0, N1)
    """
    x0f = x0.reshape(x0.shape[0], -1)  # [N0, d]
    x1f = x1.reshape(x1.shape[0], -1)  # [N1, d]
    coeff = omega / (2.0 * math.sin(omega))
    cos_w = math.cos(omega)
    norm0_sq = (x0f ** 2).sum(-1)       # [N0]
    norm1_sq = (x1f ** 2).sum(-1)       # [N1]
    dot = x0f @ x1f.T                   # [N0, N1]
    return coeff * (cos_w * (norm0_sq[:, None] + norm1_sq[None, :]) - 2.0 * dot)


def _signed_harmonic_action_cost(
    x0: torch.Tensor, x1: torch.Tensor, c: float, eps: float = 1e-6
) -> torch.Tensor:
    """Batched pairwise isotropic action cost for signed curvature c.

    Generalizes ``_harmonic_action_cost`` to c <= 0 by analytic continuation
    (w = sqrt(c) -> i*sqrt(-c) for c < 0, so sin -> i*sinh and cos -> cosh;
    the two factors of i cancel in coeff, leaving a real hyperbolic form):

        c > 0  (elliptic):   coeff = w / (2 sin w),  cos_w = cos(w),  w = sqrt(c)
        c < 0  (hyperbolic): coeff = k / (2 sinh k), cos_w = cosh(k), k = sqrt(-c)
        c = 0  (straight):   coeff = 1/2,            cos_w = 1  (recovers OT-CFM's 1/2||x1-x0||^2)

    coeff > 0 for every c in (-inf, pi^2), so the cross term -2*coeff*<x0,x1>
    always favors large <x0,x1> under minimization: the argmin OT coupling
    equals plain OT-CFM's coupling for every c, including c < 0 (contrast
    with ``_matrix_harmonic_mahalanobis_cost``, whose coupling deliberately
    flips sign for negative isotropic curvature).

    Parameters
    ----------
    x0 : Tensor, shape (N0, *dim)
    x1 : Tensor, shape (N1, *dim)
    c : float
        Signed curvature. Must satisfy c < pi^2.

    Returns
    -------
    S : Tensor, shape (N0, N1)
    """
    x0f = x0.reshape(x0.shape[0], -1)  # [N0, d]
    x1f = x1.reshape(x1.shape[0], -1)  # [N1, d]
    if abs(c) < eps:
        coeff, cos_w = 0.5, 1.0
    elif c > 0:
        w = math.sqrt(c)
        coeff = w / (2.0 * math.sin(w))
        cos_w = math.cos(w)
    else:
        k = math.sqrt(-c)
        coeff = k / (2.0 * math.sinh(k))
        cos_w = math.cosh(k)
    norm0_sq = (x0f ** 2).sum(-1)       # [N0]
    norm1_sq = (x1f ** 2).sum(-1)       # [N1]
    dot = x0f @ x1f.T                   # [N0, N1]
    return coeff * (cos_w * (norm0_sq[:, None] + norm1_sq[None, :]) - 2.0 * dot)


def _aniso_action_cost(
    x0: torch.Tensor, x1: torch.Tensor, params: "AnisoParams"
) -> torch.Tensor:
    """Batched pairwise anisotropic action cost matrix (any dimension d).

    Mehler-kernel exponent S[i,j] in the eigenbasis.
    Input tensors must already be flat (bs, d).

    Parameters
    ----------
    x0 : Tensor, shape (N0, d)
    x1 : Tensor, shape (N1, d)

    Returns
    -------
    S : Tensor, shape (N0, N1)
    """
    R, w, center = params.to_tensors(x0.device)
    x0t = (x0 - center) @ R.T          # [N0, d]
    x1t = (x1 - center) @ R.T          # [N1, d]
    coeff = w / (2 * torch.sin(w))      # [d]
    c_cos = coeff * torch.cos(w)        # [d]
    term0 = (x0t ** 2) @ c_cos          # [N0]
    term1 = (x1t ** 2) @ c_cos          # [N1]
    cross = (x0t * coeff) @ x1t.T       # [N0, N1]
    return term0[:, None] + term1[None, :] - 2 * cross


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method.

    This class implements the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher
    parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
                ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al.

    2023 style target OT conditional flow matching. This class inherits the ConditionalFlowMatcher
    and override the compute_mu_t, compute_sigma_t and compute_conditional_flow functions in order
    to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        References
        ----------
        [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for Schrödinger bridge conditional flow matching method.

    This class implements the SB-CFM methods from [1] and inherits the ConditionalFlowMatcher
    parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and
    sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        r"""Initialize the SchrodingerBridgeConditionalFlowMatcher class.

        It requires the hyper- parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
            we use exact as the default as we found this to perform better
            (more accurate and faster) in practice for reasonable batch sizes.
            We note that as batchsize --> infinity the correct choice is the
            sinkhorn method theoretically.
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) with respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise: bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch entropic OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, shape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """Albergo et al.

    2023 trigonometric interpolants class. This class inherits the ConditionalFlowMatcher and
    override the compute_mu_t and compute_conditional_flow functions in order to compute [3]'s
    trigonometric interpolants.

    [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
    """

    def compute_mu_t(self, x0, x1, t):
        r"""Compute the mean of the probability path (Eq.5) from [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: cos(pi t/2)x0 + sin(pi t/2)x1

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        r"""Compute the conditional vector field similar to [3].

        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(\pi*t/2) x0)

        References
        ----------
        [3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al.
        """
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


class ExactOptimalTransportVariancePreservingConditionalFlowMatcher(
    VariancePreservingConditionalFlowMatcher
):
    """OT-SI: Exact OT minibatch coupling combined with Albergo et al.'s trigonometric
    stochastic interpolant.

    Couples ``(x0, x1)`` via an exact (squared-Euclidean) OT plan and then applies the
    variance-preserving trigonometric path

        mu_t = cos(pi*t/2) * x0 + sin(pi*t/2) * x1
        u_t  = (pi/2) * (cos(pi*t/2) * x1 - sin(pi*t/2) * x0)

    inherited from :class:`VariancePreservingConditionalFlowMatcher`. Mirrors
    :class:`ExactOptimalTransportConditionalFlowMatcher` but with the SI path instead
    of the linear interpolant.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma=sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class HarmonicConditionalFlowMatcher(ConditionalFlowMatcher):
    """Harmonic path conditional flow matcher.

    Uses harmonic (trigonometric) interpolation between x0 and x1:

        mu_t = cos(omega*t)*x0 + sin(omega*t) * (x1 - cos(omega)*x0) / sin(omega)

    Conditional flow (velocity field):

        u_t = -omega*x0*sin(omega*t) + omega*cos(omega*t) * (x1 - cos(omega)*x0) / sin(omega)

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0 for deterministic harmonic path).
    omega : Union[float, int]
        Harmonic interpolation parameter in radians (default 1).
        Must satisfy sin(omega) != 0 (i.e., omega != 0, pi, 2*pi, ...).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, omega: Union[float, int] = 1):
        super().__init__(sigma)
        if abs(math.sin(float(omega))) < 1e-8:
            raise ValueError(
                f"sin(omega) is near zero (omega={omega}). "
                "Choose omega != 0, pi, 2*pi, ... to avoid NaN."
            )
        self.omega = float(omega)

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the harmonic probability path.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: cos(omega*t)*x0 + sin(omega*t)*(x1 - cos(omega)*x0)/sin(omega)
        """
        t = pad_t_like_x(t, x0)
        sin_omega = math.sin(self.omega)
        cos_omega = math.cos(self.omega)
        coeff = (x1 - x0 * cos_omega) / sin_omega
        return x0 * torch.cos(self.omega * t) + coeff * torch.sin(self.omega * t)

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the harmonic conditional vector field.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            unused — harmonic velocity does not depend on xt

        Returns
        -------
        ut : conditional vector field
            -omega*x0*sin(omega*t) + omega*cos(omega*t)*(x1 - cos(omega)*x0)/sin(omega)
        """
        del xt  # harmonic velocity does not depend on xt
        t = pad_t_like_x(t, x0)
        sin_omega = math.sin(self.omega)
        cos_omega = math.cos(self.omega)
        coeff = (x1 - x0 * cos_omega) / sin_omega
        return (
            -self.omega * x0 * torch.sin(self.omega * t)
            + self.omega * coeff * torch.cos(self.omega * t)
        )


class ExactOptimalTransportHarmonicConditionalFlowMatcher(HarmonicConditionalFlowMatcher):
    """OT-CFM with harmonic interpolation paths.

    Combines exact OT minibatch coupling (from ExactOptimalTransportConditionalFlowMatcher)
    with harmonic path interpolation (from HarmonicConditionalFlowMatcher).

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0).
    omega : Union[float, int]
        Harmonic interpolation parameter in radians (default 1).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, omega: Union[float, int] = 1):
        super().__init__(sigma=sigma, omega=omega)
        self.ot_sampler = OTPlanSampler(
            method="exact",
            cost_fn=lambda x0, x1: _harmonic_action_cost(x0, x1, self.omega),
        )

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class SchrodingerBridgeHarmonicConditionalFlowMatcher(HarmonicConditionalFlowMatcher):
    """VP Schrödinger bridge with harmonic oscillator reference.

    All components derive from the same VP-SDE whose noise schedule
    matches frequency ω:
        - Mean: harmonic (trigonometric)
        - Noise: σ·√(σ_t^HO · α_t^HO)  (harmonic bridge noise)
        - Cost: Mehler action
        - Score correction: (ω/2)(cot(ωt) - cot(ω(1-t)))

    At ω=π/2: cosine schedule VP Schrödinger bridge.
    At ω→0: recovers standard SB-CFM (Brownian bridge).
    """

    def __init__(
        self,
        sigma: Union[float, int] = 1.0,
        omega: Union[float, int] = 1,
        ot_method: str = "exact",
    ):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        super().__init__(sigma=sigma, omega=omega)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(
            method=ot_method,
            reg=2 * self.sigma**2,
            # Mehler action: consistent with harmonic reference
            cost_fn=lambda x0, x1: _harmonic_action_cost(x0, x1, self.omega),
        )

    def compute_sigma_t(self, t):
        """σ_t^SB = σ · √(σ_t^HO · α_t^HO)

        Harmonic bridge noise: generalizes σ·√(t(1-t)) to harmonic schedule.
        """
        w = self.omega
        sw = math.sin(w)
        sigma_ho = torch.sin(w * (1 - t)) / sw
        alpha_ho = torch.sin(w * t) / sw
        return self.sigma * torch.sqrt(sigma_ho * alpha_ho + 1e-10)

    def compute_conditional_flow(self, x0, x1, t, xt):
        """u_t = (σ_t'/σ_t)(x_t - μ_t) + μ_t'

        Score correction uses the harmonic bridge derivative:
            σ_t'/σ_t = (ω/2)(cot(ωt) - cot(ω(1-t)))
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)

        # Harmonic velocity: μ_t'
        w = self.omega
        sw = math.sin(w)
        mu_t_dot = (
            -w * torch.cos(w * (1 - t)) / sw * x0
            + w * torch.cos(w * t) / sw * x1
        )

        # Score correction: (ω/2)(cot(ωt) - cot(ω(1-t)))
        # Near t=0/1: cot(θ) → 1/θ; switch at thresh=1e-3 rad where
        # relative error of the approx is < θ²/3 ≈ 3e-7.
        _thresh = 1e-3
        wt = w * t
        w1t = w * (1 - t)
        cot_wt = torch.where(wt.abs() < _thresh, 1.0 / wt, torch.cos(wt) / torch.sin(wt))
        cot_w1t = torch.where(w1t.abs() < _thresh, 1.0 / w1t, torch.cos(w1t) / torch.sin(w1t))
        score_correction = (w / 2) * (cot_wt - cot_w1t)

        return score_correction * (xt - mu_t) + mu_t_dot

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class AnisotropicHarmonicConditionalFlowMatcher(ConditionalFlowMatcher):
    """Anisotropic harmonic flow matcher for data of any dimension d.

    Applies per-eigencomponent sinusoidal interpolation in the PCA
    eigenbasis of the target distribution (2-D is just d=2):

        ψ_t = R.T @ (sin(ω(1-t))/sin(ω) · x̃₀ + sin(ωt)/sin(ω) · x̃₁) + center

    where x̃ = R @ (x_flat − center) are PCA coordinates, ω is a vector of
    per-eigendirection frequencies, and R rows are PCA eigenvectors.

    High-variance PCA directions receive low ω (gentle, nearly-linear paths);
    low-variance directions receive high ω (tighter, sinusoidal paths).

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    aniso_params : AnisoParams
        Geometric parameters fit via ``AnisoParams.from_data(target_data)``.
        Required — there is no sensible default.
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParams" = None):
        super().__init__(sigma)
        if aniso_params is None:
            raise ValueError(
                "aniso_params is required. "
                "Use AnisoParams.from_data(target_data) to compute it."
            )
        self.aniso_params = aniso_params
        self._tc: dict = {}  # device-keyed tensor cache

    def _tensors(self, device):
        """Return (R, w, center, inv_sw, coeff, c_cos) cached per device."""
        key = str(device)
        if key not in self._tc:
            R, w, center = self.aniso_params.to_tensors(device)
            inv_sw = 1.0 / torch.sin(w)
            coeff = w * inv_sw * 0.5
            self._tc[key] = (R, w, center, inv_sw, coeff, coeff * torch.cos(w))
        return self._tc[key]

    def compute_mu_t(self, x0, x1, t):
        shape = x0.shape
        bs = shape[0]
        R, w, center, inv_sw = self._tensors(x0.device)[:4]
        t1d = t.reshape(bs, 1)
        x0t = (x0.reshape(bs, -1) - center) @ R.T
        x1t = (x1.reshape(bs, -1) - center) @ R.T
        st = torch.sin(w * (1 - t1d)) * inv_sw
        at = torch.sin(w * t1d) * inv_sw
        return ((st * x0t + at * x1t) @ R + center).reshape(shape)

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        shape = x0.shape
        bs = shape[0]
        R, w, center, inv_sw = self._tensors(x0.device)[:4]
        t1d = t.reshape(bs, 1)
        x0t = (x0.reshape(bs, -1) - center) @ R.T
        x1t = (x1.reshape(bs, -1) - center) @ R.T
        dst = -w * torch.cos(w * (1 - t1d)) * inv_sw
        dat = w * torch.cos(w * t1d) * inv_sw
        return ((dst * x0t + dat * x1t) @ R).reshape(shape)


class ExactOptimalTransportAnisotropicHarmonicConditionalFlowMatcher(
    AnisotropicHarmonicConditionalFlowMatcher
):
    """Action-OT anisotropic harmonic flow matcher (any dimension d).

    Combines anisotropic harmonic paths with exact minibatch OT coupling
    where the transport cost is the anisotropic action S(x₀, x₁) in the
    PCA eigenbasis, instead of the default ½|x₁ − x₀|².

    High-variance PCA directions (low ω) incur low transport cost; low-variance
    directions (high ω) penalise cross-gap transport, so the coupling naturally
    pairs source points to geometrically nearby target points in the eigenbasis.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    aniso_params : AnisoParams
        Geometric parameters fit via ``AnisoParams.from_data(target_data)``.
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParams" = None):
        super().__init__(sigma=sigma, aniso_params=aniso_params)

        def _cost(x0, x1):
            # x0, x1 are already flat (bs, d) — OTPlanSampler reshapes before calling cost_fn
            R, w, center, _, coeff, c_cos = self._tensors(x0.device)
            x0t = (x0 - center) @ R.T
            x1t = (x1 - center) @ R.T
            term0 = (x0t ** 2) @ c_cos
            term1 = (x1t ** 2) @ c_cos
            cross = (x0t * coeff) @ x1t.T
            return term0[:, None] + term1[None, :] - 2 * cross

        self.ot_sampler = OTPlanSampler(method="exact", cost_fn=_cost)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


def _matrix_pr_coeffs(lam: torch.Tensor, t: torch.Tensor):
    """Per-eigencomponent P(t), R(t) and their t-derivatives for a symmetric
    curvature matrix's eigenvalues ``lam``.

    Each eigenvalue lambda_i of A parameterizes a boundary-value path
    x'' = -lambda_i x, x(0)=x0_i, x(1)=x1_i:
        lambda_i > 0 (elliptic):  P=sin(w(1-t))/sin(w), R=sin(wt)/sin(w), w=sqrt(lambda_i)
        lambda_i < 0 (hyperbolic): P=sinh(w(1-t))/sinh(w), R=sinh(wt)/sinh(w), w=sqrt(-lambda_i)
        lambda_i = 0: P=1-t, R=t (continuous limit of both branches as w -> 0)

    Inputs to sin/sinh/etc. are clamped into their valid domain before the
    call (not the outputs clamped after), since torch.where evaluates both
    branches and a wrong-branch value can otherwise be non-finite.

    Parameters
    ----------
    lam : Tensor, shape (d,)
    t   : Tensor, shape (bs, 1)

    Returns
    -------
    P, R, dP, dR : Tensor, shape (bs, d)
    """
    eps = 1e-6
    pos = lam > eps
    neg = lam < -eps

    w_pos = torch.sqrt(torch.clamp(lam, min=eps))
    sin_w = torch.where(pos, torch.sin(w_pos), torch.ones_like(w_pos))
    P_pos = torch.sin(w_pos * (1 - t)) / sin_w
    R_pos = torch.sin(w_pos * t) / sin_w
    dP_pos = -w_pos * torch.cos(w_pos * (1 - t)) / sin_w
    dR_pos = w_pos * torch.cos(w_pos * t) / sin_w

    w_neg = torch.sqrt(torch.clamp(-lam, min=eps))
    sinh_w = torch.where(neg, torch.sinh(w_neg), torch.ones_like(w_neg))
    P_neg = torch.sinh(w_neg * (1 - t)) / sinh_w
    R_neg = torch.sinh(w_neg * t) / sinh_w
    dP_neg = -w_neg * torch.cosh(w_neg * (1 - t)) / sinh_w
    dR_neg = w_neg * torch.cosh(w_neg * t) / sinh_w

    P_zero = (1 - t) * torch.ones_like(lam)
    R_zero = t * torch.ones_like(lam)
    dP_zero = -torch.ones_like(P_zero)
    dR_zero = torch.ones_like(P_zero)

    P = torch.where(pos, P_pos, torch.where(neg, P_neg, P_zero))
    R = torch.where(pos, R_pos, torch.where(neg, R_neg, R_zero))
    dP = torch.where(pos, dP_pos, torch.where(neg, dP_neg, dP_zero))
    dR = torch.where(pos, dR_pos, torch.where(neg, dR_neg, dR_zero))
    return P, R, dP, dR


def _matrix_harmonic_mahalanobis_cost(x0: torch.Tensor, x1: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Batched pairwise Mahalanobis cost (x1-x0)^T A (x1-x0).

    Parameters
    ----------
    x0 : Tensor, shape (N0, d)
    x1 : Tensor, shape (N1, d)
    A  : Tensor, shape (d, d), symmetric

    Returns
    -------
    S : Tensor, shape (N0, N1)

    Reduces to a positive scalar multiple of squared-Euclidean cost (leaving
    OT-CFM's coupling unchanged) iff A = c*I with c > 0.
    """
    x0f = x0.reshape(x0.shape[0], -1)
    x1f = x1.reshape(x1.shape[0], -1)
    term0 = (x0f * (x0f @ A)).sum(-1)   # [N0]
    term1 = (x1f * (x1f @ A)).sum(-1)   # [N1]
    cross = (x0f @ A) @ x1f.T           # [N0, N1]
    return term0[:, None] + term1[None, :] - 2 * cross


class MatrixHarmonicConditionalFlowMatcher(ConditionalFlowMatcher):
    """Harmonic path conditional flow matcher with a full symmetric curvature matrix A.

    Generalizes HarmonicConditionalFlowMatcher's scalar omega and
    AnisotropicHarmonicConditionalFlowMatcher's always-positive diagonal
    frequency vector to an arbitrary symmetric matrix A with possibly
    mixed-sign eigenvalues: positive eigenvalues give an elliptic
    (sin/cos, contracting) path along that eigendirection, negative
    eigenvalues give a hyperbolic (sinh/cosh, expanding) path, and zero
    eigenvalues give a straight line — see ``_matrix_pr_coeffs``. Path:

        gamma_t = Q [P(t) (Q^T x0) + R(t) (Q^T x1)]

    where A = Q diag(lambda) Q^T.

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0).
    A : array-like, shape (d, d)
        Symmetric curvature matrix. Required — there is no sensible default.
        Fit with ``torchlfm.curvature_fitting``. Must satisfy
        lambda_max(A) < pi^2.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, A=None):
        super().__init__(sigma)
        if A is None:
            raise ValueError(
                "A is required. Use torchlfm.curvature_fitting.fit_straddling_segment "
                "(or a zero matrix) to obtain it."
            )
        A_t = torch.as_tensor(A, dtype=torch.float)
        if A_t.dim() != 2 or A_t.shape[0] != A_t.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {tuple(A_t.shape)}")
        lam = torch.linalg.eigvalsh(A_t)
        if lam.max().item() >= math.pi ** 2 - 1e-6:
            raise ValueError(
                f"lambda_max(A)={lam.max().item():.4f} is not < pi^2. "
                "Clamp A's spectrum (e.g. torchlfm.curvature.clamp_spectrum) before "
                "constructing this matcher."
            )
        self.A = A_t
        self._tc: dict = {}

    def _tensors(self, device):
        """Return (lam, Q) cached per device: A = Q diag(lam) Q^T."""
        key = str(device)
        if key not in self._tc:
            lam, Q = torch.linalg.eigh(self.A.to(device))
            self._tc[key] = (lam, Q)
        return self._tc[key]

    def compute_mu_t(self, x0, x1, t):
        shape = x0.shape
        bs = shape[0]
        lam, Q = self._tensors(x0.device)
        t1d = t.reshape(bs, 1)
        x0t = x0.reshape(bs, -1) @ Q
        x1t = x1.reshape(bs, -1) @ Q
        P, R, _, _ = _matrix_pr_coeffs(lam, t1d)
        return ((P * x0t + R * x1t) @ Q.T).reshape(shape)

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        shape = x0.shape
        bs = shape[0]
        lam, Q = self._tensors(x0.device)
        t1d = t.reshape(bs, 1)
        x0t = x0.reshape(bs, -1) @ Q
        x1t = x1.reshape(bs, -1) @ Q
        _, _, dP, dR = _matrix_pr_coeffs(lam, t1d)
        return ((dP * x0t + dR * x1t) @ Q.T).reshape(shape)


class ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher(MatrixHarmonicConditionalFlowMatcher):
    """OT-coupled matrix-harmonic flow matcher (any dimension d, mixed-sign curvature).

    Couples (x0, x1) via exact OT under the Mahalanobis cost
    (x1-x0)^T A (x1-x0) instead of squared-Euclidean distance. When
    A = c*I with c > 0, this cost is a positive scalar multiple of
    squared-Euclidean distance, so the resulting coupling is identical to
    plain ExactOptimalTransportConditionalFlowMatcher's — only the
    anisotropic part of A reweights the assignment.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    A : array-like, shape (d, d)
        Symmetric curvature matrix, see MatrixHarmonicConditionalFlowMatcher.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, A=None):
        super().__init__(sigma=sigma, A=A)
        self.ot_sampler = OTPlanSampler(
            method="exact",
            cost_fn=lambda x0, x1: _matrix_harmonic_mahalanobis_cost(x0, x1, self.A.to(x0.device)),
        )

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class FieldMatrixHarmonicConditionalFlowMatcher(ConditionalFlowMatcher):
    """Batched per-pair curvature matrix-harmonic flow matcher (recipe
    Stage 3.3: "assemble the interpolant").

    Unlike MatrixHarmonicConditionalFlowMatcher (one A shared by the whole
    matcher, fixed at construction), every pair i in the batch gets its
    own symmetric curvature matrix A_i and centre x_c_i, precomputed once
    per pair -- e.g. via ``torchlfm.curvature_field.CurvatureField.query``
    at the pair's straight midpoint and segment-local time -- and passed
    in on every call. This class does not fit or query a field itself: A
    and x_c are the caller's responsibility to compute and cache (see
    ``curvature_field``'s module docstring for the scope boundary between
    the two modules).

    Path (z0 = x0 - x_c, z1 = x1 - x_c, each expressed in pair i's own
    eigenbasis of A_i):

        gamma_t = x_c + P(t; A_i) z0 + R(t; A_i) z1

    reusing ``_matrix_pr_coeffs`` unmodified (it is already elementwise
    over a batched ``lam: (bs, d)`` against ``t: (bs, 1)``).

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0).
    validate : bool
        If True (default), raise if any pair's lambda_max(A) >= pi^2 in a
        given call. This forces a host/device sync every call (unlike
        MatrixHarmonicConditionalFlowMatcher's construction-time-only
        check, since A varies per call here rather than being fixed at
        construction) -- disable once a training loop trusts its upstream
        field's clamping, to avoid that per-step sync.
    """

    def __init__(self, sigma: Union[float, int] = 0.0, validate: bool = True):
        super().__init__(sigma)
        self.validate = validate

    def _eig_batch(self, A: torch.Tensor):
        """A: (bs, d, d) symmetric -> lam: (bs, d), Q: (bs, d, d)."""
        if A.dim() != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(f"A must have shape (bs, d, d), got {tuple(A.shape)}")
        lam, Q = torch.linalg.eigh(A)
        if self.validate:
            lam_max = lam.max().item()
            if lam_max >= math.pi**2 - 1e-6:
                raise ValueError(
                    f"lambda_max(A)={lam_max:.4f} is not < pi^2 for at least one "
                    "pair in this batch. Clamp upstream (torchlfm.curvature."
                    "clamp_spectrum / CurvatureField.A already does this per "
                    "query -- check the caller's cache), or pass validate=False "
                    "once that is trusted."
                )
        return lam, Q

    @staticmethod
    def _project(x: torch.Tensor, x_c: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """(x - x_c) expressed in each pair's own eigenbasis Q: shape (bs, d)."""
        bs = x.shape[0]
        z = x.reshape(bs, -1) - x_c.reshape(bs, -1)
        return torch.bmm(z.unsqueeze(1), Q).squeeze(1)

    def compute_mu_t(self, x0, x1, t, A, x_c):
        shape = x0.shape
        bs = shape[0]
        lam, Q = self._eig_batch(A)
        t1d = t.reshape(bs, 1)
        z0t = self._project(x0, x_c, Q)
        z1t = self._project(x1, x_c, Q)
        P, R, _, _ = _matrix_pr_coeffs(lam, t1d)
        gt = torch.bmm((P * z0t + R * z1t).unsqueeze(1), Q.transpose(1, 2)).squeeze(1)
        return (gt + x_c.reshape(bs, -1)).reshape(shape)

    def compute_conditional_flow(self, x0, x1, t, xt, A, x_c):
        del xt
        shape = x0.shape
        bs = shape[0]
        lam, Q = self._eig_batch(A)
        t1d = t.reshape(bs, 1)
        z0t = self._project(x0, x_c, Q)
        z1t = self._project(x1, x_c, Q)
        _, _, dP, dR = _matrix_pr_coeffs(lam, t1d)
        # no + x_c term here: the centre is constant along a pair's path
        # (Stage 3's tractability trick), so its time-derivative is zero.
        return torch.bmm((dP * z0t + dR * z1t).unsqueeze(1), Q.transpose(1, 2)).squeeze(1).reshape(shape)

    def sample_location_and_conditional_flow(self, x0, x1, A, x_c, t=None, return_noise=False):
        """Same contract as the base class's method, with A (bs, d, d) and
        x_c (bs, *dim) inserted as required positional arguments. They
        vary per call (a different set of pairs each minibatch), unlike
        MatrixHarmonicConditionalFlowMatcher's constructor-fixed A, so
        they cannot be constructor state here -- Stage 5's own recipe
        language is "cache each pair's (A_pair, x_c)", a caching concern
        for the caller/training harness, not this matcher.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        mu_t = self.compute_mu_t(x0, x1, t, A, x_c)
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x0)
        xt = mu_t + sigma_t * eps
        ut = self.compute_conditional_flow(x0, x1, t, xt, A, x_c)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class SignedCurvatureHarmonicConditionalFlowMatcher(ConditionalFlowMatcher):
    """Isotropic harmonic path parameterized by a single signed curvature c.

    Unifies the repulsive (c < 0), straight (c = 0, recovers OT-CFM exactly)
    and attractive (0 < c < pi^2) regimes as one real-analytic family, by
    reusing ``_matrix_pr_coeffs``'s branch selection with a scalar
    eigenvalue instead of MatrixHarmonicConditionalFlowMatcher's full
    eigendecomposition (so no data dimensionality needs to be known at
    construction time):

        gamma_t = P(t; c)*x0 + R(t; c)*x1

    with P, R given by sin/cos for c > 0, sinh/cosh for c < 0, and (1-t)/t
    (the continuous limit of both) at c = 0.

    Given a target contraction ratio rho for the midpoint (C(1/2) = rho),
    the corresponding c is available in closed form via
    ``torchlfm.curvature.Cfac_to_c(rho)``.

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0).
    c : Union[float, int]
        Signed curvature (default 0.0, the OT-CFM/straight-line case).
        Must satisfy c < pi^2 (first conjugate point).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, c: Union[float, int] = 0.0):
        super().__init__(sigma)
        if c >= math.pi ** 2 - 1e-6:
            raise ValueError(
                f"c={c} is not < pi^2 (first conjugate point); the boundary-value "
                "path is undefined at/beyond this curvature."
            )
        self.c = float(c)

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        lam = torch.tensor(self.c, device=x0.device, dtype=x0.dtype)
        P, R, _, _ = _matrix_pr_coeffs(lam, t)
        return P * x0 + R * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt  # signed-curvature velocity does not depend on xt
        t = pad_t_like_x(t, x0)
        lam = torch.tensor(self.c, device=x0.device, dtype=x0.dtype)
        _, _, dP, dR = _matrix_pr_coeffs(lam, t)
        return dP * x0 + dR * x1


class ExactOptimalTransportSignedCurvatureHarmonicConditionalFlowMatcher(
    SignedCurvatureHarmonicConditionalFlowMatcher
):
    """OT-CFM with signed-curvature harmonic interpolation paths.

    Combines exact OT minibatch coupling with the isotropic signed-curvature
    path (SignedCurvatureHarmonicConditionalFlowMatcher), coupled under the
    true action cost (``_signed_harmonic_action_cost``). Because that cost's
    cross-term coefficient is positive for every c in (-inf, pi^2), the
    resulting OT coupling equals plain ExactOptimalTransportConditionalFlowMatcher's
    for every c, including c < 0 (repulsive) -- unlike
    ExactOptimalTransportMatrixHarmonicConditionalFlowMatcher, whose Mahalanobis
    cost deliberately flips the coupling for negative isotropic curvature.

    Parameters
    ----------
    sigma : Union[float, int]
        Noise standard deviation (default 0.0).
    c : Union[float, int]
        Signed curvature (default 0.0).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, c: Union[float, int] = 0.0):
        super().__init__(sigma=sigma, c=c)
        self.ot_sampler = OTPlanSampler(
            method="exact",
            cost_fn=lambda x0, x1: _signed_harmonic_action_cost(x0, x1, self.c),
        )

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1
