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
    """Parameters for a 2-D anisotropic harmonic oscillator.

    Attributes
    ----------
    omega1 : float
        High-frequency axis (applied to the small-variance / constrained direction).
    omega2 : float
        Low-frequency axis (applied to the large-variance / free direction).
    angle : float
        Rotation angle of the eigenbasis in radians.
    center : np.ndarray, shape (2,)
        Data center point.
    """

    omega1: float = 1.0
    omega2: float = 2.0
    angle: float = 0.0
    center: "np.ndarray" = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        for name, w in [("omega1", self.omega1), ("omega2", self.omega2)]:
            if math.sin(w) <= 0:
                raise ValueError(
                    f"{name}={w:.4f} is invalid: sin({name}) = {math.sin(w):.4f} ≤ 0. "
                    "Anisotropic harmonic paths require sin(ω) > 0 (ω ∈ (0, π))."
                )

    @property
    def R(self):
        """2x2 rotation matrix from angle."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    @property
    def Omega2(self):
        """Hessian matrix R.T @ diag([ω₁², ω₂²]) @ R.

        After rotation by ``angle``, dim-0 of the eigenbasis is the
        small-variance axis (receives ω₁, the high frequency) and dim-1
        is the large-variance axis (receives ω₂, the low frequency).
        """
        R = self.R
        return R.T @ np.diag([self.omega1**2, self.omega2**2]) @ R

    @classmethod
    def from_data(cls, data, omega_ratio=2.0, omega_base=1.0):
        """Fit AnisoParams to data by estimating the covariance eigenbasis.

        Parameters
        ----------
        data : array-like, shape (N, 2)
        omega_ratio : float
            Ratio ω₁/ω₂ (default 2.0).  omega1 = omega_base * omega_ratio.
            Must satisfy ``sin(omega_base * omega_ratio) > 0``.
        omega_base : float
            Base frequency ω₂ (the low-frequency / large-variance value, default 1.0).
        """
        data = np.asarray(data, dtype=float)
        center = data.mean(0)
        cov = np.cov(data.T)
        _, eigvecs = np.linalg.eigh(cov)
        # Normalize eigenvector sign so the largest-magnitude component is
        # positive — ensures a deterministic rotation angle across data samples.
        ev = eigvecs[:, 0].copy()
        if ev[np.argmax(np.abs(ev))] < 0:
            ev = -ev
        angle = np.arctan2(ev[1], ev[0])
        return cls(
            omega1=omega_base * omega_ratio,  # small variance → high ω
            omega2=omega_base,                # large variance → low ω
            angle=float(angle),
            center=center,
        )
        
    def to_tensors(self, device="cpu"):
        """Return ``(R, w, center)`` as float32 torch tensors.

        Returns
        -------
        R : Tensor, shape (2, 2)
        w : Tensor, shape (2,)  — [ω₁, ω₂]
        center : Tensor, shape (2,)
        """
        R = torch.tensor(self.R, dtype=torch.float32, device=device)
        w = torch.tensor([self.omega1, self.omega2], dtype=torch.float32, device=device)
        center = torch.tensor(self.center, dtype=torch.float32, device=device)
        return R, w, center


@dataclass
class AnisoParamsND:
    """N-dimensional anisotropic harmonic oscillator parameters.

    Generalisation of ``AnisoParams`` to arbitrary dimension d via PCA.
    High-variance PCA directions receive low ω (gentle paths);
    low-variance directions receive high ω (tighter, more direct paths).

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
    def from_data(cls, data, omega_base: float = 0.8, omega_ratio: float = 2.0):
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

        Frequencies are linearly spaced from omega_base (index 0) to
        omega_base * omega_ratio (index d − 1).
        """
        data = np.asarray(data, dtype=float)
        data_flat = data.reshape(len(data), -1)
        N, d = data_flat.shape
        center = data_flat.mean(0)
        centered = data_flat - center
        # Thin SVD: Vt has shape (min(N, d), d).  When N < d (e.g. CIFAR-10 with a
        # small fit batch), this avoids computing the d×d right-singular-vector matrix
        # and is substantially faster than full_matrices=True.
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)  # (min(N, d), d)
        k = Vt.shape[0]
        if k < d:
            # Complete Vt to a full (d, d) orthonormal basis by appending null-space
            # vectors.  Draw random rows, project out the data subspace, then QR.
            rng = np.random.default_rng(0)
            rand = rng.standard_normal((d - k, d)).astype(float)
            rand -= (rand @ Vt.T) @ Vt   # remove data-subspace components
            Q, _ = np.linalg.qr(rand.T)  # Q: (d, d-k) orthonormal columns
            Vt = np.vstack([Vt, Q.T])    # (d, d)
        omegas = np.linspace(omega_base, omega_base * omega_ratio, d)
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


def _aniso_action_cost_nd(
    x0: torch.Tensor, x1: torch.Tensor, params: "AnisoParamsND"
) -> torch.Tensor:
    """Batched pairwise N-D anisotropic action cost matrix.

    Same Mehler-kernel formula as ``_aniso_action_cost`` but for arbitrary d.
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


def _aniso_action_cost(
    x0: torch.Tensor, x1: torch.Tensor, params: "AnisoParams"
) -> torch.Tensor:
    """Batched pairwise anisotropic action cost matrix.

    Computes the Mehler-kernel exponent S[i,j]:
        S[i,j] = Σₖ (ωₖ / 2 sinωₖ) [(x̃₀ᵢᵏ² + x̃₁ⱼᵏ²) cosωₖ - 2 x̃₀ᵢᵏ x̃₁ⱼᵏ]

    where x̃ = R @ (x − center) are eigenbasis coordinates.

    Parameters
    ----------
    x0 : Tensor, shape (N0, 2)
    x1 : Tensor, shape (N1, 2)
    params : AnisoParams

    Returns
    -------
    S : Tensor, shape (N0, N1)
    """
    R, w, center = params.to_tensors(x0.device)
    x0t = (x0 - center) @ R.T      # [N0, 2]
    x1t = (x1 - center) @ R.T      # [N1, 2]
    coeff = w / (2 * torch.sin(w))  # [2]
    c_cos = coeff * torch.cos(w)    # [2]
    term0 = (x0t ** 2) @ c_cos      # [N0]
    term1 = (x1t ** 2) @ c_cos      # [N1]
    cross = (x0t * coeff) @ x1t.T   # [N0, N1]
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
        Harmonic interpolation parameter in radians (default pi/2).
        Must satisfy sin(omega) != 0 (i.e., omega != 0, pi, 2*pi, ...).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, omega: Union[float, int] = math.pi / 2):
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
        Harmonic interpolation parameter in radians (default pi/2).
    """

    def __init__(self, sigma: Union[float, int] = 0.0, omega: Union[float, int] = math.pi / 2):
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
        omega: Union[float, int] = math.pi / 2,
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
    """Anisotropic harmonic flow matcher for 2-D data.

    Uses per-dimension sinusoidal interpolation in the eigenbasis of Ω²:

        ψ_t = R.T @ (sin(ω(1-t))/sin(ω) · x̃₀ + sin(ωt)/sin(ω) · x̃₁) + center

    where x̃ = R @ (x − center) are coordinates in the rotated eigenbasis,
    and ω = [ω₁, ω₂] are per-axis frequencies.  This generalises
    ``HarmonicConditionalFlowMatcher`` to geometry-aware anisotropic paths.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0 — deterministic path).
    aniso_params : AnisoParams, optional
        Geometric parameters.  Defaults to ``AnisoParams()`` (ω₁=1.5, ω₂=4.5,
        no rotation, centered at origin).  Use ``AnisoParams.from_data`` to
        fit parameters to your target distribution.
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParams" = None):
        super().__init__(sigma)
        self.aniso_params = aniso_params if aniso_params is not None else AnisoParams()

    def compute_mu_t(self, x0, x1, t):
        """Anisotropic sinusoidal mean path ψ_t."""
        t = pad_t_like_x(t, x0)
        R, w, center = self.aniso_params.to_tensors(x0.device)
        x0t = (x0 - center) @ R.T
        x1t = (x1 - center) @ R.T
        st = torch.sin(w * (1 - t)) / torch.sin(w)
        at = torch.sin(w * t) / torch.sin(w)
        return (st * x0t + at * x1t) @ R + center

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Analytic velocity ψ̇_t (does not depend on xt)."""
        del xt
        t = pad_t_like_x(t, x0)
        R, w, center = self.aniso_params.to_tensors(x0.device)
        x0t = (x0 - center) @ R.T
        x1t = (x1 - center) @ R.T
        dst = -w * torch.cos(w * (1 - t)) / torch.sin(w)
        dat =  w * torch.cos(w * t) / torch.sin(w)
        return (dst * x0t + dat * x1t) @ R


class ExactOptimalTransportAnisotropicHarmonicConditionalFlowMatcher(
    AnisotropicHarmonicConditionalFlowMatcher
):
    """Action-OT anisotropic harmonic flow matcher.

    Combines anisotropic harmonic paths (``AnisotropicHarmonicConditionalFlowMatcher``)
    with exact minibatch OT coupling where the transport cost is the anisotropic
    action S(x₀, x₁) instead of the default ½|x₁ − x₀|².

    The anisotropic action penalises cross-gap transport (high ω₂) more than
    along-manifold transport (low ω₁), so the coupling naturally pairs source
    points to geometrically nearby target points.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    aniso_params : AnisoParams, optional
        Geometric parameters (see ``AnisotropicHarmonicConditionalFlowMatcher``).
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParams" = None):
        super().__init__(sigma=sigma, aniso_params=aniso_params)
        p = self.aniso_params
        self.ot_sampler = OTPlanSampler(
            method="exact",
            cost_fn=lambda x0, x1: _aniso_action_cost(x0, x1, p),
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


class AnisotropicHarmonicNDConditionalFlowMatcher(ConditionalFlowMatcher):
    """N-dimensional anisotropic harmonic flow matcher.

    Generalises ``AnisotropicHarmonicConditionalFlowMatcher`` to arbitrary
    dimension *d* by applying per-eigencomponent sinusoidal interpolation in
    the PCA eigenbasis of the target distribution:

        ψ_t = R.T @ (sin(ω(1-t))/sin(ω) · x̃₀ + sin(ωt)/sin(ω) · x̃₁) + center

    where x̃ = R @ (x_flat − center) are PCA coordinates, ω is a vector of
    per-eigendirection frequencies, and R rows are PCA eigenvectors.

    High-variance PCA directions receive low ω (gentle, nearly-linear paths);
    low-variance directions receive high ω (tighter, sinusoidal paths).

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    aniso_params : AnisoParamsND
        Geometric parameters fit via ``AnisoParamsND.from_data(target_data)``.
        Required — there is no sensible default for N-D data.
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParamsND" = None):
        super().__init__(sigma)
        if aniso_params is None:
            raise ValueError(
                "aniso_params is required. "
                "Use AnisoParamsND.from_data(target_data) to compute it."
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


class ExactOptimalTransportAnisotropicHarmonicNDConditionalFlowMatcher(
    AnisotropicHarmonicNDConditionalFlowMatcher
):
    """Action-OT N-dimensional anisotropic harmonic flow matcher.

    Combines N-D anisotropic harmonic paths with exact minibatch OT coupling
    where the transport cost is the N-D anisotropic action S(x₀, x₁) in the
    PCA eigenbasis, instead of the default ½|x₁ − x₀|².

    High-variance PCA directions (low ω) incur low transport cost; low-variance
    directions (high ω) penalise cross-gap transport, so the coupling naturally
    pairs source points to geometrically nearby target points in the eigenbasis.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 0.0).
    aniso_params : AnisoParamsND
        Geometric parameters fit via ``AnisoParamsND.from_data(target_data)``.
    """

    def __init__(self, sigma: float = 0.0, aniso_params: "AnisoParamsND" = None):
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
