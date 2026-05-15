<div align="center">

# Lagrangian Flow Matching

### A least-action framework for principled path design

[![pytorch](https://img.shields.io/badge/PyTorch_1.11+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

</div>

## Description

**Lagrangian Flow Matching** generalizes Conditional Flow Matching (CFM) by replacing the linear interpolation between source and target with the analytic trajectory of a harmonic oscillator, and by replacing the Euclidean coupling cost ½‖x₁ − x₀‖² with the corresponding harmonic action. Both pieces are exposed as a single drop-in loss class: [`torchlfm.ExactOptimalTransportHarmonicConditionalFlowMatcher`](./torchlfm/conditional_flow_matching.py).

For a scalar frequency ω ∈ (0, π), the conditional probability path is

```
μ_t(x₀, x₁) = cos(ωt)·x₀ + sin(ωt) · (x₁ − cos(ω)·x₀) / sin(ω)
```

with closed-form conditional velocity

```
u_t(x₀, x₁) = −ω·x₀·sin(ωt) + ω·cos(ωt) · (x₁ − cos(ω)·x₀) / sin(ω).
```

The minibatch coupling π(x₀, x₁) is solved as an exact OT problem under the harmonic-oscillator action

```
S(x₀, x₁) = (ω / 2 sin ω) · [ (‖x₀‖² + ‖x₁‖²) · cos ω − 2 ⟨x₀, x₁⟩ ]
```

so source and target points are paired by least action rather than least Euclidean distance. The result is straighter, lower-action probability paths and higher-fidelity generation under the same simulation-free training objective. The package builds on TorchCFM ([Tong et al. 2024](https://arxiv.org/abs/2302.00482)), so existing CFM / OT-CFM / SF²M code paths continue to work.

## Method overview

`ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=0.0, omega=π/2)` runs in two steps per minibatch:

1. **Couple by least action.** Solve the exact OT plan between `x₀` and `x₁` using the harmonic action `S(x₀, x₁)` above as the pairwise cost matrix.
2. **Interpolate along the harmonic path.** Sample `t ∼ U(0, 1)` and return `(t, x_t = μ_t(x₀, x₁), u_t = u_t(x₀, x₁))` for the standard flow-matching regression loss.

`omega` controls how strongly the harmonic action penalizes long-range transport relative to the Euclidean baseline; the default `π/2` recovers a sinusoidal interpolation with `sin ω = 1`. For data-adaptive per-direction frequencies (PCA-derived eigenbasis, multi-ω Mehler kernel), see [`torchlfm.AnisotropicHarmonicNDConditionalFlowMatcher`](./torchlfm/conditional_flow_matching.py) and [`torchlfm.AnisoParamsND`](./torchlfm/conditional_flow_matching.py).

End-to-end demonstrations:

- 2D toy: [`examples/2D_tutorials/tutorial_8_gaussians_Harmonic_Path.ipynb`](./examples/2D_tutorials/tutorial_8_gaussians_Harmonic_Path.ipynb)
- Image generation: [`examples/images/tutorial_mnist_anisotropic_harmonic.ipynb`](./examples/images/tutorial_mnist_anisotropic_harmonic.ipynb)

## The `torchlfm` package

`torchlfm` exposes the same loss-function abstraction over the choice of conditional distribution `q(z)` that TorchCFM introduced, with the lagrangian variants layered on top:

- `ConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$
- `ExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \pi(x_0, x_1)$ where $\pi$ is an exact OT joint (OT-CFM).
- `TargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ — Lipman et al. 2023 style flow from a standard Gaussian to data.
- `SchrodingerBridgeConditionalFlowMatcher`: entropically regularized OT plan; the basis for SB-CFM and \[SF\]²M.
- `VariancePreservingConditionalFlowMatcher`: variance-preserving trigonometric interpolation (Albergo et al. 2023a).
- `HarmonicConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$ with the harmonic interpolation `μ_t` above (default `omega = π/2`).
- `ExactOptimalTransportHarmonicConditionalFlowMatcher`: combines exact-OT minibatch coupling under the harmonic action `S` with the harmonic interpolation — **the primary lagrangian flow-matching loss**.
- `AnisotropicHarmonicNDConditionalFlowMatcher`: data-adaptive per-direction frequencies via `AnisoParamsND.from_data(...)`; Mehler-kernel cost in the eigenbasis of Ω².

These lagrangian flow-matching variants are demonstrated in the tutorials above.

## How to cite

If you use Lagrangian Flow Matching in your research, please cite:

```bibtex
@misc{du2026lagrangian,
  title  = {Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design},
  author = {Du, Shukai* and Zhang, Junzhe* and Li, Yiming},
  year   = {2026},
  note   = {*Equal contribution. Preprint forthcoming. https://github.com/junzhez/lagrangian-flow-matching}
}
```

### Built on TorchCFM

This work builds directly on the [TorchCFM](https://github.com/atong01/conditional-flow-matching) library by Tong, Fatras, et al. Please also cite their papers when using this code:

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2024.
</summary>

```bibtex
@article{tong2024improving,
  title={Improving and generalizing flow-based generative models with minibatch optimal transport},
  author={Alexander Tong and Kilian FATRAS and Nikolay Malkin and Guillaume Huguet and Yanlei Zhang and Jarrid Rector-Brooks and Guy Wolf and Yoshua Bengio},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=CD9Snc73AW},
  note={Expert Certification}
}
```

</details>

<details>
<summary>
A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023.
</summary>

```bibtex
@article{tong2023simulation,
  title={Simulation-Free Schr{\"o}dinger Bridges via Score and Flow Matching},
  author={Tong, Alexander and Malkin, Nikolay and Fatras, Kilian and Atanackovic, Lazar and Zhang, Yanlei and Huguet, Guillaume and Wolf, Guy and Bengio, Yoshua},
  year={2023},
  journal={arXiv preprint 2307.03672}
}
```

</details>

## Implemented papers

- **Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design** (Du, Zhang & Li 2026, this work) — `torchlfm.ExactOptimalTransportHarmonicConditionalFlowMatcher` (isotropic), `torchlfm.AnisotropicHarmonicNDConditionalFlowMatcher` (data-adaptive)
- Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport (Tong et al. 2024) [Paper](https://arxiv.org/abs/2302.00482)
- Simulation-Free Schrödinger Bridges via Score and Flow Matching (Tong et al. 2023) [Paper](https://arxiv.org/abs/2307.03672)
- Flow Matching for Generative Modeling (Lipman et al. 2023) [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al. 2023) [Paper](https://openreview.net/forum?id=XVjTT1nw5z) [Code](https://github.com/gnobitab/RectifiedFlow.git)
- Building Normalizing Flows with Stochastic Interpolants (Albergo et al. 2023a) [Paper](https://openreview.net/forum?id=li7qeBbCR1t)
- Action Matching: Learning Stochastic Dynamics From Samples (Neklyudov et al. 2022) [Paper](https://arxiv.org/abs/2210.06662) [Code](https://github.com/necludov/jam)
- Multisample Flow Matching: Straightening Flows with Minibatch Couplings (Pooladian et al. 2023) [Paper](https://arxiv.org/abs/2304.14772)
- Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees (Jolicoeur-Martineau et al.) [Paper](https://arxiv.org/abs/2309.09968) [Code](https://github.com/SamsungSAILMontreal/ForestDiffusion)

## Installation

```bash
# clone project
git clone https://github.com/junzhez/lagrangian-flow-matching.git
cd lagrangian-flow-matching

# [OPTIONAL] create conda environment
conda create -n torchlfm python=3.10
conda activate torchlfm

# install pytorch according to https://pytorch.org/get-started/

# install requirements + the package in editable mode
pip install -r requirements.txt
pip install -e .
```

To run the Jupyter notebooks:

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=torchlfm
# launch notebooks under the torchlfm kernel
```

## Quick start

```python
import torch
from torchlfm import ExactOptimalTransportHarmonicConditionalFlowMatcher

x0 = torch.randn(256, 2)             # source samples
x1 = torch.randn(256, 2) + 3.0       # target samples

fm = ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=0.0)  # omega = π/2 by default
t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
# `xt` is the harmonic interpolant at time `t`; `ut` is the conditional velocity to regress.
```

For data-adaptive per-direction frequencies, swap in `torchlfm.AnisotropicHarmonicNDConditionalFlowMatcher` (see `torchlfm/conditional_flow_matching.py`).

## Project structure

```
.
├── torchlfm/                     <- Lagrangian flow matching: anisotropic action + couplings + reusable FM losses
│   ├── conditional_flow_matching.py
│   ├── optimal_transport.py
│   ├── utils.py
│   └── models/                   <- MLP and U-Net architectures
├── examples/
│   ├── 2D_tutorials/             <- 2D toy experiments and notebooks
│   ├── images/                   <- MNIST / CIFAR-10 / Flowers / ImageNet training scripts
│   ├── single_cell/              <- Single-cell trajectory inference
│   └── tabular/                  <- Tabular generation (XGBoost CFM)
├── runner/                       <- Lightning + Hydra training harness (legacy V0)
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Contributions

- Lagrangian flow matching: [Junzhe Zhang](https://github.com/junzhez), [Shukai Du](https://shukaidu.github.io/), Yiming Li
- Original TorchCFM library and CFM/OT-CFM/SF²M implementations: [Alexander Tong](http://alextong.net), [Kilian Fatras](http://kilianfatras.github.io)

Suggestions and pull requests are welcome. Before opening an issue, please confirm:

- The problem still exists on the current `main` branch.
- Your Python dependencies are up to date.

## License

Lagrangian Flow Matching is released under the MIT License. See [`LICENSE`](./LICENSE) for the full text.

```
MIT License

Copyright (c) 2026 Junzhe Zhang, Shukai Du, Yiming Li
Copyright (c) 2023 Alexander Tong (TorchCFM, on which this work builds)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
