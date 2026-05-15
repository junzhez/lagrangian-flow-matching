# CIFAR-10 experiments using Lagrangian Flow Matching

This directory contains training scripts for unconditional CIFAR-10 generation under multiple flow-matching methods. The default training recipe (single RTX 2080 GPU, 400k steps) reproduces the OT-Harmonic and OT-CFM baselines reported in the project README (see [Tong et al. 2024](https://arxiv.org/abs/2302.00482) for the upstream OT-CFM recipe).

To reproduce the experiments and save the weights, install the requirements from the main repository and then run:

- For the OT-Harmonic Conditional Flow Matching method (recommended):

```bash
python train_cifar10.py --model "otharmonic" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000
```

- For the OT-Conditional Flow Matching method:

```bash
python3 train_cifar10.py --model "otcfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000
```

- For the Independent Conditional Flow Matching (I-CFM) method:

```bash
python3 train_cifar10.py --model "icfm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000
```

- For the original Flow Matching method:

```bash
python3 train_cifar10.py --model "fm" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000
```

Note that you can train all our methods in parallel using multiple GPUs and DistributedDataParallel. You can do this by providing the number of GPUs, setting the parallel flag to True and providing the master address and port in the command line. Please refer to [the official document for the usage](https://pytorch.org/docs/stable/elastic/run.html#usage). As an example:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS_YOU_HAVE train_cifar10_ddp.py --model "otharmonic" --lr 2e-4 --ema_decay 0.9999 --batch_size 128 --total_steps 400001 --save_step 20000 --parallel True --master_addr "MASTER_ADDR" --master_port "MASTER_PORT"
```

To compute the FID from the OT-Harmonic-CFM model at end of training, run:

```bash
python3 compute_fid.py --model "otharmonic" --step 400000 --integration_method dopri5
```

For the other models, change the "otharmonic" argument by "otcfm", "icfm" or "fm". For easy reproducibility of upstream OT-CFM results, you can download the model weights at 400000 iterations here:

- [icfm weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt)

- [otcfm weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt)

- [fm weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/fm_cifar10_weights_step_400000.pt)

To recompute the FID, change the PATH variable with where you have saved the downloaded weights.

## Citation

If you find this code useful in your research, please cite:

```bibtex
@misc{du2026lagrangian,
  title  = {Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design},
  author = {Du, Shukai* and Zhang, Junzhe* and Li, Yiming},
  year   = {2026},
  note   = {*Equal contribution. Preprint forthcoming. https://github.com/junzhez/lagrangian-flow-matching}
}
```

This work builds on the [TorchCFM](https://github.com/atong01/conditional-flow-matching) library by Tong, Fatras et al.; see the [project README](../../../README.md#how-to-cite) for those citations.
