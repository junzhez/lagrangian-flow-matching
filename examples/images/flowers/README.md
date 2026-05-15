# Flowers102 experiments using Lagrangian Flow Matching

This directory contains training scripts for unconditional image generation on the [Oxford 102 Flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) using flow matching methods. Images are trained at a fixed resolution of **128×128**.

## Dataset Setup

The dataset downloads automatically via torchvision on first run. No manual preparation is needed.

## Training

### Single GPU

- For the OT-Harmonic Conditional Flow Matching method (recommended):

```bash
python train_flowers.py --model otharmonic --batch_size 128 --total_steps 200001 --save_step 20000
```

- For the OT-Conditional Flow Matching method:

```bash
python train_flowers.py --model otcfm --batch_size 128 --total_steps 200001 --save_step 20000
```

- For the Anisotropic Harmonic method (fits data-driven frequencies before training):

```bash
python train_flowers.py --model aniso --batch_size 128 --total_steps 200001 --save_step 20000
```

Available `--model` options: `otcfm`, `icfm`, `fm`, `si`, `harmonic`, `otharmonic`, `sbharmonic`, `aniso`, `otaniso`.

### Multi-GPU (DistributedDataParallel)

Use `torchrun` with `train_flowers_ddp.py`. The `--batch_size` flag is the **total** batch size across all GPUs (divided automatically per GPU):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS_YOU_HAVE \
    train_flowers_ddp.py \
    --model otharmonic \
    --batch_size 128 \
    --total_steps 200001 \
    --save_step 20000 \
    --parallel True \
    --master_addr localhost \
    --master_port 12355
```

Please refer to [the official torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#usage) for multi-node setups.

## FID Evaluation

FID computation requires a directory of real Flowers102 images at 128×128. Export the training split first:

```bash
python compute_fid.py \
    --model otharmonic \
    --step 200000 \
    --real_image_dir /path/to/flowers102/images \
    --integration_method dopri5
```

For Euler integration (faster but less accurate):

```bash
python compute_fid.py \
    --model otharmonic \
    --step 200000 \
    --real_image_dir /path/to/flowers102/images \
    --integration_method euler \
    --integration_steps 100
```

## Key Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | 128 | Total batch size |
| `--lr` | 2e-4 | Learning rate |
| `--warmup` | 5000 | LR warmup steps |
| `--total_steps` | 200001 | Total training steps |
| `--ema_decay` | 0.9999 | EMA decay rate |
| `--num_channel` | 256 | UNet base channels |
| `--sigma` | 0.0 | Flow noise std (sbharmonic needs > 0) |
| `--omega` | 1.0 | Frequency for harmonic matchers |

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
