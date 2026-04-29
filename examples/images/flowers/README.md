# Flowers102 experiments using TorchCFM

This directory contains training scripts for unconditional image generation on the [Oxford 102 Flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) using flow matching methods. Images are trained at a fixed resolution of **128×128**.

## Dataset Setup

The dataset downloads automatically via torchvision on first run. No manual preparation is needed.

## Training

### Single GPU

- For the OT-Conditional Flow Matching method:

```bash
python train_flowers.py --model otcfm --batch_size 128 --total_steps 200001 --save_step 20000
```

- For the OT-Harmonic Conditional Flow Matching method:

```bash
python train_flowers.py --model otharmonic --batch_size 128 --total_steps 200001 --save_step 20000
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
    --model otcfm \
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
    --model otcfm \
    --step 200000 \
    --real_image_dir /path/to/flowers102/images \
    --integration_method dopri5
```

For Euler integration (faster but less accurate):

```bash
python compute_fid.py \
    --model otcfm \
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

If you find this code useful in your research, please cite the following papers (expand for BibTeX):

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
</summary>

```bibtex
@article{tong2023improving,
  title={Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author={Tong, Alexander and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and {Rector-Brooks}, Jarrid and Fatras, Kilian and Wolf, Guy and Bengio, Yoshua},
  year={2023},
  journal={arXiv preprint 2302.00482}
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
