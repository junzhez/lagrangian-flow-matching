# ImageNet experiments using TorchCFM

This directory contains training scripts for unconditional image generation on ImageNet using flow matching methods. Supported resolutions: **32, 64, 128, 256**.

## Dataset Setup

Download and prepare ImageNet using the provided script:

```bash
python download_imagenet.py --data_dir ./data/imagenet
```

The script expects a `train/` subdirectory under `--data_dir` in `ImageFolder` format.

## Training

### Single GPU

- For the OT-Conditional Flow Matching method:

```bash
python train_imagenet.py --model otcfm --img_size 64 --batch_size 256 --total_steps 500001 --save_step 20000
```

- For the OT-Harmonic Conditional Flow Matching method:

```bash
python train_imagenet.py --model otharmonic --img_size 64 --batch_size 256 --total_steps 500001 --save_step 20000
```

- For the Anisotropic Harmonic method (fits data-driven frequencies before training):

```bash
python train_imagenet.py --model aniso --img_size 64 --batch_size 256 --total_steps 500001 --save_step 20000
```

Available `--model` options: `otcfm`, `icfm`, `fm`, `si`, `harmonic`, `otharmonic`, `sbharmonic`, `aniso`, `otaniso`.

### Multi-GPU (DistributedDataParallel)

Use `torchrun` with `train_imagenet_ddp.py`. The `--batch_size` flag is the **total** batch size across all GPUs (divided automatically per GPU):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS_YOU_HAVE \
    train_imagenet_ddp.py \
    --model otcfm \
    --img_size 64 \
    --batch_size 256 \
    --total_steps 500001 \
    --save_step 20000 \
    --parallel True \
    --master_addr localhost \
    --master_port 12355
```

Please refer to [the official torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#usage) for multi-node setups.

## FID Evaluation

To compute the FID score, first build custom stats from real ImageNet images at the target resolution:

```bash
python compute_fid.py \
    --model otcfm \
    --img_size 64 \
    --step 500000 \
    --real_image_dir /path/to/imagenet/val \
    --integration_method dopri5
```

On subsequent runs the custom stats are cached, so `--real_image_dir` can be omitted:

```bash
python compute_fid.py --model otcfm --img_size 64 --step 500000 --integration_method dopri5
```

For Euler integration (faster but less accurate):

```bash
python compute_fid.py --model otcfm --img_size 64 --step 500000 --integration_method euler --integration_steps 100
```

## Key Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--img_size` | 32 | Image resolution (32, 64, 128, or 256) |
| `--batch_size` | 256 | Total batch size |
| `--lr` | 1e-4 | Learning rate |
| `--warmup` | 20000 | LR warmup steps |
| `--total_steps` | 500001 | Total training steps |
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
