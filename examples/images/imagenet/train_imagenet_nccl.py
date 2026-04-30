# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import math
import os

import numpy as np
import torch
from absl import app, flags
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms
from tqdm import trange
from utils_imagenet import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    AnisoParamsND,
    AnisotropicHarmonicNDConditionalFlowMatcher,
    ConditionalFlowMatcher,
    ExactOptimalTransportAnisotropicHarmonicNDConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    ExactOptimalTransportHarmonicConditionalFlowMatcher,
    HarmonicConditionalFlowMatcher,
    SchrodingerBridgeHarmonicConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_float("omega", 1, help="omega parameter for harmonic flow matchers")
flags.DEFINE_float("omega_base", 0.8, help="base frequency for ND anisotropic flow matchers")
flags.DEFINE_float("omega_ratio", 2.0, help="frequency ratio for ND anisotropic flow matchers")
flags.DEFINE_string("freq_mode", "linear", help="frequency assignment mode for AnisoParamsND: 'linear', 'log', or 'power'")
flags.DEFINE_integer("aniso_fit_batches", 10, help="number of batches used to fit AnisoParamsND")
flags.DEFINE_float("sigma", 0.0, help="noise std for flow matcher (sbharmonic requires sigma > 0, defaults to 1.0)")
flags.DEFINE_string("ot_method", "exact", help="OT method for sbharmonic: 'exact' or 'sinkhorn'")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 256, help="base channel of UNet")

# Dataset
flags.DEFINE_integer("img_size", 32, help="image resolution: 32, 64, 128, or 256")
flags.DEFINE_string(
    "data_dir",
    "./data/imagenet",
    help="path to ImageNet root directory (must contain a train/ subdirectory)",
)

# Training
flags.DEFINE_float("lr", 1e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 500001, help="total training steps")
flags.DEFINE_integer("warmup", 20000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 256, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


def warmup_lr(step):
    if step < FLAGS.warmup:
        return step / FLAGS.warmup
    return max(0.0, (FLAGS.total_steps - step) / (FLAGS.total_steps - FLAGS.warmup))


def fit_aniso_params(dataloader):
    samples = []
    for i, (x, _) in enumerate(dataloader):
        if i >= FLAGS.aniso_fit_batches:
            break
        samples.append(x.numpy())
    data = np.concatenate(samples, axis=0)
    return AnisoParamsND.from_data(data, omega_base=FLAGS.omega_base, omega_ratio=FLAGS.omega_ratio, freq_mode=FLAGS.freq_mode)


def train(device, is_distributed, argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    img_size = FLAGS.img_size
    if img_size not in (32, 64, 128, 256):
        raise ValueError(f"--img_size must be 32, 64, 128, or 256, got {img_size}")

    if is_distributed:
        batch_size_per_gpu = FLAGS.batch_size // dist.get_world_size()
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # DATASETS/DATALOADER
    transform = transforms.Compose(
        [
            transforms.Resize(img_size + 16),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dir = os.path.join(FLAGS.data_dir, "train")
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    sampler = DistributedSampler(dataset) if is_distributed else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=False if is_distributed else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # Calculate epochs needed to cover total_steps
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)

    # MODELS
    # UNetModelWrapper auto-selects channel_mult based on image_size:
    #   64  -> (1, 2, 3, 4)
    #   128 -> (1, 1, 2, 3, 4)
    #   256 -> (1, 1, 2, 2, 4, 4)
    attn_res = "16,8" if img_size == 32 else "32,16,8"
    net_model = UNetModelWrapper(
        dim=(3, img_size, img_size),
        num_res_blocks=3,
        num_channels=FLAGS.num_channel,
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=attn_res,
        dropout=0.0,
        use_scale_shift_norm=True,
        resblock_updown=True,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if is_distributed:
        net_model = DistributedDataParallel(net_model, device_ids=[device.index])
        ema_model = DistributedDataParallel(ema_model, device_ids=[device.index])

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = FLAGS.sigma
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "harmonic":
        FM = HarmonicConditionalFlowMatcher(sigma=sigma, omega=FLAGS.omega)
    elif FLAGS.model == "otharmonic":
        FM = ExactOptimalTransportHarmonicConditionalFlowMatcher(sigma=sigma, omega=FLAGS.omega)
    elif FLAGS.model == "sbharmonic":
        sb_sigma = sigma if sigma > 0 else 1.0
        FM = SchrodingerBridgeHarmonicConditionalFlowMatcher(
            sigma=sb_sigma, omega=FLAGS.omega, ot_method=FLAGS.ot_method
        )
    elif FLAGS.model == "aniso":
        fit_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        aniso_params = fit_aniso_params(fit_loader)
        FM = AnisotropicHarmonicNDConditionalFlowMatcher(sigma=sigma, aniso_params=aniso_params)
    elif FLAGS.model == "otaniso":
        fit_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        aniso_params = fit_aniso_params(fit_loader)
        FM = ExactOptimalTransportAnisotropicHarmonicNDConditionalFlowMatcher(
            sigma=sigma, aniso_params=aniso_params
        )
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of "
            "['otcfm', 'icfm', 'fm', 'si', 'harmonic', 'otharmonic', 'sbharmonic', 'aniso', 'otaniso']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    global_step = 0

    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            if sampler is not None:
                sampler.set_epoch(epoch)

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for _ in step_pbar:
                    global_step += 1

                    optim.zero_grad()
                    x1 = next(datalooper).to(device)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                    vt = net_model(t, xt)
                    loss = torch.mean((vt - ut) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)

                    if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0:
                        if not is_distributed or dist.get_rank() == 0:
                            generate_samples(
                                net_model, is_distributed, savedir, global_step, net_="normal", img_size=img_size
                            )
                            generate_samples(
                                ema_model, is_distributed, savedir, global_step, net_="ema", img_size=img_size
                            )
                            torch.save(
                                {
                                    "net_model": net_model.state_dict(),
                                    "ema_model": ema_model.state_dict(),
                                    "sched": sched.state_dict(),
                                    "optim": optim.state_dict(),
                                    "step": global_step,
                                },
                                savedir + f"{FLAGS.model}_imagenet{img_size}_weights_step_{global_step}.pt",
                            )

                    if global_step >= FLAGS.total_steps:
                        return


def main(argv):
    local_rank = os.environ.get("LOCAL_RANK")
    is_distributed = local_rank is not None

    if is_distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train(device=device, is_distributed=is_distributed, argv=argv)
    finally:
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
