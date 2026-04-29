# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import sys

import torch
from absl import flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 256, help="base channel of UNet")
flags.DEFINE_integer("img_size", 32, help="image resolution used during training: 32, 64, 128, or 256")

# Evaluation
flags.DEFINE_string("input_dir", "./results", help="directory containing model checkpoints")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training step of the checkpoint to evaluate")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="batch size for generation")
flags.DEFINE_string("real_image_dir", None, help="path to real ImageNet images at target resolution (for custom stats)")
flags.DEFINE_string("custom_stats_name", None, help="name for custom stats (default: imagenet{img_size})")

FLAGS(sys.argv)

img_size = FLAGS.img_size
stats_name = FLAGS.custom_stats_name or f"imagenet{img_size}"

# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

attn_res = "16,8" if img_size == 32 else "32,16,8"
new_net = UNetModelWrapper(
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

# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_imagenet{img_size}_weights_step_{FLAGS.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()

# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(new_net, solver=FLAGS.integration_method)


def gen_1_img(unused_latent):
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, img_size, img_size, device=device)
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                new_net,
                x,
                t_span,
                rtol=FLAGS.tol,
                atol=FLAGS.tol,
                method=FLAGS.integration_method,
            )
    traj = traj[-1, :]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img


if FLAGS.real_image_dir:
    print(f"Computing custom stats '{stats_name}' from {FLAGS.real_image_dir}")
    fid.make_custom_stats(
        name=stats_name,
        fdir=FLAGS.real_image_dir,
        mode="clean",
        num=None,
    )

print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name=stats_name,
    batch_size=FLAGS.batch_size_fid,
    dataset_res=img_size,
    num_gen=FLAGS.num_gen,
    dataset_split="custom",
    mode="clean",
)
print()
print("FID has been computed")
print()
print("FID: ", score)
