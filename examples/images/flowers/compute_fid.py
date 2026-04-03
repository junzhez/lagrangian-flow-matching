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

IMG_SIZE = 128  # fixed resolution matching train_flower.py

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Evaluation
flags.DEFINE_string("input_dir", "./results", help="directory containing model checkpoints")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training step of the checkpoint to evaluate")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate for FID")
flags.DEFINE_float("tol", 1e-5, help="integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 256, help="batch size for generation")
flags.DEFINE_string(
    "real_image_dir",
    "",
    help="path to a directory of real Flowers102 images for FID computation",
)

FLAGS(sys.argv)

# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# UNetModelWrapper auto-selects channel_mult=(1,1,2,3,4) for 128x128
new_net = UNetModelWrapper(
    dim=(3, IMG_SIZE, IMG_SIZE),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    dropout=0.1,
).to(device)

# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_flower102_weights_step_{FLAGS.step}.pt"
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
        x = torch.randn(FLAGS.batch_size_fid, 3, IMG_SIZE, IMG_SIZE, device=device)
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


if not FLAGS.real_image_dir:
    raise ValueError("--real_image_dir must be set to a directory of real Flowers102 images")

print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    fdir2=FLAGS.real_image_dir,
    batch_size=FLAGS.batch_size_fid,
    num_gen=FLAGS.num_gen,
    mode="clean",
)
print()
print("FID has been computed")
print()
print("FID: ", score)
