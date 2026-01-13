import argparse, os, glob
import numpy as np
import torch
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.insert(0, current_dir)
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, dir_path)

from sim_gan.gan_models.models import sim_gan_euler   # DCGenerator (SimDCGAN)
from sim_gan.gan_models.models import vanila_gan      # VGenerator (SimVGAN)

def build_generator(gan_type: str, device: torch.device):
    if gan_type == 'SimDCGAN':
        netG = sim_gan_euler.DCGenerator(0)
    elif gan_type == 'SimVGAN':
        netG = vanila_gan.VGenerator(0)
    else:
        raise ValueError(f'Unknown GAN_TYPE: {gan_type}')
    return netG.to(device).eval()

def latest_ckpt(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "checkpoint_epoch_*_iters_*"))
        if not files:
            raise FileNotFoundError(f"No checkpoints in {path}")
        return max(files, key=os.path.getmtime)
    return path  # path is a file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--GAN_TYPE", required=True, choices=["SimDCGAN","SimVGAN"])
    ap.add_argument("--CKPT", required=True, help="Checkpoint file OR directory with checkpoints")
    ap.add_argument("--N", type=int, default=64, help="number of beats to generate")
    ap.add_argument("--OUT_DIR", type=str, default="samples")
    ap.add_argument("--SEED", type=int, default=0)
    ap.add_argument("--DEVICE", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.OUT_DIR, exist_ok=True)
    device = torch.device(args.DEVICE)
    torch.manual_seed(args.SEED); np.random.seed(args.SEED)

    # 1) build generator architecture that matches training
    netG = build_generator(args.GAN_TYPE, device)

    # 2) load checkpoint
    ckpt_path = latest_ckpt(args.CKPT)
    ckpt = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()

    # 3) sample noise and generate
    with torch.no_grad():
        z = torch.randn(args.N, 100, device=device)      # noise dim was 100 in training
        beats = netG(z).cpu().numpy()                    # shape [N, 216]

    # 4) save
    np.save(os.path.join(args.OUT_DIR, "beats.npy"), beats)
    # optional: per-beat CSVs
    for i, b in enumerate(beats[:min(args.N, 16)]):
        np.savetxt(os.path.join(args.OUT_DIR, f"beat_{i:04d}.csv"), b, delimiter=",")

    print(f"Saved {args.N} beats to {args.OUT_DIR} (checkpoint: {ckpt_path})")

if __name__ == "__main__":
    main()
