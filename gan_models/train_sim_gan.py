import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob, textwrap
#import neurokit2 as nk


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# TODO: Remove
#sys.path.insert(0, current_dir)
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, dir_path)

from sim_gan.data_reader import ecg_dataset_pytorch
from tensorboardX import SummaryWriter
from sim_gan.gan_models.models import sim_gan_euler
from sim_gan.dynamical_model import equations
from sim_gan.dynamical_model.ode_params import ODEParams
import math
import logging
from sim_gan.dynamical_model import typical_beat_params
from sim_gan.gan_models.models import vanila_gan
from sim_gan.data_reader import dataset_configs
import argparse
from sim_gan.data_reader.transforms_resample import ResampleTo360
from sim_gan.data_reader.ecg_dataset_copd_from_csv import EcgCOPDBeatsFromCSV
from sim_gan.data_reader.ecg_dataset_copd_from_csv import _read_copd_csv
from sim_gan.gan_models.models.windowed_discriminator import WindowedDiscriminator
from sim_gan.gan_models.train_utils import _sample_z_seq, assemble_window_from_beats, make_real_windows_grouped
from sim_gan.dynamical_model.Euler.euler_window import euler_loss_window

_full_cache = {}  # path -> dict(ecg_128, magnification)
try:
    import neurokit2 as nk
except Exception:
    nk = None

# vital_sqi (simple SQIs we can compute on short windows)
try:
    from vital_sqi.sqi.standard_sqi import kurtosis_sqi
except Exception:
    kurtosis_sqi = None

from scipy.signal import correlate

parser = argparse.ArgumentParser(description='Train an SIM ECG GAN of type Vanilla GAN or DCGAN.', )
parser.add_argument('--GAN_TYPE', type=str, help='Type of gan, either SimDCGAN or SimVGAN.',
                    required=True, choices=['SimVGAN', 'SimDCGAN'])
parser.add_argument('--MODEL_DIR', type=str, help='Directory to write summaries and checkpoints.',
                    required=True)
#TODO: uncomment "required=True"
parser.add_argument('--BEAT_TYPE', type=str, help='Type of heartbeat to learn to generate..',
                    required=True,  
                    choices=['N', 'S', 'V', 'F'])
parser.add_argument('--BATCH_SIZE', type=int, help='batch size.',
                    required=True)
                 #   )
parser.add_argument('--NUM_ITERATIONS', type=int, help='Number of iterations.', required=True)
parser.add_argument('--PHASE', type=str, required=True,
                    choices=['pretrain', 'finetune'],
                    help='pretrain = MIT-BIH @360; finetune = COPD @128 (upsampled to 360 in loader)')
parser.add_argument('--CKPT', type=str, default='',
                    help='Path to a generator checkpoint to load before training (for both pretrain and fine-tuning).')
parser.add_argument('--LR_G', type=float, default=1e-4, help='Generator LR (fine-tune default).')
parser.add_argument('--LR_D', type=float, default=2e-4, help='Discriminator LR (fine-tune default).')
parser.add_argument('--COPD_DIR', type=str, default='', help='Root under ForOsherPipeline')
parser.add_argument('--BEATS_N', type=int, default=2, help='Number of consecutive beats per window (>=1)')


TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                      30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                      160.0 * math.pi / 180.0]


def beat_quality_v3(x: np.ndarray, fs: int) -> float:
    """
    Robust quality for ~0.6 s beats:
      - MAD-normalize (scale-invariant)
      - zero-pad to >=2 s to get usable PSD resolution
      - QRS-band SNR (5–40 Hz) vs rest
      - Sharpness (peak-to-RMS)
      - Non-flatness (derivative activity)
    Returns [0,1].
    """
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < max(120, int(0.3*fs)):
        return float('nan')

    # clean + robust center/scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.median(x)
    mad = np.median(np.abs(x)) + 1e-9
    xz = x / (1.4826 * mad)

    # ---- PSD with zero-padding to at least 2 s ----
    targ = max(xz.size, int(2*fs))
    zp = np.pad(xz, (0, targ - xz.size)) if targ > xz.size else xz
    X = np.abs(np.fft.rfft(zp))**2
    f = np.fft.rfftfreq(zp.size, 1.0/fs)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return X[m].sum() + 1e-12
    qrs = bp(5, 40)
    rest = bp(0, 1) + bp(40, fs/2)
    s_snr = 1.0 - np.exp(-(qrs / rest))           # (0,1)

    # ---- sharpness: peak-to-RMS on the unpadded, z-scored beat ----
    rms = np.sqrt(np.mean(xz**2) + 1e-12)
    pk = np.max(np.abs(xz)) / (rms + 1e-12)
    s_sharp = float(np.clip((pk - 2.0)/(8.0 - 2.0), 0.0, 1.0))  # pk~2..8 -> 0..1

    # ---- non-flatness: derivative activity ----
    dx = np.diff(xz)
    thr = 0.02 * (np.median(np.abs(dx)) + 1e-9)
    act = 1.0 - np.mean(np.abs(dx) < thr)
    s_nonflat = float(np.clip(act, 0.0, 1.0))

    q = 0.5*s_snr + 0.3*s_sharp + 0.2*s_nonflat
    return float(np.clip(q, 0.0, 1.0))


def debug_one_beat(x: np.ndarray, fs: int, tag=""):
    x = np.asarray(x, float)
    print(f"[{tag}] len={x.size} fs={fs} min={x.min():.4g} max={x.max():.4g} std={x.std():.4g}")
    # guess fs if we accidentally pass the wrong one
    if 200 <= x.size <= 230 and fs != 360:
        print("  hint: len~216 -> this looks like 360 Hz beat; you're passing fs=", fs)
    if 70 <= x.size <= 90 and fs != 128:
        print("  hint: len~77 -> this looks like 128 Hz beat; you're passing fs=", fs)

    # show components used in v3
    x = np.nan_to_num(x - np.median(x))
    mad = np.median(np.abs(x)) + 1e-9
    xz = x / (1.4826*mad)
    targ = max(xz.size, int(2*fs))
    zp = np.pad(xz, (0, targ - xz.size))
    X = np.abs(np.fft.rfft(zp))**2
    f = np.fft.rfftfreq(zp.size, 1.0/fs)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return X[m].sum() + 1e-12
    qrs = bp(5, 40); rest = bp(0,1)+bp(40, fs/2)
    snr = qrs/rest
    rms = np.sqrt(np.mean(xz**2)+1e-12)
    pk = np.max(np.abs(xz))/(rms+1e-12)
    dx = np.diff(xz); thr = 0.02*(np.median(np.abs(dx))+1e-9)
    act = 1.0 - np.mean(np.abs(dx) < thr)

    print(f"  SNR(5-40/rest)={snr:.3f}  pk2rms={pk:.3f}  activity={act:.3f}")



def beat_quality_v2(x: np.ndarray, fs: int) -> float:
    """
    Robust single-beat ECG quality in [0,1]:
    - spectral QRS-band SNR (5–40 Hz)
    - sharpness (peak-to-RMS)
    - non-flatness (derivative activity)
    All with MAD-based normalization to be scale-invariant.
    """
    x = np.asarray(x, float)
    if x.size < max(120, int(0.3*fs)):   # need ~0.33–0.5 s minimum
        return float('nan')

    # center & robust-scale
    x = x - np.median(x)
    mad = np.median(np.abs(x)) + 1e-9
    xz = x / (1.4826*mad)                # ~z-score via MAD

    # 1) spectral SNR-like (QRS band vs rest)
    X = np.abs(np.fft.rfft(xz))**2
    f = np.fft.rfftfreq(xz.size, 1.0/fs)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return X[m].sum() + 1e-12
    qrs = bp(5, 40)
    rest = bp(0, 1) + bp(40, fs/2)
    snr = qrs/rest
    s_snr = 1.0 - np.exp(-snr)           # (0,1)

    # 2) sharpness (QRS prominence)
    rms = np.sqrt(np.mean(xz**2) + 1e-12)
    pk = np.max(np.abs(xz)) / (rms + 1e-12)
    # map pk ~ [2..8+] → [0..1]
    s_sharp = np.clip((pk - 2.0)/(8.0 - 2.0), 0.0, 1.0)

    # 3) non-flatness
    dx = np.diff(xz)
    thr = 0.02 * np.median(np.abs(dx)) + 1e-9
    act = 1.0 - np.mean(np.abs(dx) < thr)  # fraction of “moving” samples
    s_nonflat = float(np.clip(act, 0.0, 1.0))

    # combine (tune weights as we like)
    q = 0.5*s_snr + 0.3*s_sharp + 0.2*s_nonflat
    return float(np.clip(q, 0.0, 1.0))


def basic_ecg_quality(x: np.ndarray, fs: int) -> float:
    x = np.asarray(x, float)
    if x.size < int(0.4*fs):  # needs ~0.4 s minimum
        return float('nan')
    x = np.nan_to_num(x, nan=0.0)

    # 1) spectral SNR-like (5–40Hz vs. low+high)
    Pxx = np.abs(np.fft.rfft(x - np.median(x)))**2
    f = np.fft.rfftfreq(x.size, 1.0/fs)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return Pxx[m].sum() + 1e-12
    snr = band(5, 40) / (band(0, 1) + band(40, fs/2))
    s_snr = 1.0 - np.exp(-snr)  # (0,1)

    # 2) clipping / flatline penalties
    rng = np.percentile(x, 99) - np.percentile(x, 1) + 1e-12
    near_top = np.mean(x > (np.max(x) - 0.01*rng))
    near_bot = np.mean(x < (np.min(x) + 0.01*rng))
    flat = np.mean(np.abs(np.diff(x)) < 1e-6)
    penalty = np.clip(near_top + near_bot + flat, 0.0, 1.0)
    s_clip = 1.0 - penalty  # (0,1)

    # 3) regularity via autocorrelation peak at plausible HR (40–200 bpm)
    ac = correlate(x - x.mean(), x - x.mean(), mode="full")
    ac = ac[ac.size//2:]
    lmin = int(fs*0.3); lmax = int(fs*1.5)
    if lmax > lmin and lmax < ac.size:
        pk = np.max(ac[lmin:lmax]) / (ac[0] + 1e-12)
        s_reg = np.clip(pk, 0.0, 1.0)
    else:
        s_reg = 0.5

    return float(np.clip(0.5*s_snr + 0.3*s_clip + 0.2*s_reg, 0.0, 1.0))


# --- fast spectral SNR-like score usable on a single beat ---
def spectral_snr_quality(x: np.ndarray, fs: int) -> float:
    x = np.asarray(x, float)
    if x.size < 40 or np.allclose(x, x.mean()):
        return float('nan')
    x = x - np.median(x)
    Pxx = np.abs(np.fft.rfft(x))**2
    f = np.fft.rfftfreq(x.size, 1.0/fs)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return Pxx[m].sum() + 1e-12
    qrs = band(5, 40)
    low = band(0, 1)
    high = band(40, fs/2)
    ratio = qrs / (low + high)
    return float(np.clip(1.0 - np.exp(-ratio), 0.0, 1.0))

# --- NeuroKit (works if long enough); otherwise fallback to SNR ---
def nk_or_snr_quality(x: np.ndarray, fs: int) -> float:
    x = np.asarray(x, float)
    if nk is not None and x.size >= int(5*fs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sigs, _ = nk.ecg_process(x, sampling_rate=fs)
            qcol = next((c for c in sigs.columns if c.lower().startswith("ecg_quality")), None)
            if qcol is not None:
                q = float(np.nanmean(np.asarray(sigs[qcol]).ravel()))
                if not np.isnan(q):
                    return q
        except Exception:
            pass
    return spectral_snr_quality(x, fs)

# --- vital_sqi: combine simple SQIs into a [0,1] score, even for short beats ---
def vital_sqi_score(x: np.ndarray, fs: int) -> float:
    x = np.asarray(x, float)
    if x.size < int(0.5*fs):
        return float('nan')
    # (1) kurtosis (clip 1..10 to 0..1)
    try:
        k = float(kurtosis_sqi(x)) if kurtosis_sqi is not None else np.nan
    except Exception:
        k = np.nan
    ks = np.clip((k - 1.0) / (10.0 - 1.0), 0.0, 1.0) if not np.isnan(k) else np.nan
    # (2) spectral SNR-like
    ss = spectral_snr_quality(x, fs)
    vals = [v for v in (ks, ss) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')



def nk_or_snr_quality_old(x: np.ndarray, fs: int) -> float:
    """Mean NeuroKit ECG_Quality over x; fallback to spectral SNR in [0,1]."""
    x = np.asarray(x, dtype=float)
    if x.size < 40 or np.allclose(x, x[0]):  # too short/flat
        return float('nan')

    # Try NeuroKit (if available)
    if nk is not None:
        try:
            sigs, _ = nk.ecg_process(x, sampling_rate=fs)
            # column name can vary ("ECG_Quality" / "ECG_Quality_Raw"...)
            qcol = next((c for c in sigs.columns if c.lower().startswith("ecg_quality")), None)
            if qcol is not None:
                q = float(np.nanmean(np.asarray(sigs[qcol]).ravel()))
                if not np.isnan(q):
                    return q
        except Exception:
            pass

    # Fallback: power ratio in the QRS band (5–40 Hz) vs low+high bands
    x = x - np.median(x)
    if np.allclose(x, 0):  # still flat
        return float('nan')
    # Welch-lite via rFFT
    Pxx = np.abs(np.fft.rfft(x))**2
    f = np.fft.rfftfreq(x.size, 1.0/fs)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return Pxx[m].sum() + 1e-12
    qrs = band(5, 40)
    low = band(0, 1)
    high = band(40, fs/2)
    ratio = qrs / (low + high)
    q = 1.0 - np.exp(-ratio)              # map to (0,1)
    return float(np.clip(q, 0.0, 1.0))


def latest_ckpt(path_or_dir):
    if os.path.isdir(path_or_dir):
        files = glob.glob(os.path.join(path_or_dir, "checkpoint_epoch_*_iters_*"))
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {path_or_dir}")
        return max(files, key=os.path.getmtime)
    return path_or_dir

def load_generator_ckpt(netG, ckpt_path, device):
    p = latest_ckpt(ckpt_path)
    print(f"Loading generator checkpoint: {p}")
    ckpt = torch.load(p, map_location=device)
    netG.load_state_dict(ckpt['generator_state_dict'], strict=True)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_typical_N_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(TYPICAL_ODE_N_PARAMS).to(device)
    return params


def generate_typical_S_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_S_PARAMS).to(device)
    return params


def generate_typical_F_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_F_PARAMS).to(device)
    return params

def generate_typical_V_ode_params(b_size, device):
    noise_param = torch.Tensor(np.random.normal(0, 0.1, (b_size, 15))).to(device)
    params = 0.1 * noise_param + torch.Tensor(typical_beat_params.TYPICAL_ODE_V_PARAMS).to(device)
    return params


def ode_loss(hb_batch, ode_params, device, beat_type):
    """

    :param hb_batch:
    :return:
    """
    delta_t = ode_params.h
    delta_t = 0.01302 # 1/ 76.8

    batch_size = hb_batch.size()[0]
    if beat_type == "N":
        params_batch = generate_typical_N_ode_params(batch_size, device)
    elif beat_type == "S":
        params_batch = generate_typical_S_ode_params(batch_size, device)
    elif beat_type == 'F':
        params_batch = generate_typical_F_ode_params(batch_size, device)
    elif beat_type == 'V':
        params_batch = generate_typical_V_ode_params(batch_size, device)
    else:
        raise NotImplementedError()

    logging.debug("params batch shape: {}".format(params_batch.size()))
    #logging.info("delta_t: {}".format(delta_t))
    x_t = torch.tensor(-0.417750770388669).to(device)
    y_t = torch.tensor(-0.9085616622823985).to(device)
    t = torch.tensor(0.0).to(device)
    f_ode_z_signal = None
    delta_hb_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)

        f_ode_x = equations.d_x_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))

        y_t = y_t + delta_t * f_ode_y
        x_t = x_t + delta_t * f_ode_x
        t += 1 / 360

        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            delta_hb_signal = delta_hb
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))
    return delta_hb_signal, f_ode_z_signal


def euler_loss(hb_batch, params_batch, x_batch, y_batch, ode_params):
    """

    :param hb_batch: Nx216
    :param params_batch: Nx15
    :return:
    """
    logging.debug('hb batch shape: {}'.format(hb_batch.shape))
    logging.debug('params batch shape: {}'.format(params_batch.shape))
    logging.debug('x batch shape: {}'.format(x_batch.shape))
    logging.debug('y batch shape: {}'.format(y_batch.shape))

    delta_t = ode_params.h
    delta_t = 0.01302 # 1/ 76.8
    #logging.info("delta_t: {}".format(delta_t))

    t = torch.tensor(0.0)
    f_ode_z_signal = None
    f_ode_x_signal = None
    f_ode_y_signal = None
    delta_hb_signal = None
    delta_x_signal = None
    delta_y_signal = None
    for i in range(215):
        delta_hb = (hb_batch[:, i + 1] - hb_batch[:, i]) / delta_t
        delta_y = (y_batch[:, i + 1] - y_batch[:, i]) / delta_t
        delta_x = (x_batch[:, i + 1] - x_batch[:, i]) / delta_t
        delta_hb = delta_hb.view(-1, 1)
        delta_x = delta_x.view(-1, 1)
        delta_y = delta_y.view(-1, 1)
        logging.debug("Delta heart-beat shape: {}".format(delta_hb.shape))
        y_t = y_batch[:, i].view(-1, 1)
        x_t = x_batch[:, i].view(-1, 1)
        z_t = hb_batch[:, i].view(-1, 1)
        f_ode_x = equations.d_x_d_t(y_t, x_t, t,  ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)
        t += 1 / 512

        logging.debug("f ode z shape {}".format(f_ode_z.shape))  # Nx1
        logging.debug("f ode x shape {}".format(f_ode_x.shape))
        logging.debug("f ode y shape {}".format(f_ode_y.shape))
        if f_ode_z_signal is None:
            f_ode_z_signal = f_ode_z
            f_ode_x_signal = f_ode_x
            f_ode_y_signal = f_ode_y
            delta_hb_signal = delta_hb
            delta_x_signal = delta_x
            delta_y_signal = delta_y
        else:
            f_ode_z_signal = torch.cat((f_ode_z_signal, f_ode_z), 1)
            f_ode_x_signal = torch.cat((f_ode_x_signal, f_ode_x), 1)
            f_ode_y_signal = torch.cat((f_ode_y_signal, f_ode_y), 1)
            delta_hb_signal = torch.cat((delta_hb_signal, delta_hb), 1)
            delta_x_signal = torch.cat((delta_x_signal, delta_x), 1)
            delta_y_signal = torch.cat((delta_y_signal, delta_y), 1)


    logging.debug("f signal shape: {}".format(f_ode_z_signal.shape))
    logging.debug("delta hb signal shape: {}".format(delta_hb_signal.shape))

    return delta_hb_signal, f_ode_z_signal, f_ode_x_signal, f_ode_y_signal, delta_x_signal, delta_y_signal

# class ToTensorKeepMeta:
#     def __call__(self, sample):
#         import torch, numpy as np
#         hb  = sample['cardiac_cycle']
#         lab = sample['label']
#         if not torch.is_tensor(hb):  hb  = torch.from_numpy(hb).float()
#         if isinstance(lab, np.ndarray): lab = torch.from_numpy(lab)
#         else: lab = torch.as_tensor(lab)
#         sample['cardiac_cycle'] = hb
#         sample['label'] = lab
#         return sample

class ToTensorKeepMeta(object):
    def __call__(self, sample: dict):
        import numpy as np, torch
        hb = sample['cardiac_cycle']
        if isinstance(hb, np.ndarray):
            hb = torch.from_numpy(hb).float()
        elif not torch.is_tensor(hb):
            hb = torch.as_tensor(hb, dtype=torch.float32)
        sample['cardiac_cycle'] = hb
        # crucial: DO NOT touch other keys (src_path, r_index_128/360, label, ...)
        return sample


def wrap_path_for_title(p, base='/ForOsherPipeline/', width=34, extras=''):
    # show path relative to base (if present)
    rel = p.split(base, 1)[-1] if base in p else p
    # wrap to multiple lines
    rel_wrapped = textwrap.fill(rel, width=width)
    return rel_wrapped + (f"\n{extras}" if extras else "")


def train(batch_size, num_train_steps, model_dir, beat_type, generator_net, discriminator_net):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ode_params = ODEParams(device)

    # ---------------- paths / TB ----------------
    model_dir_abs = model_dir if os.path.isabs(model_dir) else os.path.join(current_dir, model_dir)
    os.makedirs(model_dir_abs, exist_ok=True)
    tb_dir = model_dir_abs
    print("[CKPT] model_dir_abs =", model_dir_abs)
    print("[CKPT] cwd           =", os.getcwd())
    print("[PATHS] TB_DIR      =", str(tb_dir))
    writer = SummaryWriter(logdir=str(tb_dir))
    writer.add_text("run/hello", "starting run", 0)
    writer.add_scalar("debug/boot", 1.0, 0)
    writer.flush()

    # ---------------- tiny helpers (self-contained) ----------------
    BEAT_LEN = 216

    def rand_crop_216(x):
        B, T = x.shape
        if T == BEAT_LEN: return x
        s = torch.randint(0, T - BEAT_LEN + 1, (B,), device=x.device)
        idx = torch.arange(BEAT_LEN, device=x.device)[None, :] + s[:, None]
        return x.gather(1, idx)

    def _sample_z_seq(B, z_dim, n_beats, device, rho=0.95):
        z0 = torch.randn(B, 1, z_dim, device=device)
        eps = torch.randn(B, n_beats, z_dim, device=device)
        z = torch.zeros_like(eps)
        z[:, 0] = z0[:, 0]
        sd = math.sqrt(max(1e-8, 1.0 - rho * rho))
        for t in range(1, n_beats):
            z[:, t] = rho * z[:, t - 1] + sd * eps[:, t]
        return z

    def assemble_window_from_beats(netG, z_seq, fade=32):
        B, n, _ = z_seq.shape
        beats = []
        for t in range(n):
            bt = netG(z_seq[:, t, :])          # (B,216) or (B,1,216)
            if bt.dim() == 3: bt = bt.squeeze(1)
            beats.append(bt)
        win = torch.zeros(B, n * BEAT_LEN, device=z_seq.device)
        if not fade or fade <= 0:
            for t in range(n):
                s = t * BEAT_LEN
                win[:, s:s + BEAT_LEN] += beats[t]
            return win
        w = torch.hann_window(2 * fade, device=z_seq.device)
        left, right = w[:fade], w[fade:]
        for t in range(n):
            s = t * BEAT_LEN
            seg = beats[t]
            if t > 0:
                win[:, s:s + fade] = win[:, s:s + fade] * right + seg[:, :fade] * left
                win[:, s + fade:s + BEAT_LEN] = seg[:, fade:]
            else:
                win[:, s:s + BEAT_LEN] = seg
        return win

    def seam_loss(win, n_beats, fade=32):
        if fade <= 0 or n_beats <= 1: return win.new_tensor(0.0)
        L, loss = BEAT_LEN, 0.0
        for k in range(1, n_beats):
            a, b = k * L - fade, k * L + fade
            seg = win[:, a:b]
            left, right = seg[:, :fade], seg[:, fade:]
            loss += ((left - right) ** 2).mean()
            dleft = left[:, 1:] - left[:, :-1]
            dright = right[:, 1:] - right[:, :-1]
            loss += ((dleft - dright) ** 2).mean()
        return loss / (n_beats - 1)

    def euler_loss_over_any(x_window):
        T = x_window.size(1)
        if T == BEAT_LEN:
            return ode_loss(x_window, ode_params, device, beat_type)
        assert T % BEAT_LEN == 0, f"window length {T} not multiple of {BEAT_LEN}"
        parts_d, parts_f = [], []
        n = T // BEAT_LEN
        for k in range(n):
            seg = x_window[:, k * BEAT_LEN:(k + 1) * BEAT_LEN]
            dhb, fz = ode_loss(seg, ode_params, device, beat_type)
            parts_d.append(dhb); parts_f.append(fz)
        return torch.cat(parts_d, 1), torch.cat(parts_f, 1)

    def make_real_windows_grouped(beats, src_paths, n_beats=1):
        B = beats.size(0)
        if B < n_beats: return None, None
        out, meta, i = [], [], 0
        while i + n_beats <= B:
            ok, p0 = True, src_paths[i]
            for j in range(1, n_beats):
                if src_paths[i + j] != p0: ok = False; break
            if ok:
                out.append(beats[i:i + n_beats].reshape(1, -1))
                meta.append((i, i + n_beats - 1))
                i += n_beats
            else:
                i += 1
        if not out: return None, None
        return torch.cat(out, 0).to(beats.device), meta

    # quality helpers (safe fallbacks)
    def nk_or_snr_quality(x, fs):
        try:
            import neurokit2 as nk
            sigs, _ = nk.ecg_process(x, sampling_rate=fs)
            q = sigs.get("ECG_Quality", None)
            qv = float(np.nanmean(q)) if q is not None else float('nan')
            if np.isnan(qv):
                # very short segments can produce NaN → crude SNR-like fallback
                x0 = x - np.mean(x)
                return float(10.0 * np.log10(np.mean(x0 * x0) + 1e-8))
            return qv
        except Exception:
            x0 = x - np.mean(x)
            return float(10.0 * np.log10(np.mean(x0 * x0) + 1e-8))

    def vital_sqi_score(x, fs):
        # if you have a real vital_sqi(), call it here; otherwise reuse NK/SNR
        return nk_or_snr_quality(x, fs)

    def beat_quality_v2(x, fs):
        # simple binary-ish proxy: strong R-peak presence
        return float(np.max(np.abs(x)) > 0.02)

    # ---------------- dataset ----------------
    if args.PHASE == 'finetune':
        composed = transforms.Compose([ecg_dataset_pytorch.Scale(), ToTensorKeepMeta()])
        dataset = EcgCOPDBeatsFromCSV(root_dir=args.COPD_DIR, beat_type=args.BEAT_TYPE,
                                      pre_ms=200, post_ms=400, use_detection_fallback=False,
                                      transform=composed)
    else:
        composed = transforms.Compose([ecg_dataset_pytorch.Scale(), ecg_dataset_pytorch.ToTensor()])
        positive_configs = dataset_configs.DatasetConfigs(
            'train', args.BEAT_TYPE, one_vs_all=True, lstm_setting=False,
            over_sample_minority_class=False, under_sample_majority_class=False,
            only_take_heartbeat_of_type=args.BEAT_TYPE, add_data_from_gan=False,
            gan_configs=None)
        dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs, transform=composed)

    shuffle_flag = False if int(getattr(args, 'BEATS_N', 1)) > 1 else True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle_flag, num_workers=1)
    print("Size of real dataset is {}".format(len(dataset)))

    # ---------------- models / opt ----------------
    netG = generator_net.to(device)
    netD = discriminator_net.to(device)
    netDw = WindowedDiscriminator(netD, crop_len=BEAT_LEN, stride=BEAT_LEN, agg="mean").to(device)
    D_beat, D_win = netD, netDw

    bce, mse = nn.BCELoss(), nn.MSELoss()
    beta1 = 0.5
    if args.PHASE == 'finetune':
        lr_g, lr_d = args.LR_G, args.LR_D
    else:
        lr_g = lr_d = 2e-4
    writer.add_scalar('Learning_Rate D', lr_d)
    writer.add_scalar('Learning_Rate G', lr_g)
    optimizer_d = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    # ---------------- loop ----------------
    epoch, iters = 0, 0
    global _full_cache
    if '_full_cache' not in globals():
        _full_cache = {}

    # <<< tiny helpers (local, no imports needed)
    def _rms(x, eps=1e-8):
        # x: (B, T) -> (B, 1)
        return torch.sqrt((x**2).mean(dim=1, keepdim=True).clamp_min(eps))

    def _rms_norm(x):
        # normalize per-window RMS (only for D inputs)
        return x / _rms(x)

    def _band_energy(x, fs, f_lo, f_hi):
        # x: (B, T)
        X = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(x.size(1), d=1.0/fs).to(x.device)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        # if the band is empty (can happen with very short T), fall back to the lowest non-DC bin
        if not mask.any():
            df = fs / x.size(1)
            mask = (freqs > 0) & (freqs <= 2*df)
        return (X[:, mask].abs()**2).mean(dim=1)  # (B,)


    while True:
        if iters == num_train_steps: break
        for i, data in enumerate(dataloader):
            if iters == num_train_steps: break

            beats_n = int(getattr(args, 'BEATS_N', 1))

            # real beats → windows (group by src_path if present)
            ecg_batch = data['cardiac_cycle'].float().to(device)
            real_beats = ecg_batch.squeeze(1) if ecg_batch.dim() == 3 else ecg_batch  # (B,216)
            assert real_beats.size(1) == BEAT_LEN

            if isinstance(data, dict) and ('src_path' in data):
                src_paths = data['src_path']
                real_win, meta = make_real_windows_grouped(real_beats, src_paths, n_beats=beats_n)
            else:
                B = (real_beats.size(0) // beats_n) * beats_n
                if B == 0: continue
                real_win = real_beats[:B].view(-1, beats_n * BEAT_LEN)
                meta = None
            if real_win is None or real_win.size(0) == 0: continue

            Bp = real_win.size(0)

            # ---------- D step ----------
            optimizer_d.zero_grad(set_to_none=True)

            # <<< D sees RMS-normalized windows/crops to remove scale cheating
            real_win_D  = _rms_norm(real_win)
            crop_real_D = _rms_norm(rand_crop_216(real_win))
            # >>> 

            logits_win_r  = D_win(real_win_D).view(-1)          # CHANGED: was real_win
            logits_beat_r = D_beat(crop_real_D).view(-1)        # CHANGED: was rand_crop_216(real_win)

            with torch.no_grad():
                z_seq = _sample_z_seq(Bp, z_dim=100, n_beats=beats_n, device=device, rho=0.9)
                #fake_win = assemble_window_from_beats(netG, z_seq, fade=32)
                fake_win = assemble_window_from_beats(netG, z_seq, fade=64)

            # <<<  normalize fakes too (for D only)
            fake_win_D  = _rms_norm(fake_win.detach())
            crop_fake_D = _rms_norm(rand_crop_216(fake_win.detach()))
            # >>> 

            logits_win_f  = D_win(fake_win_D).view(-1)          # CHANGED: was fake_win.detach()
            logits_beat_f = D_beat(crop_fake_D).view(-1)        # CHANGED: was rand_crop_216(fake_win.detach())

            ce_loss_d_real = bce(logits_win_r, torch.ones_like(logits_win_r))
            ce_loss_d_fake = bce(logits_win_f, torch.zeros_like(logits_win_f))
            loss_d = ce_loss_d_real + 0.5 * bce(logits_beat_r, torch.ones_like(logits_beat_r)) \
                   + ce_loss_d_fake + 0.5 * bce(logits_beat_f, torch.zeros_like(logits_beat_f))
            writer.add_scalar('Discriminator/cross_entropy_on_real_batch', ce_loss_d_real.item(), iters)
            writer.add_scalar('Discriminator/cross_entropy_on_fake_batch', ce_loss_d_fake.item(), iters)
            writer.add_scalar('Discriminator/total_loss', loss_d.item(), iters)
            loss_d.backward()
            optimizer_d.step()

            # ---------- G step ----------
            optimizer_g.zero_grad(set_to_none=True)

            z_seq = _sample_z_seq(Bp, z_dim=100, n_beats=beats_n, device=device, rho=0.9)
            fake_win = assemble_window_from_beats(netG, z_seq, fade=32)

            # <<< adversarial uses normalized inputs to D
            logits_win_g  = D_win(_rms_norm(fake_win)).view(-1)              # CHANGED
            logits_beat_g = D_beat(_rms_norm(rand_crop_216(fake_win))).view(-1)  # CHANGED
            # >>> 
            g_adv = bce(logits_win_g, torch.ones_like(logits_win_g)) \
                  + 0.5 * bce(logits_beat_g, torch.ones_like(logits_beat_g))

            dhb, fz = euler_loss_over_any(fake_win)
            mse_eul = mse(dhb, fz)
            # ---- schedules / weights ----
            # Stronger Euler early, then relax after ~6k iters
            lam_eul  = 1.0 if iters >= 6000 else 2.0
            lam_seam = 0.05 if beats_n == 2 else 0.03
            lam_diff = 0.05                           # was 0.10/0.05
            lam_hf   = 0.02                           # was 0.015
            lam_amp  = 0.25                           # was 0.20
            lam_lf   = 0.02                           # was 0.03 (baseline drift a bit over-penalized)
            lam_mid  = 0.03                           # was 0.01 (sharper QRS & T)

            # lam_eul  = 2.0 if iters < 6000 else 1.0
            # # Small, steady seam weight; use wider fade in your assembler (fade=48..64)
            # lam_seam = 0.03
            # # Smoothness (reduce later)
            # lam_diff = 0.10 if iters < 3000 else 0.05
            # # Tiny HF (>45 Hz) penalty
            # lam_hf   = 0.015
            # # Mean/RMS matching (helps scale & offset)
            # lam_amp  = 0.20
            # # <<< baseline drift & mid-band (QRS support)
            # lam_lf   = 0.03   # 0.05–0.40 Hz (will auto-fallback if too short)
            # lam_mid  = 0.01   # 5–20 Hz
            # >>> 

            # ---- components ----
            # Euler (yours)
            dhb, fz = euler_loss_over_any(fake_win)
            mse_eul = mse(dhb, fz)

            # Smoothness (1st derivative)
            diff_pen = (fake_win[:, 1:] - fake_win[:, :-1]).abs().mean()

            # High-frequency power above 45 Hz @ 360 Hz
            def highband_power(x, fs=360, f_hi=45):
                X = torch.fft.rfft(x, dim=1)
                freqs = torch.fft.rfftfreq(x.size(1), d=1.0/fs).to(x.device)
                mask = (freqs > f_hi)
                if not mask.any():
                    return torch.tensor(0.0, device=x.device)
                return (X[:, mask].abs()**2).mean()
            hf_pen = highband_power(fake_win)

            # Amplitude: match mean and std to the current real windows in the batch
            real_win_batch = real_win[:fake_win.size(0)]
            mu_f, sd_f = fake_win.mean(1), fake_win.std(1)
            mu_r, sd_r = real_win_batch.mean(1), real_win_batch.std(1)
            amp_pen = (mu_f - mu_r).abs().mean() + (sd_f - sd_r).abs().mean()

            # <<< Baseline drift (very low band) & QRS mid-band support
            lf_real = _band_energy(real_win_batch, fs=360, f_lo=0.05, f_hi=0.40)
            lf_fake = _band_energy(fake_win,        fs=360, f_lo=0.05, f_hi=0.40)
            lf_loss = ((lf_fake - lf_real)**2).mean()

            mb_real = _band_energy(real_win_batch, fs=360, f_lo=5.0,  f_hi=20.0)
            mb_fake = _band_energy(fake_win,        fs=360, f_lo=5.0,  f_hi=20.0)
            mid_loss = ((mb_fake - mb_real)**2).mean()
            # >>> 

            # ---- total ----
            ce_loss_g_fake = g_adv  # keep your name
            total_g_loss = (
                ce_loss_g_fake
                + lam_eul  * mse_eul
                + lam_seam * seam_loss(fake_win, beats_n, fade=48)  # <-- use fade=48..64 in assembler
                + lam_diff * diff_pen
                + lam_hf   * hf_pen
                + lam_amp  * amp_pen
                + lam_lf   * lf_loss       
                + lam_mid  * mid_loss       
            )
            total_g_loss.backward()
            optimizer_g.step()

            # logs
            if iters % 100 == 0:
                with torch.no_grad():
                    dr, fr = euler_loss_over_any(real_win)
                    mse_real = F.mse_loss(dr, fr).item()
                print(f"[{iters}] real(min/max) {real_win.min():.5f}/{real_win.max():.5f} "
                      f"; fake(min/max) {fake_win.min():.5f}/{fake_win.max():.5f}")
                print(f"[{iters}] Euler MSE  real={mse_real:.4f}  fake={mse_eul.item():.4f}")
            writer.add_scalar('Generator/mse_ode', mse_eul.item(), iters)
            writer.add_scalar('Generator/cross_entropy_on_fake_batch', ce_loss_g_fake.item(), iters)
            # a couple of helpful scalars
            writer.add_scalar('Generator/lf_loss',  lf_loss.item(),  iters)
            writer.add_scalar('Generator/mid_loss', mid_loss.item(), iters)
            writer.add_scalar('Generator/amp_pen',  amp_pen.item(),  iters)
            writer.add_scalar('Generator/hf_pen',   hf_pen.item(),   iters)
            writer.add_scalar('Generator/diff_pen', diff_pen.item(), iters)

            # ---------- Validation plot with TITLES + FULL-SIGNAL QUALITY ----------
            if iters % 25 == 0:
                with torch.no_grad():
                    netG.eval()
                    n = min(16, real_win.size(0))
                    z_seq_val = _sample_z_seq(n, z_dim=100, n_beats=beats_n, device=device, rho=0.9)
                    #fakes_val = assemble_window_from_beats(netG, z_seq_val, fade=32)
                    fakes_val = assemble_window_from_beats(netG, z_seq_val, fade=64)
                    real_show = real_win[:n]
                    netG.train()

                    # metadata (for titles) — use first beat of each window if available
                    if isinstance(data, dict) and ('src_path' in data):
                        win_src = []
                        if meta is not None:
                            for (s_idx, e_idx) in meta[:n]:
                                try:   win_src.append(src_paths[s_idx])
                                except: win_src.append("?")
                        else:
                            win_src = ['?'] * n
                    else:
                        win_src = ['?'] * n

                    rows, cols = 4, 4
                    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
                    axes = axes.ravel()

                    def wrap_path_for_title(p, extras="", width=28):
                        base = str(p).split('/ForOsherPipeline/')[-1] if isinstance(p, str) else '?'
                        if len(base) <= width: txt = base
                        else: txt = '\n'.join([base[i:i + width] for i in range(0, len(base), width)])
                        return (txt + ("" if not extras else f"\n{extras}"))

                    FS_BEAT = 360
                    USE_FULLSIGNAL_FOR_REAL_Q = True
                    QUALITY_WIN_SEC = 8.0

                    for j in range(n):
                        fake = fakes_val[j].detach().cpu().numpy()
                        real = real_show[j].detach().cpu().numpy()

                        # fake qualities @360
                        try: q_nk_fake = nk_or_snr_quality(fake, fs=FS_BEAT)
                        except: q_nk_fake = float('nan')
                        try: q_vs_fake = vital_sqi_score(fake, fs=FS_BEAT)
                        except: q_vs_fake = float('nan')
                        try: q_basic_fake = beat_quality_v2(fake, fs=FS_BEAT)
                        except: q_basic_fake = float('nan')

                        # real qualities (prefer 128Hz full-signal slice when available)
                        q_nk_real = q_vs_real = q_basic_real = float('nan')
                        if USE_FULLSIGNAL_FOR_REAL_Q and isinstance(data, dict) and ('src_path' in data):
                            try:
                                path_i = win_src[j]
                                if path_i not in _full_cache:
                                    from sim_gan.data_reader.ecg_dataset_copd_from_csv import _read_copd_csv
                                    _full_cache[path_i] = _read_copd_csv(path_i)
                                rec = _full_cache[path_i]
                                ecg128 = rec['ecg_128'] / float(rec['magnification'])
                                fs128 = 128
                                n128 = ecg128.size
                                half = int((QUALITY_WIN_SEC / 2.0) * fs128)
                                mid = n128 // 2
                                s = max(0, mid - half); e = min(n128, mid + half)
                                seg128 = ecg128[s:e]
                                q_nk_real = nk_or_snr_quality(seg128, fs=fs128)
                                q_vs_real = vital_sqi_score(seg128, fs=fs128)
                                q_basic_real = beat_quality_v2(seg128, fs=fs128)
                            except Exception:
                                # fallback on the shown 360Hz window
                                try: q_nk_real = nk_or_snr_quality(real, fs=FS_BEAT)
                                except: pass
                                try: q_vs_real = vital_sqi_score(real, fs=FS_BEAT)
                                except: pass
                                try: q_basic_real = beat_quality_v2(real, fs=FS_BEAT)
                                except: pass
                        else:
                            try: q_nk_real = nk_or_snr_quality(real, fs=FS_BEAT)
                            except: pass
                            try: q_vs_real = vital_sqi_score(real, fs=FS_BEAT)
                            except: pass
                            try: q_basic_real = beat_quality_v2(real, fs=FS_BEAT)
                            except: pass

                        axes[j].plot(fake, label=f'fake ({len(fake)})')
                        axes[j].plot(real, label=f'real ({len(real)})')
                        axes[j].set_title(wrap_path_for_title(win_src[j], width=28), fontsize=8, loc='left')

                        def fmt(v, nd=2):
                            return "nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"
                        axes[j].text(0.02, 0.98,
                                     "NK  : F={}  R={}\nVS  : F={}  R={}\nBasic: F={}  R={}".format(
                                         fmt(q_nk_fake), fmt(q_nk_real),
                                         fmt(q_vs_fake), fmt(q_vs_real),
                                         fmt(q_basic_fake), fmt(q_basic_real)),
                                     transform=axes[j].transAxes, ha='left', va='top', fontsize=7,
                                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
                        if j % cols == 0:
                            axes[j].legend(fontsize=7, loc='upper right')

                    for j in range(n, rows * cols): axes[j].axis('off')
                    fig.suptitle(f"Fake vs Real windows (N={beats_n}) — iter {iters}", fontsize=12)
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    writer.add_figure('Generator/window_output_example', fig, iters)
                    plt.close(fig)

            # ---------- checkpoints ----------
            #if iters == 0 or iters % getattr(args, "SAVE_EVERY", 200) == 0:
            if iters % 50 == 0:
                path = os.path.join(model_dir_abs, f'checkpoint_epoch_{epoch}_iters_{iters}')
                tmp = path + ".tmp"
                torch.save({'epoch': epoch, 'generator_state_dict': netG.state_dict()}, tmp)
                os.replace(tmp, path)
                print(f"[CKPT] saved -> {path}")
                writer.flush()

            iters += 1
        epoch += 1

    final_path = os.path.join(model_dir_abs, f'checkpoint_epoch_{epoch}_iters_{iters}')
    torch.save({'epoch': epoch, 'generator_state_dict': netG.state_dict()}, final_path)
    writer.flush()
    writer.close()

def get_gradient_norm_l2(model):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'ode_params_generator' not in name:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    model_dir = args.MODEL_DIR
    gan_type = args.GAN_TYPE
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if gan_type == 'SimVGAN':
        netG = vanila_gan.VGenerator(0)
        netD = vanila_gan.VDiscriminator(0)
    elif gan_type == 'SimDCGAN':
        netG = sim_gan_euler.DCGenerator(0)
        netD = sim_gan_euler.DCDiscriminator(0)
    else:
        raise ValueError(f"Invalid gan type {gan_type}.")

    # init weights only if we are NOT loading a checkpoint
    if not args.CKPT:
        netD.apply(weights_init)
        netG.apply(weights_init)
    else:
        load_generator_ckpt(netG, args.CKPT, device)


    train(args.BATCH_SIZE, args.NUM_ITERATIONS, model_dir, beat_type=args.BEAT_TYPE, generator_net=netG,
          discriminator_net=netD)
