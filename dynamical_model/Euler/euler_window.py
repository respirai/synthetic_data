import torch
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# TODO: Remove
#sys.path.insert(0, current_dir)
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, dir_path)
from sim_gan.dynamical_model import equations
from sim_gan.dynamical_model.ode_params import ODEParams
from sim_gan.dynamical_model import typical_beat_params

def _typical_params(beat_type, batch, device):
    noise = torch.randn(batch, 15, device=device) * 0.1
    if beat_type == "N":
        base = torch.tensor([0.7, 0.25, -0.5*math.pi, -7.0, 0.1, -15.0*math.pi/180.0,
                             30.0, 0.1, 0.0, -3.0, 0.1, 15.0*math.pi/180.0, 0.2, 0.4,
                             160.0*math.pi/180.0], device=device)
    elif beat_type == "S":
        base = torch.tensor(typical_beat_params.TYPICAL_ODE_S_PARAMS, device=device)
    elif beat_type == "V":
        base = torch.tensor(typical_beat_params.TYPICAL_ODE_V_PARAMS, device=device)
    elif beat_type == "F":
        base = torch.tensor(typical_beat_params.TYPICAL_ODE_F_PARAMS, device=device)
    else:
        raise NotImplementedError
    return base[None, :].repeat(batch, 1) + 0.1 * noise

def euler_loss_window(ecg_window, ode_params: ODEParams, device, beat_type="N"):
    """
    ecg_window: (B, T) at 360 Hz (T = 216 * N).
    Returns: (delta_hb_signal, f_ode_z_signal) both (B, T-1).
    """
    delta_t = ode_params.h            # 1/360 by default
    B, T = ecg_window.shape

    params_batch = _typical_params(beat_type, B, device)

    # initial state (from the paper / original code)
    x_t = torch.tensor(-0.417750770388669, device=device)
    y_t = torch.tensor(-0.9085616622823985, device=device)
    t   = torch.tensor(0.0, device=device)

    f_ode_z_signal = []
    delta_hb_signal = []

    for i in range(T - 1):
        z_t = ecg_window[:, i].view(-1, 1)
        delta = (ecg_window[:, i+1] - ecg_window[:, i]) / delta_t
        delta_hb_signal.append(delta.view(-1, 1))

        f_ode_x = equations.d_x_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_y = equations.d_y_d_t(y_t, x_t, t, ode_params.rrpc, ode_params.h)
        f_ode_z = equations.d_z_d_t(x_t, y_t, z_t, t, params_batch, ode_params)
        f_ode_z_signal.append(f_ode_z)

        y_t = y_t + delta_t * f_ode_y
        x_t = x_t + delta_t * f_ode_x
        t   = t + 1.0 / 360.0

    f_ode_z_signal = torch.cat(f_ode_z_signal, dim=1)
    delta_hb_signal= torch.cat(delta_hb_signal, dim=1)
    return delta_hb_signal, f_ode_z_signal
