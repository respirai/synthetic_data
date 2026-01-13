import torch
import torch.nn.functional as F

@torch.no_grad()
def _sample_z_seq(b, z_dim, n_beats, device, rho=0.9):
    """AR(1) latent sequence per sample for morphology continuity."""
    z0 = torch.randn(b, z_dim, device=device)
    zs = [z0]
    for _ in range(1, n_beats):
        eps = torch.randn(b, z_dim, device=device)
        zs.append(rho * zs[-1] + (1.0 - rho**2)**0.5 * eps)
    return torch.stack(zs, dim=1)  # (B, n_beats, z_dim)

def assemble_window_from_beats(netG, z_seq, fade=16):
    """
    netG: beat generator   z_seq: (B, n_beats, Z)
    Returns: (B, 216*n_beats) continuous window via overlap-add crossfades.
    """
    B, K, Z = z_seq.shape
    beats = []
    for k in range(K):
        y = netG(z_seq[:, k, :])           # (B, 216)
        y = torch.tanh(y)                  # keep amplitude bounded
        beats.append(y)
    beat = torch.stack(beats, dim=1)       # (B, K, 216)

    if fade <= 0:
        return beat.reshape(B, -1)

    # cosine crossfade between consecutive beats over `fade` samples
    fade = min(fade, 40)  # safety
    w = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, fade, device=beat.device))
    w_in  = w.view(1,1,-1)                 # rising  (0->1)
    w_out = (1 - w).view(1,1,-1)           # falling (1->0)

    T = K * 216
    out = torch.zeros(B, T, device=beat.device)
    for b in range(B):
        t = 0
        for k in range(K):
            seg = beat[b, k]               # (216,)
            if k == 0:
                out[b, t:t+216] = seg
            else:
                # overlap last fade samples
                t0 = t - fade
                out[b, t0:t0+fade] = out[b, t0:t0+fade] * w_out + seg[:fade] * w_in
                out[b, t:t+216-fade] = seg[fade:]
            t += 216
    return out


def make_real_windows_grouped(ecg_batch, src_paths, n_beats: int = 1):
    """
    Group consecutive beats from the same source file into windows of length 216*n_beats.
    ecg_batch: (B,216) or (B,1,216)
    src_paths: list/array of length B with file IDs
    Returns: (X, M) where X: (B', 216*n_beats) and M: metadata indices used.
    """
    x = ecg_batch.squeeze(1) if ecg_batch.dim()==3 else ecg_batch  # (B,216)
    B = x.size(0)
    out, meta = [], []
    i = 0
    while i + n_beats <= B:
        same = all(src_paths[j] == src_paths[i] for j in range(i, i+n_beats))
        if same:
            out.append(x[i:i+n_beats, :].reshape(1, -1))  # (1, 216*n)
            meta.append((i, i+n_beats))
            i += n_beats
        else:
            i += 1  # shift window by one beat until a same-file block appears
    if not out:
        return None, None
    return torch.cat(out, dim=0), meta
