import os, glob
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import resample_poly
import wfdb
from wfdb.processing import xqrs_detect  # QRS detector (works at any fs)

class EcgCOPDBeatsFromRecords(Dataset):
    """
    Turns long COPD ECG records (128 Hz) into beat-centered 216-sample windows at 360 Hz.
    Directory can contain .npy or .csv files with a single ECG channel.
    If you have per-record scale (e.g., 'magnification'), apply it before returning.
    """
    def __init__(self, root_dir, pattern="*.npy", beat_type="N", transform=None,
                 pre_ms=200, post_ms=400, fs_in=128, fs_out=360):
        self.root_dir = root_dir
        self.paths = sorted([p for ext in pattern.split(";")
                                for p in glob.glob(os.path.join(root_dir, ext))])
        if not self.paths:
            # also try CSV by default
            self.paths = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        if not self.paths:
            raise FileNotFoundError(f"No ECG files under {root_dir}")

        self.transform = transform
        self.pre = int(round(pre_ms * fs_out / 1000.0))   # 72 at 360 Hz
        self.post = int(round(post_ms * fs_out / 1000.0)) # 144 at 360 Hz
        self.win_len = self.pre + self.post               # 216
        self.fs_in, self.fs_out = fs_in, fs_out
        self.ratio_up, self.ratio_down = 45, 16           # 128→360 is 45/16
        self.items = []  # list of (file_index, r_index_360)

        # Pre-index beats for all records
        for fi, p in enumerate(self.paths):
            x = self._load_1d(p)  # 128 Hz long record
            # R-peak detection at native rate:
            r_idx_128 = xqrs_detect(sig=x, fs=self.fs_in, verbose=False)
            if r_idx_128 is None or len(r_idx_128) == 0:
                continue
            # Upsample whole record to 360 Hz once:
            x360 = resample_poly(x, self.ratio_up, self.ratio_down)
            # Map R indices from 128→360
            r_idx_360 = (r_idx_128.astype(np.float64) * self.ratio_up / self.ratio_down).round().astype(int)
            # Keep beats fully inside bounds
            for r in r_idx_360:
                s, e = r - self.pre, r + self.post
                if s >= 0 and e <= len(x360):
                    self.items.append((fi, r))
        # cache for loaded & upsampled records
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fi, r = self.items[idx]
        path = self.paths[fi]
        # cached upsampled record
        if fi not in self._cache:
            x = self._load_1d(path)
            x360 = resample_poly(x, self.ratio_up, self.ratio_down).astype(np.float32)
            self._cache[fi] = x360
        x360 = self._cache[fi]
        s, e = r - self.pre, r + self.post
        beat = x360[s:e].copy()  # (216,)

        sample = {'cardiac_cycle': beat, 'beat_type': 'N'}  # if you don’t have labels, default to 'N'
        if self.transform:
            sample = self.transform(sample)
        # ensure torch.float32 tensor later (existing ToTensor does it)
        return sample

    @staticmethod
    def _load_1d(path):
        if path.lower().endswith(".npy"):
            arr = np.load(path).squeeze()
        else:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32).squeeze()
        return np.asarray(arr, dtype=np.float32)
