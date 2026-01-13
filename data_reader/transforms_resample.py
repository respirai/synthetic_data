# transforms_resample.py
import numpy as np
from scipy.signal import resample_poly

class ResampleTo360:
    """Upsample a 128 Hz beat to 360 Hz and crop/pad to 216 samples."""
    def __call__(self, sample):
        x = sample['cardiac_cycle']          # 1D numpy or torch; convert to numpy if needed
        x = np.asarray(x, dtype=np.float32)

        # 128 -> 360 Hz with polyphase filter (anti-imaging/aliasing handled)
        x360 = resample_poly(x, up=45, down=16)

        # Make sure we end up with exactly 216 samples (≈600 ms @ 360 Hz)
        if x360.shape[0] > 216:
            # center-crop (or align to R-peak if you have the index)
            start = (x360.shape[0] - 216) // 2
            x360 = x360[start:start+216]
        elif x360.shape[0] < 216:
            pad = 216 - x360.shape[0]
            x360 = np.pad(x360, (pad//2, pad - pad//2), mode='edge')

        sample['cardiac_cycle'] = x360.astype(np.float32)
        return sample
