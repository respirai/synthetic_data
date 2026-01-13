import os, glob, json, io, csv, re
import numpy as np
from torch.utils.data import Dataset

# high-quality resampling; falls back to PyTorch linear if SciPy not present
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    import torch
    import torch.nn.functional as F

def _resample_128_to_360(x):
    if _HAS_SCIPY:
        return resample_poly(x, up=45, down=16).astype(np.float32)
    # fallback (linear interp)
    import torch, torch.nn.functional as F
    xt = torch.from_numpy(x.astype(np.float32)).view(1,1,-1)
    L_out = int(round(len(x) * 360.0 / 128.0))
    return F.interpolate(xt, size=L_out, mode='linear', align_corners=False).squeeze().numpy()

_NORM = re.compile(r'[^a-z0-9]+')
def _norm(s: str) -> str:
    return _NORM.sub('', s.strip().lower())

def _sniff_delimiter(path):
    with open(path, 'r', newline='') as f:
        head = f.read(4096)
    try:
        return csv.Sniffer().sniff(head).delimiter
    except Exception:
        return ',' if head.count(',') >= head.count(';') else ';'

def _parse_int_array(cell):
    if cell is None: return np.array([], dtype=int)
    s = str(cell).strip()
    if not s or s.lower() in ('nan','none'): return np.array([], dtype=int)
    try:
        if s.startswith('[') and s.endswith(']'):
            return np.asarray(json.loads(s), dtype=int)
    except Exception:
        pass
    sep = ';' if ';' in s else (',' if ',' in s else ' ')
    toks = [t for t in s.replace('[','').replace(']','').split(sep) if t.strip()]
    out = []
    for t in toks:
        try: out.append(int(float(t)))
        except: pass
    return np.asarray(out, dtype=int)

def _parse_float_array(cell):
    if cell is None: return np.array([], dtype=np.float32)
    s = str(cell).strip()
    if not s or s.lower() in ('nan','none'): return np.array([], dtype=np.float32)
    try:
        if s.startswith('[') and s.endswith(']'):
            arr = json.loads(s)
            return np.asarray(arr, dtype=np.float32)
    except Exception:
        pass
    sep = ';' if ';' in s else (',' if ',' in s else ' ')
    toks = [t for t in s.replace('[','').replace(']','').split(sep) if t.strip()]
    out = []
    for t in toks:
        try: out.append(float(t))
        except: pass
    return np.asarray(out, dtype=np.float32)

PREFERRED_ECG_NAMES = [
    'ecgdenoised','ecgfiltered','ecgclean','ecgcleaned',
    'ecgproc','ecgprocessed','ecg','ecgdata'
]

def _read_copd_csv(path):
    """
    Robust reader for ECGRec*ECG_denoised.csv.
    Supports:
      - ECG as per-row numeric values, or a single array cell.
      - RWL/RRI as array cell(s).
      - Magnification per-row or single cell.
    Returns: dict(ecg_128: float32[N], magnification: float, rwl_128: int[M])
    """
    # Skip zero-byte files fast
    try:
        if os.path.getsize(path) == 0:
            raise ValueError(f"Empty file: {path}")
    except OSError:
        raise ValueError(f"Unreadable file (missing permissions?): {path}")
    
    delim = _sniff_delimiter(path)
    with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
        r = csv.reader(f, delimiter=delim)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError(f"Empty file: {path}")
        raw_to_norm = {h: _norm(h) for h in header}
        # ECG column index (prefer denoised/filtered)
        ecg_idx = None
        for pref in PREFERRED_ECG_NAMES:
            for j,h in enumerate(header):
                if raw_to_norm[h] == pref:
                    ecg_idx = j; break
            if ecg_idx is not None: break
        if ecg_idx is None:
            for j,h in enumerate(header):
                nh = raw_to_norm[h]
                if ('ecg' in nh) and all(k not in nh for k in ('rwl','rri','rrinterval','magnification','mag')):
                    ecg_idx = j; break

        mag_idx = None; rwl_idx = None; rri_idx = None
        for j,h in enumerate(header):
            nh = raw_to_norm[h]
            if mag_idx is None and (nh == 'magnification' or nh.endswith('magnification') or nh == 'mag'):
                mag_idx = j
            if rwl_idx is None and (nh == 'rwl' or 'rwavelocation' in nh or 'rpeaks' in nh):
                rwl_idx = j
            if rri_idx is None and (nh == 'rri' or 'rwaveinterval' in nh or nh == 'rr' or 'rrinterval' in nh):
                rri_idx = j

        rows = list(r)  # we’ll scan a bit; fine for training
        if ecg_idx is None:
            # heuristic fallback: most numeric column
            best = (-1, None)
            for j in range(len(header)):
                if j in (rwl_idx, rri_idx): continue
                cnt = 0
                for row in rows[:2000]:
                    if j < len(row):
                        try:
                            float(row[j]); cnt += 1
                        except: pass
                if cnt > best[0]:
                    best = (cnt, j)
            ecg_idx = best[1]

        if ecg_idx is None:
            raise ValueError(f"Could not identify ECG column in {path}. Headers: {header}")

    # -------- extract ECG ----------
    # Case A: per-row numeric values
    ecg_vals = []
    for row in rows:
        if ecg_idx < len(row) and row[ecg_idx] != '':
            try: ecg_vals.append(float(row[ecg_idx]))
            except: pass

    if len(ecg_vals) >= 10:
        ecg_128 = np.asarray(ecg_vals, dtype=np.float32)
    else:
        # Case B: ECG stored as a single array cell (common in device exports)
        first_nonempty = None
        for row in rows:
            if ecg_idx < len(row) and row[ecg_idx] not in (None,'','NaN'):
                first_nonempty = row[ecg_idx]; break
        arr = _parse_float_array(first_nonempty)
        if arr.size == 0:
            raise ValueError(
                f"ECG column empty or unreadable in {path} (delim='{delim}'). "
                f"Heads={header[:6]}..."
            )
        ecg_128 = arr.astype(np.float32)

    # -------- magnification ----------
    mags = []
    if mag_idx is not None:
        for row in rows:
            if mag_idx < len(row) and row[mag_idx] != '':
                try: mags.append(float(row[mag_idx]))
                except: pass
    if len(mags) > 0:
        magnification = float(np.median(np.asarray(mags, dtype=np.float32)))
    else:
        # try single metadata cell in first non-empty row
        magnification = 1000.0
        if mag_idx is not None and rows and mag_idx < len(rows[0]):
            try:
                mv = float(rows[0][mag_idx]); 
                if not np.isnan(mv): magnification = mv
            except: pass

    # -------- RWL / RRI ----------
    rwl_128 = np.array([], dtype=int)
    if rwl_idx is not None:
        for row in rows:
            if rwl_idx < len(row) and row[rwl_idx] not in (None,'','NaN'):
                rwl_128 = _parse_int_array(row[rwl_idx])
                if rwl_128.size > 0: break
    if rwl_128.size == 0 and rri_idx is not None:
        # derive RWL from RRI
        first = None
        for row in rows:
            if rri_idx < len(row) and row[rri_idx] not in (None,'','NaN'):
                rri = _parse_int_array(row[rri_idx])
                if rri.size > 0:
                    cs = np.cumsum(rri, dtype=int)
                    # anchor at the first interval if needed
                    rwl_128 = cs[(cs >= 0) & (cs < len(ecg_128))]
                    break

    return dict(ecg_128=ecg_128, magnification=magnification, rwl_128=rwl_128)


class EcgCOPDBeatsFromCSV(Dataset):
    """
    Crawls: <root>/<PATIENT_ID>/patch_files/ECGRec*ECG_denoised.csv
    Uses RWL (R-wave locations at 128 Hz) if present; otherwise can optionally
    fall back to detection (not enabled by default).
    Produces beat-centered 216-sample windows at 360 Hz, scaled to mV.
    """
    def __init__(self, root_dir, beat_type='N',
                 pre_ms=200, post_ms=400, use_detection_fallback=False,
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.beat_type = beat_type
        self.pre_360 = int(round(pre_ms  * 360 / 1000.0))   # 72
        self.post_360= int(round(post_ms * 360 / 1000.0))   # 144
        self.win_len = self.pre_360 + self.post_360         # 216
        # Find files: */patch_files/ECGRec*ECG_denoised.csv
        pattern = os.path.join(root_dir, "*", "patch_files", "ECGRec*ECG_denoised.csv")
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            # recursive fallback (if extra nesting exists)
            pattern = os.path.join(root_dir, "**", "patch_files", "ECGRec*ECG_denoised.csv")
            self.paths = sorted(glob.glob(pattern, recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No ECGRec*ECG_denoised.csv under {root_dir}")

        # Build a global index of (file_index, r_idx_360) for all valid beats
        self.items = []
        self._cache = {}  # file_index -> dict(x360, r_idx_360)
        self.use_detection_fallback = use_detection_fallback
        
        print(f"[COPD] Found {len(self.paths)} files. Indexing beats…")
        bad = 0
        for fi, p in enumerate(self.paths):
            try:
                rec = _read_copd_csv(p)
            except Exception as e:
                print(f"[COPD] Skipping file: {p}  ({e})")
                bad += 1
                continue
            # scale to mV
            x_mV_128 = rec['ecg_128'] / float(rec['magnification'])
            # upsample once to 360 Hz
            x_mV_360 = _resample_128_to_360(x_mV_128)
            # map RWL if exists
            if rec['rwl_128'].size == 0:
                print(f"[INFO] No RWL array in {p}. "
                    f"Set use_detection_fallback=True to auto-detect R-peaks.")
            if rec['rwl_128'].size > 0:
                r_idx_360 = np.round(rec['rwl_128'].astype(np.float64) * (45.0/16.0)).astype(np.int32)
            else:
                if not use_detection_fallback:
                    # skip this file if no RWL and fallback disabled
                    continue
                # Optional fallback: detect R-peaks at 360 Hz
                try:
                    import wfdb
                    from wfdb.processing import xqrs_detect
                    r_idx_360 = xqrs_detect(sig=x_mV_360, fs=360, verbose=False)
                except Exception:
                    r_idx_360 = np.array([], dtype=int)

            # keep only beats fully inside [0, len)
            keep_pairs = []  # list of (r360, r128)
            if rec['rwl_128'].size > 0:
                r_idx_128 = rec['rwl_128'].astype(np.int32)
                mapped_360 = np.round(r_idx_128.astype(np.float64) * (45.0/16.0)).astype(np.int32)
                for r128, r360 in zip(r_idx_128, mapped_360):
                    s = r360 - self.pre_360
                    e = r360 + self.post_360
                    if s >= 0 and e <= len(x_mV_360):
                        keep_pairs.append((int(r360), int(r128)))
            else:
                # detection fallback produced only r_idx_360; derive approx r128 by inverse mapping
                for r360 in np.asarray(r_idx_360, dtype=np.int32):
                    s = int(r360) - self.pre_360
                    e = int(r360) + self.post_360
                    if s >= 0 and e <= len(x_mV_360):
                        r128 = int(round(r360 * (16.0/45.0)))
                        r128 = int(np.clip(r128, 0, len(x_mV_128)-1))
                        keep_pairs.append((int(r360), r128))

            if not keep_pairs:
                continue

            self._cache[fi] = dict(
                x360=x_mV_360.astype(np.float32),
                r_idx_360=np.asarray([p[0] for p in keep_pairs], dtype=np.int32),
            )
            # add to global index: (file_index, r360, r128)
            for (r360, r128) in keep_pairs:
                self.items.append((fi, r360, r128))
        print(f"[COPD] Indexed beats. Valid files: {len(self.paths)-bad}  Skipped: {bad}")

        if not self.items:
            raise RuntimeError("Found files but no valid beats after slicing. Check RWL/RRI and magnification.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fi, r360, r128 = self.items[idx]
        buf = self._cache[fi]
        x360 = buf['x360']
        s = r360 - self.pre_360
        e = r360 + self.post_360
        beat = x360[s:e].copy().astype(np.float32)

        return {
            'cardiac_cycle': beat,                      # (216,)
            'beat_type': self.beat_type,
            'label': np.array(1, dtype=np.int64),       # keeps ToTensor happy
            'src_path': self.paths[fi],
            'r_index_128': np.array(r128, dtype=np.int64),
            'r_index_360': np.array(r360, dtype=np.int64),
        }
