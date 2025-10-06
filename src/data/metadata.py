# src/data/metadata.py
import json
import pillow_heif
import pandas as pd
import numpy as np
pillow_heif.register_heif_opener()

# ---------- helpers ----------
def _frac_to_float(val):
    """Accept [num, den] or scalar; return float or None."""
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            den = float(val[1]) if float(val[1]) != 0 else np.nan
            return float(val[0]) / den
        return float(val)
    except Exception:
        return None

def _apex_to_seconds(tv):
    """APEX Tv = log2(1/t) => t = 2^-Tv."""
    try:
        tvf = float(tv)
        return 2.0 ** (-tvf)
    except Exception:
        return None

def _wb_auto_flag(v):
    """EXIF WhiteBalance often 0=Auto, 1=Manual. We output 1.0 for Auto, 0.0 for Manual."""
    try:
        vi = int(v)
        return 1.0 if vi == 0 else 0.0
    except Exception:
        return np.nan

# ---------- robust parser ----------
def process_metadata(json_data):
    """
    Parse EXIF-like JSON (your structure uses {"exif": {"Exif": ..., "0th": ...}}) into a single numeric row.
    Returns a 1-row DataFrame suitable for concatenation across samples.
    """
    # default row with NaNs (we'll fill later in dataset)
    row = {
        "iso_log":          np.nan,
        "shutter_s_log":    np.nan,  # log10(seconds)
        "fnumber":          np.nan,
        "exposure_bias_ev": np.nan,
        "brightness_ev":    np.nan,
        "focal_length_mm":  np.nan,
        "white_balance_auto": np.nan,
        "meta_missing":     0.0,     # set to 1.0 only when the *file* is missing/unreadable
        "Make":             None,    # raw string → will be one-hot'd
    }

    exif = (json_data or {}).get("exif", {})
    exif_main = exif.get("Exif", {}) or {}
    exif_0th  = exif.get("0th", {}) or {}

    # ISO
    iso = exif_main.get("ISOSpeedRatings") or exif_0th.get("ISOSpeedRatings")
    try:
        if iso is not None:
            iso_f = float(iso)
            if np.isfinite(iso_f) and iso_f > 0:
                row["iso_log"] = float(np.log10(iso_f))
    except Exception:
        pass

    # Shutter seconds: prefer ExposureTime fraction → seconds; else APEX ShutterSpeedValue → seconds
    t_s = _frac_to_float(exif_main.get("ExposureTime"))
    if t_s is None:
        t_s = _apex_to_seconds(_frac_to_float(exif_main.get("ShutterSpeedValue")))
    if t_s is not None and np.isfinite(t_s) and t_s > 0:
        row["shutter_s_log"] = float(np.log10(t_s))

    # Aperture (f-number)
    fnum = _frac_to_float(exif_main.get("FNumber"))
    if fnum is not None and np.isfinite(fnum):
        row["fnumber"] = float(fnum)

    # Exposure bias / Brightness / Focal length
    eb = _frac_to_float(exif_main.get("ExposureBiasValue"))
    if eb is not None and np.isfinite(eb):
        row["exposure_bias_ev"] = float(eb)

    bv = _frac_to_float(exif_main.get("BrightnessValue"))
    if bv is not None and np.isfinite(bv):
        row["brightness_ev"] = float(bv)

    fl = _frac_to_float(exif_main.get("FocalLength"))
    if fl is not None and np.isfinite(fl):
        row["focal_length_mm"] = float(fl)

    # WhiteBalance flag
    row["white_balance_auto"] = _wb_auto_flag(exif_main.get("WhiteBalance"))

    # Camera make (string → one-hot with prefix 'Make_')
    row["Make"] = exif_0th.get("Make")

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=["Make"], prefix="Make")
    return df

def make_missing_metadata_df():
    """
    Return a 1-row DataFrame representing a missing metadata file:
    meta_missing=1.0 and other numeric fields = NaN (filled later).
    """
    base = {
        "iso_log": np.nan,
        "shutter_s_log": np.nan,
        "fnumber": np.nan,
        "exposure_bias_ev": np.nan,
        "brightness_ev": np.nan,
        "focal_length_mm": np.nan,
        "white_balance_auto": np.nan,
        "meta_missing": 1.0,
    }
    return pd.DataFrame([base])

def process_metadata_json_file(json_path):
    try:
        with open(json_path, 'r') as f:
            metadata_json_file = json.load(f)
            return process_metadata(metadata_json_file)
    except FileNotFoundError:
        # caller should usually pass through make_missing_metadata_df()
        return make_missing_metadata_df()
    except json.JSONDecodeError:
        return make_missing_metadata_df()
    except Exception:
        return make_missing_metadata_df()
