import os
import json
from pathlib import Path
from PIL import Image

# Optional HEIC/HEIF support (won't fail if not installed)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import piexif

# Change this to your root folder
ROOT = Path("../../data/Pantone/original/focus/")

# Extensions to include
EXTS = (".jpg", ".jpeg", ".heic", ".heif")


def exif_to_grouped_dict(exif_dict):
    """
    Convert piexif dict (with numeric tags) into a grouped, readable dict:
    { "0th": {...}, "Exif": {...}, "GPS": {...}, "1st": {...} }
    """
    grouped = {}
    for ifd_name in ("0th", "Exif", "GPS", "1st"):
        if ifd_name not in exif_dict:
            continue
        ifd = exif_dict[ifd_name]
        if not isinstance(ifd, dict):
            continue
        out = {}
        tags_map = piexif.TAGS.get(ifd_name, {})
        for tag_id, value in ifd.items():
            tag_name = tags_map.get(tag_id, {"name": str(tag_id)})["name"]
            # Convert bytes to string if possible
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore")
                except Exception:
                    value = str(value)
            out[tag_name] = value
        if out:
            grouped[ifd_name] = out
    return grouped


def extract_one(image_path: Path):
    meta = {
        "file": image_path.name,
        "relative_path": str(image_path.relative_to(ROOT)),
        "format": None,
        "mode": None,
        "size": None,  # [width, height]
        "exif": {},    # grouped dict here
    }

    try:
        with Image.open(image_path) as im:
            meta["format"] = im.format
            meta["mode"] = im.mode
            meta["size"] = [im.width, im.height]
    except Exception as e:
        meta["error_open"] = str(e)

    try:
        exif_dict = piexif.load(str(image_path))
        grouped = exif_to_grouped_dict(exif_dict)
        if grouped:
            meta["exif"] = grouped
    except Exception as e:
        meta["exif_error"] = str(e)

    return meta


def main():
    files = list(ROOT.rglob("*"))
    image_files = [p for p in files if p.is_file() and p.suffix.lower() in EXTS]

    if not image_files:
        print(f"[WARN] No images found under {ROOT.resolve()} with extensions {EXTS}")
        return

    for p in sorted(image_files):
        print(f"[INFO] Processing {p} ...")
        meta = extract_one(p)
        out_path = p.with_suffix(".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[OK]  Saved → {out_path}")


if __name__ == "__main__":
    main()
