#!/usr/bin/env python3
import argparse, os, re, sys
from pathlib import Path
import pandas as pd

def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

def main():
    ap = argparse.ArgumentParser(
        description="Partition runs into bg / no-bg view folders via symlinks using analysis manifest(s)."
    )
    ap.add_argument("--analysis_manifest", required=True, nargs="+",
                    help="One or more CSVs with at least 'ckpt_path' column "
                         "(e.g., results/.../analysis/analysis_manifest.csv).")
    ap.add_argument("--bg_out_dir", required=True,
                    help="Output folder for BG view, e.g. ./checkpoints/results/MIXED/full_bg")
    ap.add_argument("--nobg_out_dir", required=True,
                    help="Output folder for NO-BG view, e.g. ./checkpoints/results/MIXED/full_no-bg")
    ap.add_argument("--force_label", choices=["bg","no-bg"], default=None,
                    help="Force assign all rows from these manifest(s) to this label (recommended when you have separate bg/no-bg manifests).")
    ap.add_argument("--bg_pattern", default=r"(^|[^a-z])bg([^a-z]|$)", help="Regex for BG detection (if not forcing).")
    ap.add_argument("--nobg_pattern", default=r"no-?bg", help="Regex for NO-BG detection (if not forcing).")
    ap.add_argument("--columns", nargs="+",
                    default=["ckpt_path","val_csv","test_csv","report_dir","cfg_name","tag","run_id"],
                    help="Columns to scan for patterns when not forcing.")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be done without creating symlinks.")
    args = ap.parse_args()

    bg_re   = re.compile(args.bg_pattern, flags=re.I)
    nb_re   = re.compile(args.nobg_pattern, flags=re.I)

    out_bg  = Path(args.bg_out_dir)
    out_nb  = Path(args.nobg_out_dir)

    total_bg = total_nb = 0
    unclassified = []

    for manifest_path in args.analysis_manifest:
        df = pd.read_csv(manifest_path)

        if "ckpt_path" not in df.columns:
            print(f"[fatal] Manifest {manifest_path} lacks 'ckpt_path' column.", file=sys.stderr)
            sys.exit(2)

        # Ensure all 'scan columns' exist so string join works
        for c in args.columns:
            if c not in df.columns:
                df[c] = ""

        for _, row in df.iterrows():
            ckpt = Path(str(row["ckpt_path"])).expanduser()
            # Make absolute so symlink targets are stable
            if not ckpt.is_absolute():
                ckpt = (Path.cwd() / ckpt).resolve()

            run_dir = ckpt.parent           # .../full/<run_id>
            run_id  = run_dir.name
            test_dir = run_dir.parent / "final_test" / run_id

            # Classification
            if args.force_label:
                label = args.force_label
            else:
                hay = " | ".join(str(row[c]) for c in args.columns if c in row and pd.notna(row[c]))
                # Also inject paths into the haystack
                hay = f"{hay} | {str(ckpt)} | {str(run_dir)}"
                if nb_re.search(hay):
                    label = "no-bg"
                elif bg_re.search(hay):
                    label = "bg"
                else:
                    label = None

            if label is None:
                unclassified.append(run_id)
                continue

            if label == "bg":
                dest = out_bg / run_id
                if not args.dry_run:
                    symlink(run_dir, dest)
                # link test dir if present
                if test_dir.is_dir():
                    tdest = out_bg / "final_test" / run_id
                    if not args.dry_run:
                        symlink(test_dir, tdest)
                total_bg += 1

            else:  # no-bg
                dest = out_nb / run_id
                if not args.dry_run:
                    symlink(run_dir, dest)
                if test_dir.is_dir():
                    tdest = out_nb / "final_test" / run_id
                    if not args.dry_run:
                        symlink(test_dir, tdest)
                total_nb += 1

    print(f"[ok] BG linked: {total_bg}   NO-BG linked: {total_nb}")
    if unclassified:
        print(f"[warn] Unclassified runs: {len(unclassified)} (adjust --force_label or patterns). "
              f"Examples: {unclassified[:5]}")

if __name__ == "__main__":
    main()
