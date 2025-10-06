import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
from math import ceil

csv = "/home/data/Pantone/test/test.csv"
original = "/home/data/Pantone/original"

def generate_summary_from_csv(csv_path, originals_root, folder_meta="/home/data/Pantone/original/desc.txt", pdf_path="/home/data/Pantone/test/test.pdf", n_per_class=5):
    """
    Generates a PDF summary from a CSV test set.
    Shows original + cropped images grouped by color class.
    """
    # ----------------
    # Load data
    # ----------------
    df = pd.read_csv(csv_path)

    # Load folder descriptions
    folder_descriptions = {} 
    with open(folder_meta, "r", encoding="utf-8") as f: 
        for line in f: 
            line = line.strip() 
            if not line or ":" not in line: 
                continue 
            key, desc = line.split(":", 1) 
            folder_descriptions[key.strip()] = desc.strip()

    # Build samples
    samples = []
    for _, row in df.iterrows():
        basename = str(row["basename"]).zfill(3)
        folder = row["folder"]
        cropped_path = row["image_path"]
        original_path = os.path.join(originals_root, folder, f"{basename}.jpg")

        samples.append({
            "color": basename,  # assuming basename == color class
            "folder": folder,
            "original_path": original_path,
            "cropped_path": cropped_path,
        })

    # ----------------
    # Generate PDF
    # ----------------
    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        cover_text = ["Folder descriptions:"]
        for folder, desc in folder_descriptions.items():
            cover_text.append(f"{folder}: {desc}")

        ax.text(0.05, 0.95, "\n".join(cover_text), va="top", ha="left", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Group by class (color)
        grouped = {}
        for s in samples:
            grouped.setdefault(s["color"], []).append(s)

        for cls, group in sorted(grouped.items(), key=lambda x: int(x[0])):
            n_samples = len(group)
            n_rows = ceil(n_samples / n_per_class)

            fig, axes = plt.subplots(
                n_rows, n_per_class * 2, figsize=(3 * n_per_class * 2, 3 * n_rows)
            )
            fig.suptitle(
                f"Class {cls}\nFolders: {', '.join(sorted(set(s['folder'] for s in group)))}",
                fontsize=14,
            )

            # Flatten axes
            if n_rows == 1:
                axes = [axes]
            axes_flat = [
                ax
                for row in axes
                for ax in (row if isinstance(row, (list, np.ndarray)) else [row])
            ]

            for i, s in enumerate(group):
                # Original
                if os.path.exists(s["original_path"]):
                    img_orig = Image.open(s["original_path"])
                    axes_flat[i * 2].imshow(img_orig)
                axes_flat[i * 2].set_title("Original")
                axes_flat[i * 2].axis("off")

                # Cropped
                if os.path.exists(s["cropped_path"]):
                    img_crop = Image.open(s["cropped_path"])
                    axes_flat[i * 2 + 1].imshow(img_crop)
                axes_flat[i * 2 + 1].set_title("Cropped")
                axes_flat[i * 2 + 1].axis("off")

            # Unused axes
            for j in range(i * 2 + 2, len(axes_flat)):
                axes_flat[j].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved at {pdf_path}")
    return samples



generate_summary_from_csv(csv, original)