"""Convert PNG mask files in a folder to grayscale JPG.

Nerfstudio expects masks as single-channel JPG files. Use this script to
convert PNG segmentation masks (e.g. from SAM or manual annotation) before
running the training pipeline.
"""

import os
import cv2
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    folder: str
    """Folder containing PNG mask files to convert"""
    delete_originals: bool = False
    """If set, remove the original PNG files after successful conversion"""


def main(args: Args):
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return

    png_files = [f for f in os.listdir(args.folder) if f.lower().endswith(".png")]
    if not png_files:
        print(f"No PNG files found in '{args.folder}'.")
        return

    print(f"\nConverting {len(png_files)} PNG mask(s) in '{args.folder}' to grayscale JPG...")
    converted, failed = 0, 0
    for filename in sorted(png_files):
        png_path = os.path.join(args.folder, filename)
        jpg_path = os.path.join(args.folder, filename.rsplit(".", 1)[0] + ".jpg")
        try:
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("cv2.imread returned None")
            cv2.imwrite(jpg_path, img)
            if args.delete_originals:
                os.remove(png_path)
            print(f"  {filename} → {os.path.basename(jpg_path)}")
            converted += 1
        except Exception as e:
            print(f"  Failed: {filename} — {e}")
            failed += 1

    print(f"\n  Converted: {converted}  Failed: {failed}")
    if args.delete_originals and converted:
        print(f"  Original PNG files removed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
