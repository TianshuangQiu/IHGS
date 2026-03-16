"""Aggregate per-view (left/right) PSNR/SSIM/LPIPS for learning-rate ablation runs."""

import torch
from glob import glob
import os
import numpy as np
import json
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    metrics_dir: str = "output_metrics_lr"
    """Directory containing per-run .pth metric files"""
    split_at: int = 100
    """Frame index that splits left-view frames (0:split_at) from right-view frames (split_at:)"""
    output: str = ""
    """Path to save aggregated JSON (default: {metrics_dir}/metrics.json)"""


def main(args: Args):
    output_path = args.output or f"{args.metrics_dir}/metrics.json"
    files = sorted(glob(f"{args.metrics_dir}/*.pth"))
    out_dict = {}

    if not files:
        print(f"No .pth files found under '{args.metrics_dir}/'.")
        return

    print(f"\nLearning-rate ablation — per-view metrics ({args.metrics_dir})")
    print(f"  Left view:  frames 0–{args.split_at - 1}  (captured from the left stereo camera)")
    print(f"  Right view: frames {args.split_at}+  (captured from the right stereo camera)")
    print("  A large left/right gap suggests the model fit one view better,")
    print("  which may indicate an imbalanced learning rate across the two views.\n")
    print("  PSNR/SSIM higher is better | LPIPS lower is better\n")

    for f in files:
        run_dict = torch.load(f)
        psnr, ssim, lpips = [], [], []
        for r in run_dict:
            psnr.append(r["psnr"])
            ssim.append(r["ssim"])
            lpips.append(r["lpips"])
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        lpips = np.array(lpips)

        s = args.split_at
        mean = lambda arr, mask: float(np.mean(arr[mask & ~np.isinf(arr)]))
        full_mask = np.ones(len(psnr), dtype=bool)
        left_mask = np.zeros(len(psnr), dtype=bool)
        left_mask[:s] = True
        right_mask = ~left_mask

        mp = mean(psnr, full_mask)
        ms = mean(ssim, full_mask)
        ml = mean(lpips, full_mask)
        mp_l = mean(psnr, left_mask)
        ms_l = mean(ssim, left_mask)
        ml_l = mean(lpips, left_mask)
        mp_r = mean(psnr, right_mask)
        ms_r = mean(ssim, right_mask)
        ml_r = mean(lpips, right_mask)

        name = os.path.basename(f)
        out_dict[name] = {
            "psnr": mp, "ssim": ms, "lpips": ml,
            "psnr_left": mp_l, "ssim_left": ms_l, "lpips_left": ml_l,
            "psnr_right": mp_r, "ssim_right": ms_r, "lpips_right": ml_r,
        }

        imbalance = abs(mp_l - mp_r)
        bias = "left" if mp_l > mp_r else "right"
        print(f"  {name}")
        print(f"    Overall — PSNR: {mp:.4f}  SSIM: {ms:.4f}  LPIPS: {ml:.4f}")
        print(f"    Left    — PSNR: {mp_l:.4f}  SSIM: {ms_l:.4f}  LPIPS: {ml_l:.4f}")
        print(f"    Right   — PSNR: {mp_r:.4f}  SSIM: {ms_r:.4f}  LPIPS: {ml_r:.4f}")
        print(f"    → {bias.capitalize()} view reconstructed better "
              f"(PSNR gap: {imbalance:.4f} dB via per-pixel MSE difference)\n")

    with open(output_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"  Aggregated metrics saved to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
