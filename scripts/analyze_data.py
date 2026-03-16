"""Aggregate PSNR/SSIM/LPIPS metrics from per-run output folders."""

import torch
from glob import glob
import os
import numpy as np
import json
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    metrics_dir: str = "output_metrics"
    """Directory containing per-run subfolders, each with a metrics.pth file"""
    output: str = ""
    """Path to save aggregated JSON (default: {metrics_dir}/metrics.json)"""


def main(args: Args):
    output_path = args.output or f"{args.metrics_dir}/metrics.json"
    folders = sorted(glob(f"{args.metrics_dir}/*/"))
    out_dict = {}
    results = []

    for folder in folders:
        metrics_path = os.path.join(folder, "metrics.pth")
        if not os.path.exists(metrics_path):
            continue

        run_dict = torch.load(metrics_path)
        psnr, ssim, lpips = [], [], []
        for r in run_dict:
            psnr.append(r["psnr"])
            ssim.append(r["ssim"])
            lpips.append(r["lpips"])
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        lpips = np.array(lpips)

        # Try to read dataset name from the config path file written during eval
        cfgpath_file = os.path.join(folder, "merged", "cfgpath.txt")
        try:
            with open(cfgpath_file, "r") as f:
                name = f.readlines()[1].strip()
        except (FileNotFoundError, IndexError):
            name = os.path.basename(folder.rstrip("/"))

        mean_psnr = float(np.mean(psnr[~np.isinf(psnr)]))
        mean_ssim = float(np.mean(ssim[~np.isinf(ssim)]))
        mean_lpips = float(np.mean(lpips[~np.isinf(ssim)]))

        if name in out_dict:
            name = name + "_2"
        out_dict[name] = {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips}
        results.append((name, mean_psnr, mean_ssim, mean_lpips))

    if not results:
        print(f"No metrics.pth files found under '{args.metrics_dir}/'.")
        return

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nReconstruction quality metrics — {args.metrics_dir}")
    print("  PSNR: higher is better (log-scale pixel fidelity)")
    print("  SSIM: higher is better (structural/perceptual similarity, 0–1)")
    print("  LPIPS: lower is better (learned perceptual distance)")
    print()
    print(f"  {'Dataset':<45} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("  " + "-" * 73)
    for name, psnr, ssim, lpips in results:
        print(f"  {name:<45} {psnr:>8.4f} {ssim:>8.4f} {lpips:>8.4f}")

    best = results[0]
    worst = results[-1]
    print(f"\n  Best:  {best[0]}  (PSNR {best[1]:.4f})")
    print(f"  Worst: {worst[0]}  (PSNR {worst[1]:.4f})")
    print(f"\n  Total runs evaluated: {len(results)}")

    with open(output_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n  Aggregated metrics saved to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
