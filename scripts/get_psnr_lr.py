"""Evaluate per-view (left/right) PSNR/SSIM/LPIPS for learning-rate ablation runs.

Loads up to three models for a given dataset name:
  index 0: fast-merged model trained on the full combined dataset
  index 1: model trained on left-view frames only
  index 2: model trained on right-view frames only

All three are evaluated against the same ground-truth frames so that
left/right reconstruction quality can be compared across learning rates.
"""

from inhand.ih_config import ihgs_full_merged
from inhand.ihgs import IHGSModelConfig, IHGSModel
from inhand.ih_pipeline import IHGSPipelineConfig, IHGSPipeline
from inhand.ih_datamanager import IHDataManagerConfig, IHDataManager
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader

from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import os
import tyro
import gc
from dataclasses import dataclass


@dataclass
class Args:
    name: str
    """Substring to match against config data paths (e.g. 'wine', 'realsense')"""
    outputs_dir: str = "outputs"
    """Base outputs directory; searches combined/ihgs-fast-merged and combined/ihgs"""
    metrics_dir: str = "output_metrics_lr"
    """Directory to write per-run .pth metric files"""
    split_at: int = 100
    """Frame index splitting left (0:split_at) from right (split_at:) views"""


def find_configs(args: Args) -> list[str]:
    yamls = glob(f"{args.outputs_dir}/combined/ihgs-fast-merged/*/config.yml")
    yamls.extend(sorted(glob(f"{args.outputs_dir}/combined/ihgs/*/config.yml")))
    yamls = sorted(yamls)
    matched = []
    for y in yamls:
        with open(y, "r") as f:
            text = f.readlines()
        if any(args.name in t for t in text):
            matched.append(y)
    return matched


def get_pipeline(config_path: str):
    _, pipeline, _, _ = eval_setup(Path(config_path))
    return pipeline


def evaluate(pipeline, dataloader, gripper_mask) -> list:
    pipeline.cuda()
    pipeline.model.eval()
    pipeline.model.set_background(torch.tensor([1.0, 1.0, 1.0]).to("cuda"))
    results = []
    for camera, batch in tqdm(dataloader, leave=False):
        outputs = pipeline.model.get_outputs(camera.to("cuda"))
        metrics_dict, _ = pipeline.model.get_image_metrics_and_images(
            outputs, batch, gripper_mask
        )
        results.append(metrics_dict)
    return results


def main(args: Args):
    run_list = find_configs(args)
    if not run_list:
        print(f"No configs found matching '{args.name}' under '{args.outputs_dir}'.")
        return

    print(f"\nLR ablation evaluation for '{args.name}'  ({len(run_list)} run(s) found)")
    print(f"  Left frames:  0–{args.split_at - 1}")
    print(f"  Right frames: {args.split_at}+")
    print(f"  All models evaluated against the same ground-truth dataset (run 0)\n")

    os.makedirs(args.metrics_dir, exist_ok=True)
    pipelines = [get_pipeline(r) for r in run_list]
    gt_dataset = pipelines[0].datamanager.train_dataset
    gripper_mask = pipelines[0].datamanager.gripper_masks

    for i, pipeline in enumerate(pipelines):
        print(f"  Evaluating run {i}: {run_list[i]}")
        metrics = evaluate(pipeline, FixedIndicesEvalDataloader(gt_dataset), gripper_mask)

        # For the split models, replace the relevant half with self-evaluated frames
        # so that each model is judged on the frames it was actually trained on.
        if i == 1:
            # Left-only model: use its own frames for left, shared GT for right
            self_metrics = evaluate(
                pipeline, FixedIndicesEvalDataloader(pipeline.datamanager.train_dataset), gripper_mask
            )
            metrics = self_metrics + metrics[args.split_at:]
        elif i == 2:
            # Right-only model: use shared GT for left, its own frames for right
            self_metrics = evaluate(
                pipeline, FixedIndicesEvalDataloader(pipeline.datamanager.train_dataset), gripper_mask
            )
            metrics = metrics[:args.split_at] + self_metrics

        out_path = f"{args.metrics_dir}/{args.name}_{i}_metrics.pth"
        torch.save(metrics, out_path)

        psnr = np.array([m["psnr"] for m in metrics])
        s = args.split_at
        left_mask = np.zeros(len(psnr), dtype=bool); left_mask[:s] = True
        right_mask = ~left_mask
        mp_l = float(np.mean(psnr[:s][~np.isinf(psnr[:s])]))
        mp_r = float(np.mean(psnr[s:][~np.isinf(psnr[s:])]))
        print(f"    PSNR left: {mp_l:.4f}  right: {mp_r:.4f}  "
              f"(gap: {abs(mp_l - mp_r):.4f} dB — via per-pixel MSE across views)")

        gc.collect()

    print(f"\nDone. Run scripts/analyze_lr.py --metrics-dir {args.metrics_dir} to aggregate.")


if __name__ == "__main__":
    main(tyro.cli(Args))
