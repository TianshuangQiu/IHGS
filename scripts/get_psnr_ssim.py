"""Evaluate PSNR/SSIM/LPIPS on trained ihgs-full-merged models."""

from inhand.ih_config import ihgs_full_merged
from inhand.ihgs import IHGSModelConfig, IHGSModel
from inhand.ih_pipeline import IHGSPipelineConfig, IHGSPipeline
from inhand.ih_datamanager import IHDataManagerConfig, IHDataManager
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import os
import tyro
import gc
from dataclasses import dataclass
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader


@dataclass
class Args:
    outputs_dir: str = "outputs/combined/ihgs-full-merged"
    """Directory to search for training configs"""
    filter: str = ""
    """Only evaluate runs whose data path contains this substring (empty = all)"""
    exclude: str = "alt"
    """Skip runs whose data path contains this substring"""
    metrics_dir: str = "output_metrics_new"
    """Directory to write per-run metrics.pth files"""
    checkpoint_step: int = 49999
    """Only evaluate runs that have a checkpoint at this step"""


def find_configs(args: Args) -> dict[str, str]:
    """Return {run_name: config_path} for all matching completed runs."""
    pattern = f"{args.outputs_dir}/*/config.yml"
    yamls = sorted(glob(pattern), reverse=False)
    run_list = {}
    for y in yamls:
        ckpt = Path(y).parent / "nerfstudio_models" / f"step-{args.checkpoint_step:09d}.ckpt"
        if not ckpt.exists():
            continue
        with open(y, "r") as f:
            text = f.readlines()
        for t in text:
            words = t.split()
            filter_ok = (not args.filter) or any(args.filter in w for w in words)
            exclude_ok = (not args.exclude) or not any(args.exclude in w for w in words)
            if filter_ok and exclude_ok:
                name = t.split("- ")[-1].strip()
                if name in run_list:
                    name = name + "_2"
                run_list[name] = y
                break
    return run_list


def evaluate_run(yml_path: str, obj_name: str, metrics_dir: str) -> list:
    train_config, pipeline, _, _ = eval_setup(Path(yml_path))
    os.makedirs(f"{metrics_dir}/{obj_name}", exist_ok=True)
    with open(f"{metrics_dir}/{obj_name}/cfgpath.txt", "w") as f:
        f.write(yml_path + "\n" + obj_name)

    pipeline.cuda()
    pipeline.model.eval()
    pipeline.model.set_background(torch.tensor([1.0, 1.0, 1.0]).to("cuda"))
    dataloader = FixedIndicesEvalDataloader(pipeline.datamanager.train_dataset)

    metrics_dict_list = []
    for camera, batch in tqdm(dataloader, desc=obj_name, leave=False):
        outputs = pipeline.model.get_outputs(camera.to("cuda"))
        metrics_dict, _ = pipeline.model.get_image_metrics_and_images(
            outputs, batch, pipeline.datamanager.gripper_masks
        )
        metrics_dict_list.append(metrics_dict)
    return metrics_dict_list


def main(args: Args):
    run_list = find_configs(args)

    if not run_list:
        print(f"No completed runs found in '{args.outputs_dir}'.")
        if args.filter:
            print(f"  filter='{args.filter}', exclude='{args.exclude}'")
        return

    print(f"\nEvaluating {len(run_list)} run(s) from '{args.outputs_dir}'")
    print(f"  filter='{args.filter or '(none)'}', exclude='{args.exclude or '(none)'}'")
    print(f"  Writing metrics to: {args.metrics_dir}/\n")

    for name, yml_path in run_list.items():
        print(f"  → {name}")
        metrics = evaluate_run(yml_path, name, args.metrics_dir)
        torch.save(metrics, f"{args.metrics_dir}/{name}/metrics.pth")
        psnr_vals = [m["psnr"] for m in metrics]
        print(f"    mean PSNR: {np.mean([v for v in psnr_vals if not np.isinf(v)]):.4f}")
        gc.collect()

    print(f"\nDone. Run scripts/analyze_ablation.py --metrics-dir {args.metrics_dir} to aggregate.")


if __name__ == "__main__":
    main(tyro.cli(Args))
