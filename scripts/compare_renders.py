import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    folder1: str
    """Path to first defect detection result folder (e.g. defect_detection/wire-plug)"""
    folder2: str
    """Path to second defect detection result folder (e.g. defect_detection/wire-plug-2)"""
    acc_threshold: float = 0.7
    """Accumulation threshold for masking"""
    save_frame: int = 80
    """Frame index to save as depth output"""
    output: str = "depth.npy"
    """Output path for the saved depth frame"""


def main(args: Args):
    frames1 = np.load(f"{args.folder1}/frames.npy")
    frames2 = np.load(f"{args.folder2}/frames.npy")
    acc1 = np.load(f"{args.folder1}/accs.npy")
    acc2 = np.load(f"{args.folder2}/accs.npy")

    all_frames = np.concatenate([frames1, frames2[:, 1:]], axis=1)
    all_frames[:, 0] = all_frames[:, 0] - np.mean(all_frames[:, 0])
    all_frames[:, 1] = all_frames[:, 1] - np.mean(all_frames[:, 1])
    all_frames[:, 2] = all_frames[:, 2] - np.mean(all_frames[:, 2])
    all_accs = np.concatenate([acc1, acc2[:, 1:]], axis=1)
    n_frames, n_views = all_frames.shape[:2]
    print(f"\nLoaded {n_frames} frames, {n_views} views")
    print(f"  View 0: '{args.folder1}' — reference (e.g. good object)")
    print(f"  View 1: '{args.folder1}' — comparison render (e.g. damaged object, same camera)")
    print(f"  View 2: '{args.folder2}' — additional view from second dataset run")
    print(f"  Accumulation threshold for masking: {args.acc_threshold} "
          f"(pixels below this are background and excluded from comparison)\n")

    t = args.acc_threshold
    diff1 = np.abs(
        all_frames[:, 0] * (all_accs[:, 0, :, :] > t)
        - all_frames[:, 1] * (all_accs[:, 1, :, :] > t)
    )
    diff2 = np.abs(
        all_frames[:, 1] * (all_accs[:, 1, :, :] > t)
        - all_frames[:, 2] * (all_accs[:, 2, :, :] > t)
    )
    diff3 = np.abs(
        all_frames[:, 0] * (all_accs[:, 0, :, :] > t)
        - all_frames[:, 2] * (all_accs[:, 2, :, :] > t)
    )

    m1, m2, m3 = np.mean(diff1), np.mean(diff2), np.mean(diff3)
    print("Per-pixel L1 difference (lower = more similar):")
    print(f"  diff1 (view0 vs view1 — good vs damaged, same camera):   {m1:.6f}")
    print(f"  diff2 (view1 vs view2 — damaged vs alt-run, noise floor): {m2:.6f}")
    print(f"  diff3 (view0 vs view2 — good vs alt-run, cross-check):   {m3:.6f}")

    if m1 > m2 * 1.2:
        print(f"\n  → diff1 ({m1:.6f}) is notably larger than diff2 ({m2:.6f}): "
              f"the good-vs-damaged difference exceeds the noise floor, "
              f"suggesting a real geometric or appearance change between the two objects.")
    else:
        print(f"\n  → diff1 ({m1:.6f}) is close to diff2 ({m2:.6f}): "
              f"the difference between the two objects is within the noise floor. "
              f"The objects may be visually indistinguishable at this threshold.")

    np.save(
        args.output,
        all_frames[args.save_frame] * (all_accs[args.save_frame, 0, :, :] > t),
    )
    print(f"\n  Frame {args.save_frame} depth/frame saved to: {args.output}")

    from skimage.metrics import structural_similarity as ssim
    from tqdm import tqdm

    ssims = []
    for frame in tqdm(all_frames, desc="Computing SSIM"):
        ssims.append(
            [
                ssim(frame[0], frame[1], multichannel=True, data_range=1, channel_axis=2),
                ssim(frame[1], frame[2], multichannel=True, data_range=1, channel_axis=2),
                ssim(frame[0], frame[2], multichannel=True, data_range=1, channel_axis=2),
            ]
        )

    ssims = np.array(ssims)
    s1, s2, s3 = np.mean(ssims, axis=0)
    print(f"\nMean SSIM (higher = more similar, 0–1):")
    print(f"  view0 vs view1 (good vs damaged):     {s1:.4f}")
    print(f"  view1 vs view2 (damaged vs alt-run):  {s2:.4f}")
    print(f"  view0 vs view2 (good vs alt-run):     {s3:.4f}")
    if s1 < s2 - 0.02:
        print(f"\n  → SSIM for good-vs-damaged ({s1:.4f}) is lower than the noise baseline "
              f"({s2:.4f}), confirming a structurally meaningful difference between objects.")


if __name__ == "__main__":
    main(tyro.cli(Args))
