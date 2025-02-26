import os
import tyro
import subprocess
from subprocess import Popen
import shutil
from glob import glob


def clean_folder(folder):
    subfolders = glob(os.path.join(folder, "*"))
    for subfolder in subfolders:
        if os.path.isdir(subfolder):
            if not os.path.basename(subfolder) in ["masks", "images", "gripper_masks"]:
                shutil.rmtree(subfolder)


def main(data_path: str, gpu_0: int = 0, gpu_1: int = 1):
    clean_folder(f"{data_path}/left")
    clean_folder(f"{data_path}/right")
    os.chdir("../inhand")
    subprocess.run(
        ["python", "scripts/create_tsfm.py", "--data_path", f"{data_path}/left"]
    )
    subprocess.run(
        ["python", "scripts/create_tsfm.py", "--data_path", f"{data_path}/right"]
    )
    os.chdir("../IHGS")
    commands = [
        f"CUDA_VISIBLE_DEVICES={gpu_0} ns-train ihgs --data {data_path}/left",
        f"CUDA_VISIBLE_DEVICES={gpu_1} ns-train ihgs --data {data_path}/right",
    ]
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()
    subprocess.call(
        f"python3 inhand/merge_dataset_with_opt.py --data_path {data_path}", shell=True
    )
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()
    subprocess.call(
        f"python3 inhand/merge_dataset_with_opt.py --data_path {data_path} --second_run",
        shell=True,
    )
    commands = [
        f"CUDA_VISIBLE_DEVICES={gpu_0} ns-train ihgs-full-merged --data {data_path}/combined",
        # f"CUDA_VISIBLE_DEVICES=1 ns-train ihgs-cross-merged --data {data_path}/combined",
    ]
    processes = [Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()


if __name__ == "__main__":
    tyro.cli(main)
