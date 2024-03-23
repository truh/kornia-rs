from typing import Callable, Optional, Tuple
from pathlib import Path
import argparse
from tqdm import tqdm

import torch
import torch.utils.data as data
import kornia_rs as KR
import kornia as K
import cv2

import time
import timeit


def kornia_transforms(img_path: str):
    img = KR.read_image_jpeg(img_path)
    img = KR.resize(img, (32, 32), interpolation="bilinear")
    return img


def opencv_transforms(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    return img


class ImageDataset(data.Dataset):
    def __init__(self, root: Path, backend: str) -> None:
        self.root = root
        self.backend = backend
        self.images = list(self.root.glob("*.jpeg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        if self.backend == "kornia_rs":
            img = kornia_transforms(img_path)
        elif self.backend == "opencv":
            img = opencv_transforms(img_path)
        elif self.backend in ["kornia_cpu", "kornia_gpu"]:
            img = K.io.load_image(img_path, K.io.ImageLoadType.RGB8)

        return img


@torch.compile
def jresize(img: torch.Tensor):
    return K.geometry.resize(img, (32, 32), interpolation="bilinear")


def load_dataset_pipeline(images_dir: str, backend: str, num_workers: int):

    dataset = ImageDataset(Path(images_dir), backend=backend)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=num_workers
    )

    for sample in dataloader:
        with torch.inference_mode():
            if backend == "kornia_cpu":
                # img = K.geometry.resize(sample, (32, 32), interpolation="bilinear")
                img = jresize(sample)
            elif backend == "kornia_gpu":
                img = K.geometry.resize(
                    sample.cuda(), (32, 32), interpolation="bilinear"
                )


def main():
    parser = argparse.ArgumentParser(description="Kornia Benchmark Data Generation")
    parser.add_argument(
        "--images-dir", type=str, required=True, help="path to the images folder"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="path to the output folder"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="number of workers to load the data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="number of workers to load the data"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="kornia",
        choices=["kornia_cpu", "kornia_gpu", "kornia_rs", "opencv"],
        help="backend to use for the benchmark",
    )
    args = parser.parse_args()

    timer = timeit.Timer(
        stmt="load_dataset_pipeline(images_dir, backend, num_workers)",
        setup="from __main__ import load_dataset_pipeline",
        globals={
            "images_dir": args.images_dir,
            "backend": args.backend,
            "num_workers": args.num_workers,
        },
    )
    print(
        f"Running benchmark for {args.backend} backend with {args.num_workers} workers"
    )
    print(f"Images folder: {args.images_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Backend: {args.backend}")
    print(f"Time: {timer.timeit(10) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
