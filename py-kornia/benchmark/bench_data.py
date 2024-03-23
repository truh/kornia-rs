from typing import Callable, Optional, Tuple
from pathlib import Path
import argparse
from tqdm import tqdm

import torch
import torch.utils.data as data
import kornia_rs as KR
import kornia as K
import cv2
import albumentations as A

import time
import timeit


@torch.inference_mode()
def kornia_transforms(img_path: str):
    img = KR.read_image_jpeg(img_path)
    # img = KR.resize(img, (32, 32), interpolation="bilinear")
    img = K.utils.image_to_tensor(img)
    return img.float() / 255.0


def opencv_transforms(img_path: str):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    return img


class Augmentator:
    def __init__(self, backend: str) -> None:
        self.backend = backend

        if "kornia" in backend:
            self.augmentation = K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1)
        elif backend == "opencv":
            self.augmentation = A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            )

    def __call__(self, img):
        if self.backend == "kornia_rs":
            img = self.augmentation(img)
        elif self.backend == "opencv":
            img = self.augmentation(image=img)["image"]
            img = K.utils.image_to_tensor(img).float() / 255.0
        return img


class ImageDataset(data.Dataset):
    def __init__(self, root: Path, backend: str, augment) -> None:
        self.root = root
        self.backend = backend
        self.images = list(self.root.glob("*.jpeg"))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        if self.backend == "kornia_rs":
            img = kornia_transforms(img_path)
            img = self.augment(img)
        elif self.backend == "opencv":
            img = opencv_transforms(img_path)
            img = self.augment(img)
        elif self.backend in ["kornia_cpu", "kornia_gpu"]:
            img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)

        return img


def load_dataset_pipeline(images_dir: str, backend: str, num_workers: int):

    augment = Augmentator(backend)

    dataset = ImageDataset(Path(images_dir), backend=backend, augment=augment)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=num_workers
    )

    for sample in dataloader:
        with torch.inference_mode():
            if backend == "kornia_cpu" or backend == "kornia_gpu":
                if "gpu" in backend:
                    sample = sample.cuda()

                img = augment(sample)


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
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="number of iterations to run the benchmark",
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
    print(f"Number of iterations: {args.num_iterations}")

    print(f"Time: {timer.timeit(args.num_iterations) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
