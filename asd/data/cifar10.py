"""CIFAR-10 / CIFAR-100 / SVHN data loaders for ASD training, profiling, and evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _svhn(root, train, download, transform):
    """SVHN wrapper that mimics the CIFAR dataset API (train=True/False)."""
    split = "train" if train else "test"
    return datasets.SVHN(root=root, split=split, download=download, transform=transform)


# Per-dataset normalization stats
_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "cls": datasets.CIFAR10,
        "num_classes": 10,
        "augment": "standard",  # RandomCrop(4) + HFlip
    },
    "cifar100": {
        "mean": (0.5071, 0.4866, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "cls": datasets.CIFAR100,
        "num_classes": 100,
        "augment": "standard",
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "cls": _svhn,
        "num_classes": 10,
        # No HFlip for SVHN — digits don't look the same flipped.
        "augment": "svhn",
    },
}

# Legacy constants kept for back-compat
CIFAR10_MEAN = _STATS["cifar10"]["mean"]
CIFAR10_STD = _STATS["cifar10"]["std"]


def _build_transform(augmentation: str, mean, std) -> transforms.Compose:
    normalize = transforms.Normalize(mean, std)
    if augmentation == "standard":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == "svhn":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == "strong":
        # Stronger augmentation — RandAugment adds photometric variety that
        # shifts the teacher's soft-label manifold; especially helpful for the
        # student at high compression where it has to learn robust features.
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == "none":
        return transforms.Compose([transforms.ToTensor(), normalize])
    raise ValueError(f"Unknown augmentation: {augmentation!r}")


def get_cifar_loaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: str = "standard",
    calibration_samples: int | None = None,
) -> dict[str, DataLoader]:
    """Unified loader builder for CIFAR-10 or CIFAR-100.

    Returns a dict with keys "train", "test", and (if calibration_samples is
    given) "calibration".
    """
    if dataset not in _STATS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Available: {list(_STATS)}")
    stats = _STATS[dataset]
    cls = stats["cls"]

    # If caller asks for "standard" on a dataset with a different default aug
    # (e.g. SVHN doesn't want HFlip), honor the dataset's preference.
    effective_aug = augmentation
    if augmentation == "standard" and stats.get("augment") != "standard":
        effective_aug = stats["augment"]

    train_transform = _build_transform(effective_aug, stats["mean"], stats["std"])
    eval_transform = _build_transform("none", stats["mean"], stats["std"])

    train_set = cls(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = cls(root=data_dir, train=False, download=True, transform=eval_transform)

    pin_memory = torch.cuda.is_available()

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=True, persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        ),
    }
    if calibration_samples is not None:
        calib_set = cls(root=data_dir, train=True, download=True, transform=eval_transform)
        n = min(calibration_samples, len(calib_set))
        gen = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(calib_set), generator=gen)[:n].tolist()
        loaders["calibration"] = DataLoader(
            Subset(calib_set, indices),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        )
    return loaders


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: str = "standard",
    calibration_samples: int | None = None,
) -> dict[str, DataLoader]:
    """Build CIFAR-10 train/test loaders (and optional calibration loader).

    Args:
        data_dir: Directory to download CIFAR-10 into (created if missing).
        batch_size: Batch size for all loaders.
        num_workers: DataLoader worker count.
        augmentation: "standard" (random crop + flip) or "none" (normalize only).
            Applied only to the training loader. Test/calibration always use "none".
        calibration_samples: If set, also return a "calibration" loader — a
            deterministic random subset of the training set with no augmentation,
            used for activation profiling.

    Returns:
        Dict with keys "train", "test", and optionally "calibration".
    """
    # Legacy wrapper around get_cifar_loaders for CIFAR-10.
    return get_cifar_loaders(
        dataset="cifar10",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation=augmentation,
        calibration_samples=calibration_samples,
    )


