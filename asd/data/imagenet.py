"""ImageNet loader that streams from s3://ray-example-data/imagenet/.

The reference bucket ships a parquet file with 803k training-image URLs (one
per row, as `image_url`). Labels are inferred from the WordNet ID embedded in
the URL path (`.../train/<wnid>/<image>.JPEG`). Since the `test/` split in that
bucket has no public labels, we hold out `val_per_class` images per class from
the training URLs as a deterministic validation set.

Key design notes:

- Reads are anonymous (the bucket is public). We use boto3 with UNSIGNED
  config — lighter than fsspec and one connection per worker process.
- On-disk cache in `cache_dir` lets multi-epoch training avoid re-downloading
  (first epoch is I/O-bound; subsequent epochs run at GPU speed).
- The cache is content-addressed by the S3 key path, so it's safe to share
  across different subsample sizes or splits.
"""

from __future__ import annotations

import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard ImageNet normalization (matches torchvision pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_WNID_RE = re.compile(r"/train/([^/]+)/")

# Default S3 bucket + parquet key
DEFAULT_BUCKET = "ray-example-data"
DEFAULT_PARQUET_KEY = "imagenet/metadata_file.parquet"


@dataclass
class ImageNetSplit:
    urls: list[str]
    labels: list[int]
    wnid_to_idx: dict[str, int]


def _s3_anon_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _url_to_bucket_key(url: str) -> tuple[str, str]:
    # s3://bucket/key/path.jpg -> ("bucket", "key/path.jpg")
    assert url.startswith("s3://"), url
    body = url[len("s3://") :]
    bucket, _, key = body.partition("/")
    return bucket, key


def _load_metadata(
    bucket: str = DEFAULT_BUCKET,
    parquet_key: str = DEFAULT_PARQUET_KEY,
    local_cache: str = "/tmp/imagenet_metadata_file.parquet",
) -> list[str]:
    """Download (once) and parse the parquet, returning all train URLs."""
    if not os.path.exists(local_cache):
        client = _s3_anon_client()
        client.download_file(bucket, parquet_key, local_cache)
    import pyarrow.parquet as pq
    return pq.read_table(local_cache).column("image_url").to_pylist()


def build_splits(
    train_per_class: int = 100,
    val_per_class: int = 10,
    seed: int = 0,
    max_classes: int | None = None,
) -> tuple[ImageNetSplit, ImageNetSplit]:
    """Group all train-URL parquet rows by wnid, then hold `val_per_class` out
    per class for validation and keep (up to) `train_per_class` for training.
    Deterministic under `seed`. If `max_classes` is set, restrict to that many
    classes alphabetically (useful for smoke tests).
    """
    urls = _load_metadata()
    by_wnid: dict[str, list[str]] = defaultdict(list)
    for u in urls:
        m = _WNID_RE.search(u)
        if not m:
            continue
        by_wnid[m.group(1)].append(u)

    wnids = sorted(by_wnid.keys())
    if max_classes is not None:
        wnids = wnids[:max_classes]
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    g = torch.Generator().manual_seed(seed)
    train_urls, train_labels, val_urls, val_labels = [], [], [], []
    for w in wnids:
        pool = by_wnid[w]
        perm = torch.randperm(len(pool), generator=g).tolist()
        val_idx = perm[:val_per_class]
        train_idx = perm[val_per_class : val_per_class + train_per_class]
        lbl = wnid_to_idx[w]
        for i in val_idx:
            val_urls.append(pool[i]); val_labels.append(lbl)
        for i in train_idx:
            train_urls.append(pool[i]); train_labels.append(lbl)

    return (
        ImageNetSplit(train_urls, train_labels, wnid_to_idx),
        ImageNetSplit(val_urls, val_labels, wnid_to_idx),
    )


class S3ImageNet(Dataset):
    """Stream ImageNet JPEGs from S3, cache locally, apply transforms."""

    def __init__(
        self,
        split: ImageNetSplit,
        cache_dir: str = "/mnt/local_storage/imagenet_cache",
        transform=None,
    ):
        self.urls = split.urls
        self.labels = split.labels
        self.wnid_to_idx = split.wnid_to_idx
        self.cache_dir = cache_dir
        self.transform = transform
        os.makedirs(cache_dir, exist_ok=True)
        self._client = None  # Lazy per-worker

    def __len__(self) -> int:
        return len(self.urls)

    def _ensure_client(self):
        if self._client is None:
            self._client = _s3_anon_client()
        return self._client

    def _read_bytes(self, url: str) -> bytes:
        bucket, key = _url_to_bucket_key(url)
        cache_path = os.path.join(self.cache_dir, key)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        buf = io.BytesIO()
        self._ensure_client().download_fileobj(bucket, key, buf)
        data = buf.getvalue()
        # Write atomically: temp file + rename.
        tmp = cache_path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, cache_path)
        return data

    def __getitem__(self, idx: int):
        url = self.urls[idx]
        label = self.labels[idx]
        data = self._read_bytes(url)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_loaders(
    batch_size: int = 64,
    num_workers: int = 8,
    train_per_class: int = 100,
    val_per_class: int = 10,
    calibration_samples: int | None = None,
    cache_dir: str = "/mnt/local_storage/imagenet_cache",
    max_classes: int | None = None,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """Build train / test / (optional) calibration loaders streaming from S3.

    Returns dict with keys "train", "test", and (if calibration_samples is set)
    "calibration". The "test" loader uses the val-per-class hold-out — the
    upstream `test/` prefix has no labels.
    """
    train_split, val_split = build_splits(
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        max_classes=max_classes,
        seed=seed,
    )

    train_ds = S3ImageNet(train_split, cache_dir=cache_dir, transform=_train_transform())
    val_ds = S3ImageNet(val_split, cache_dir=cache_dir, transform=_eval_transform())

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=num_workers > 0),
        "test": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0),
    }

    if calibration_samples is not None:
        from torch.utils.data import Subset
        n = min(calibration_samples, len(train_split.urls))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_split.urls), generator=g)[:n].tolist()
        calib_split = ImageNetSplit(
            urls=[train_split.urls[i] for i in idx],
            labels=[train_split.labels[i] for i in idx],
            wnid_to_idx=train_split.wnid_to_idx,
        )
        calib_ds = S3ImageNet(calib_split, cache_dir=cache_dir, transform=_eval_transform())
        loaders["calibration"] = DataLoader(
            calib_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return loaders
