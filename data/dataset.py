"""TensorFlow data pipeline for FreiHAND landmark training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Sequence

import cv2
import numpy as np
import tensorflow as tf

try:
    from data.transforms import (
        compute_square_crop_box,
        crop_and_resize_image,
        jitter_square_crop_box,
        normalize_image,
        project_3D_to_2D,
    )
except ImportError:
    from transforms import (  # type: ignore
        compute_square_crop_box,
        crop_and_resize_image,
        jitter_square_crop_box,
        normalize_image,
        project_3D_to_2D,
    )

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "FreiHAND_pub_v2"


@dataclass(frozen=True)
class LandmarkSample:
    """Container for a single FreiHAND sample."""

    image_path: str
    intrinsics: np.ndarray
    keypoints_3d: np.ndarray


def _resolve_dataset_root(dataset_root: str | Path | None) -> Path:
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _resolve_annotations_root(dataset_root: str | Path | None, annotations_path: str | Path | None) -> Path:
    if annotations_path is not None:
        return Path(annotations_path)
    return _resolve_dataset_root(dataset_root)


def _load_json_array(path: Path, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    array = np.asarray(data, dtype=np.float32)
    if expected_shape is not None and array.shape[1:] != expected_shape:
        raise ValueError(f"Unexpected shape for {path.name}: {array.shape}, expected (*, {expected_shape})")
    return array


def load_annotation_samples(dataset_root, annotations_path=None):
    """Load FreiHAND training image paths together with `K` and `xyz` labels."""
    dataset_root = _resolve_dataset_root(dataset_root)
    annotations_root = _resolve_annotations_root(dataset_root, annotations_path)

    intrinsics = _load_json_array(annotations_root / "training_K.json", expected_shape=(3, 3))
    keypoints_3d = _load_json_array(annotations_root / "training_xyz.json", expected_shape=(21, 3))

    image_dir = dataset_root / "training" / "rgb"
    image_paths = [str(image_dir / f"{index:08d}.jpg") for index in range(len(keypoints_3d))]
    missing_paths = [path for path in image_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing training images, first missing path: {missing_paths[0]}")

    return [
        LandmarkSample(path, intrinsics[index], keypoints_3d[index])
        for index, path in enumerate(image_paths)
    ]


def load_image(path, image_size):
    """Load an RGB image from disk and resize it to the requested size."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return normalize_image(crop_and_resize_image(image, np.eye(3, dtype=np.float32), (0, 0, image.shape[1], image.shape[0]), image_size)[0])


def summarize_annotations(dataset_root, annotations_path=None):
    """Return a compact summary of FreiHAND training annotations."""
    annotations_root = _resolve_annotations_root(dataset_root, annotations_path)
    dataset_root = _resolve_dataset_root(dataset_root)

    intrinsics = _load_json_array(annotations_root / "training_K.json", expected_shape=(3, 3))
    keypoints_3d = _load_json_array(annotations_root / "training_xyz.json", expected_shape=(21, 3))
    mano = _load_json_array(annotations_root / "training_mano.json")
    sample_image = cv2.imread(str(dataset_root / "training" / "rgb" / "00000000.jpg"), cv2.IMREAD_COLOR)

    return {
        "num_samples": int(len(keypoints_3d)),
        "intrinsics_shape": tuple(int(dim) for dim in intrinsics.shape),
        "keypoints_shape": tuple(int(dim) for dim in keypoints_3d.shape),
        "mano_shape": tuple(int(dim) for dim in mano.shape),
        "image_shape": None if sample_image is None else tuple(int(dim) for dim in sample_image.shape),
    }


def _crop_and_prepare_sample(
    image_path: str,
    intrinsics: np.ndarray,
    keypoints_3d: np.ndarray,
    image_size: Sequence[int],
    training: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    uv = project_3D_to_2D(keypoints_3d, intrinsics)
    crop_box = compute_square_crop_box(uv, image.shape, padding=18.0)
    if training:
        crop_box = jitter_square_crop_box(crop_box, image.shape)

    cropped_image, updated_intrinsics = crop_and_resize_image(image, intrinsics, crop_box, image_size)
    return normalize_image(cropped_image), updated_intrinsics.astype(np.float32), keypoints_3d.astype(np.float32)


def _tf_parse_sample(
    image_path: tf.Tensor,
    intrinsics: tf.Tensor,
    keypoints_3d: tf.Tensor,
    image_size: tuple[int, int],
    training: bool,
) -> dict[str, tf.Tensor]:
    def _parse_numpy(path_bytes: bytes, K: np.ndarray, xyz: np.ndarray):
        return _crop_and_prepare_sample(
            image_path=path_bytes.decode("utf-8"),
            intrinsics=K,
            keypoints_3d=xyz,
            image_size=image_size,
            training=training,
        )

    image, updated_intrinsics, xyz = tf.numpy_function(
        _parse_numpy,
        [image_path, intrinsics, keypoints_3d],
        [tf.float32, tf.float32, tf.float32],
    )
    image.set_shape((image_size[0], image_size[1], 3))
    updated_intrinsics.set_shape((3, 3))
    xyz.set_shape((21, 3))

    return {
        "image": image,
        "K": updated_intrinsics,
        "keypoints": xyz,
        "path": image_path,
    }


def _apply_color_jitter(sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    image = sample["image"]
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    sample["image"] = tf.clip_by_value(image, 0.0, 1.0)
    return sample


def prepare_training_arrays(dataset_root, annotations_path=None, image_size=(224, 224), limit=None):
    """Load arrays required to build the TensorFlow dataset."""
    del image_size
    samples = load_annotation_samples(dataset_root, annotations_path)
    if limit is not None:
        samples = samples[: int(limit)]

    image_paths = np.asarray([sample.image_path for sample in samples], dtype=np.str_)
    intrinsics = np.stack([sample.intrinsics for sample in samples]).astype(np.float32)
    keypoints_3d = np.stack([sample.keypoints_3d for sample in samples]).astype(np.float32)
    return image_paths, intrinsics, keypoints_3d


def build_dataset(
    dataset_root: str | Path | None = None,
    annotations_path: str | Path | None = None,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    shuffle: bool = True,
    limit: int | None = None,
    training: bool = True,
) -> tf.data.Dataset:
    """Build a `tf.data.Dataset` for FreiHAND landmark training."""
    image_paths, intrinsics, keypoints_3d = prepare_training_arrays(
        dataset_root=dataset_root,
        annotations_path=annotations_path,
        image_size=image_size,
        limit=limit,
    )

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, intrinsics, keypoints_3d))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000), reshuffle_each_iteration=training)

    dataset = dataset.map(
        lambda path, K, xyz: _tf_parse_sample(path, K, xyz, image_size=image_size, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training:
        dataset = dataset.map(_apply_color_jitter, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_freihand_dataset(**kwargs: Any) -> tf.data.Dataset:
    """Backward-friendly alias for the main dataset builder."""
    return build_dataset(**kwargs)


if __name__ == "__main__":
    dataset = build_dataset(batch_size=16, training=True)
    batch = next(iter(dataset))
    print(batch["image"].shape)
    print(batch["keypoints"].shape)
    print(batch["K"].shape)