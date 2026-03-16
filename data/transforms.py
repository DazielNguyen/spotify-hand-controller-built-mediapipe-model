"""Image and geometry transforms used by the FreiHAND data pipeline."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _coerce_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("size must be an int or a sequence of length 2")
    return int(size[0]), int(size[1])


def resize_image(image, size):
    """Resize an image to `(height, width)`."""
    target_height, target_width = _coerce_size(size)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def normalize_image(image):
    """Convert image pixels to float32 in the [0, 1] range."""
    image = np.asarray(image, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image /= 255.0
    return image


def flatten_landmarks(landmarks):
    """Flatten landmark coordinates into a single vector."""
    return np.asarray(landmarks, dtype=np.float32).reshape(-1)


def project_3D_to_2D(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-space 3D hand joints into pixel coordinates."""
    xyz = np.asarray(xyz, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")

    z = np.clip(xyz[:, 2:3], 1e-6, None)
    normalized_xyz = xyz / z
    uvw = (K @ normalized_xyz.T).T
    return uvw[:, :2]


def compute_square_crop_box(
    uv: np.ndarray,
    image_shape: Sequence[int],
    padding: float = 16.0,
) -> tuple[int, int, int, int]:
    """Compute a square crop around projected keypoints."""
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    uv = np.asarray(uv, dtype=np.float32)
    finite_mask = np.isfinite(uv).all(axis=1)

    if not np.any(finite_mask):
        return 0, 0, image_width, image_height

    valid_uv = uv[finite_mask]
    min_xy = valid_uv.min(axis=0) - padding
    max_xy = valid_uv.max(axis=0) + padding
    center_xy = (min_xy + max_xy) / 2.0
    side = float(np.max(max_xy - min_xy))
    side = float(np.clip(max(side, 32.0), 32.0, min(image_width, image_height)))

    x0 = float(np.clip(center_xy[0] - side / 2.0, 0.0, image_width - side))
    y0 = float(np.clip(center_xy[1] - side / 2.0, 0.0, image_height - side))
    x1 = x0 + side
    y1 = y0 + side
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def jitter_square_crop_box(
    crop_box: Sequence[int],
    image_shape: Sequence[int],
    scale_range: tuple[float, float] = (0.95, 1.05),
    shift_fraction: float = 0.05,
) -> tuple[int, int, int, int]:
    """Randomly perturb a square crop while keeping it inside the image."""
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    side = min(x1 - x0, y1 - y0)
    max_side = float(min(image_width, image_height))

    scale = np.random.uniform(scale_range[0], scale_range[1])
    side = float(np.clip(side * scale, 32.0, max_side))

    center_x = (x0 + x1) / 2.0 + np.random.uniform(-shift_fraction, shift_fraction) * side
    center_y = (y0 + y1) / 2.0 + np.random.uniform(-shift_fraction, shift_fraction) * side
    x0 = float(np.clip(center_x - side / 2.0, 0.0, image_width - side))
    y0 = float(np.clip(center_y - side / 2.0, 0.0, image_height - side))
    x1 = x0 + side
    y1 = y0 + side
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def crop_and_resize_image(
    image: np.ndarray,
    K: np.ndarray,
    crop_box: Sequence[int],
    output_size: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image and update the intrinsic matrix for the crop + resize."""
    x0, y0, x1, y1 = [int(value) for value in crop_box]
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError(f"Invalid crop box {crop_box} for image shape {image.shape}")

    output_height, output_width = _coerce_size(output_size)
    resized = resize_image(cropped, (output_height, output_width))

    crop_height = max(y1 - y0, 1)
    crop_width = max(x1 - x0, 1)
    scale_x = output_width / float(crop_width)
    scale_y = output_height / float(crop_height)

    updated_K = np.asarray(K, dtype=np.float32).copy()
    updated_K[0, 0] *= scale_x
    updated_K[1, 1] *= scale_y
    updated_K[0, 2] = (updated_K[0, 2] - x0) * scale_x
    updated_K[1, 2] = (updated_K[1, 2] - y0) * scale_y
    return resized, updated_K
