"""Explore FreiHAND annotations and generate a projection sanity-check image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import DEFAULT_DATASET_ROOT, summarize_annotations
from data.transforms import project_3D_to_2D
from utils.landmarks import HAND_CONNECTIONS


def _load_json(path: Path):
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _draw_projection(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for start_index, end_index in HAND_CONNECTIONS:
        start_point = tuple(np.round(uv[start_index]).astype(int))
        end_point = tuple(np.round(uv[end_index]).astype(int))
        cv2.line(canvas, start_point, end_point, color=(0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

    for point in uv:
        center = tuple(np.round(point).astype(int))
        cv2.circle(canvas, center, radius=4, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def _resolve_sample_index(index: int | None, image_name: str | None, num_samples: int) -> int:
    if index is not None and image_name is not None:
        raise ValueError("Use either --index or --image-name, not both")

    if image_name is not None:
        stem = Path(image_name).stem
        if not stem.isdigit():
            raise ValueError(f"image name must contain a numeric stem, got {image_name}")
        index = int(stem)

    if index is None:
        index = int(np.random.randint(0, num_samples))

    if index < 0 or index >= num_samples:
        raise IndexError(f"Sample index {index} is out of range [0, {num_samples - 1}]")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", type=int, default=None, help="Sample index in FreiHAND training split")
    parser.add_argument("--image-name", type=str, default=None, help="Image file name such as 00000005.jpg")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the rendered projection")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    intrinsics = np.asarray(_load_json(dataset_root / "training_K.json"), dtype=np.float32)
    keypoints_3d = np.asarray(_load_json(dataset_root / "training_xyz.json"), dtype=np.float32)
    mano = np.asarray(_load_json(dataset_root / "training_mano.json"), dtype=np.float32)
    sample_index = _resolve_sample_index(args.index, args.image_name, len(keypoints_3d))
    image_name = f"{sample_index:08d}.jpg"
    output_path = args.output or (REPO_ROOT / "outputs" / f"test_projection_{sample_index:08d}.jpg")

    print("FreiHAND summary:", summarize_annotations(dataset_root))
    print("selected sample index:", sample_index)
    print("selected image:", image_name)
    print(f"xyz[{sample_index}, 0] (wrist):", keypoints_3d[sample_index, 0])
    print(f"K[{sample_index}]:\n", intrinsics[sample_index])
    print("keypoints tensor shape:", keypoints_3d.shape)
    print("training_mano shape:", mano.shape)

    image_path = dataset_root / "training" / "rgb" / image_name
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    uv = project_3D_to_2D(keypoints_3d[sample_index], intrinsics[sample_index])
    rendered = _draw_projection(image_bgr, uv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)
    print(f"Saved projection visualization to: {output_path}")


if __name__ == "__main__":
    main()