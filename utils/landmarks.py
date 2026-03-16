"""Landmark utilities shared by training and visualization code."""

from __future__ import annotations

import numpy as np

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def as_pairs(landmarks):
    """Convert a flat landmark vector into an `(N, C)` array."""
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim == 1:
        if array.size % 2 == 0:
            return array.reshape(-1, 2)
        if array.size % 3 == 0:
            return array.reshape(-1, 3)
        raise ValueError(f"Cannot infer channel count from shape {array.shape}")
    return array


def normalize_relative_to_wrist(landmarks):
    """Express landmarks relative to the wrist joint at index 0."""
    array = as_pairs(landmarks)
    return array - array[0:1]


def finger_is_open(points, tip_index, pip_index):
    """Use vertical ordering as a simple open-finger heuristic."""
    array = as_pairs(points)
    return bool(array[tip_index, 1] < array[pip_index, 1])


def classify_simple_gesture(landmarks):
    """Classify a coarse hand pose from 2D or 3D landmarks."""
    points = as_pairs(landmarks)
    open_map = {
        "thumb": points[4, 0] > points[3, 0],
        "index": finger_is_open(points, 8, 6),
        "middle": finger_is_open(points, 12, 10),
        "ring": finger_is_open(points, 16, 14),
        "pinky": finger_is_open(points, 20, 18),
    }
    open_count = sum(bool(value) for value in open_map.values())

    if open_count == 0:
        return "fist"
    if open_count == 5:
        return "open_palm"
    if open_map["index"] and open_map["middle"] and open_count == 2:
        return "peace"
    if open_map["thumb"] and open_count == 1:
        return "thumbs_up"
    return "unknown"