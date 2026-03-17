"""Render MANO mesh to OBJ file.

This script renders the MANO hand mesh to an OBJ file for visualization.
"""

from pathlib import Path

import numpy as np
import torch
import trimesh

# Fix for Python 3.11+
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MANO_MODEL_PATH = REPO_ROOT / "models" / "mano" / "MANO_RIGHT.pkl"


def load_mano_model():
    """Load MANO model."""
    try:
        from smplx import MANO
    except ImportError:
        raise ImportError("smplx is required. Install with: pip install smplx")

    if not MANO_MODEL_PATH.exists():
        raise FileNotFoundError(f"MANO model not found at {MANO_MODEL_PATH}")

    mano_model = MANO(
        str(MANO_MODEL_PATH),
        model_type="mano",
        use_pca=False,
        is_rhand=True,
        num_pca_comps=45,
    )

    return mano_model


def render_mano_mesh():
    """Render MANO mesh to OBJ file."""
    print("=" * 60)
    print("Rendering MANO Mesh to OBJ")
    print("=" * 60)

    # Load MANO model
    print("\n[1] Loading MANO model...")
    mano_model = load_mano_model()
    print("    MANO model loaded successfully")

    # Create zero inputs
    print("\n[2] Creating zero inputs...")
    batch_size = 1
    pose = torch.zeros(batch_size, 45, dtype=torch.float32)
    betas = torch.zeros(batch_size, 10, dtype=torch.float32)
    global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)

    # Forward pass
    print("\n[3] Running MANO forward pass...")
    with torch.no_grad():
        output = mano_model(
            betas=betas,
            hand_pose=pose,
            global_orient=global_orient,
        )

    # Get vertices [1, 778, 3]
    vertices = output.vertices[0].numpy()  # shape: [778, 3]
    print(f"    vertices shape: {vertices.shape}")

    # Get faces from MANO model
    # faces are stored in the model's 'faces' property
    faces = mano_model.faces  # shape: [1538, 3]
    print(f"    faces shape: {faces.shape}")

    # Create trimesh object
    print("\n[4] Creating trimesh mesh...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save to OBJ file
    output_path = OUTPUT_DIR / "hand.obj"
    print(f"\n[5] Saving mesh to {output_path}...")
    mesh.export(str(output_path))

    print("\n" + "=" * 60)
    print(f"SUCCESS! Mesh saved to: {output_path}")
    print("Open this file with a 3D viewer (Preview on Mac, 3D Viewer on Windows)")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    try:
        render_mano_mesh()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
