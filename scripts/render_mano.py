"""Render MANO mesh to .obj file using trimesh.

This script renders the MANO hand mesh to a 3D .obj file
that can be viewed in any 3D viewer (Preview on Mac, 3D Viewer on Windows).

Expected output:
- outputs/hand.obj - 3D mesh file
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

# Fix for Python 3.11+: inspect.getargspec was removed
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MANO model path
MANO_MODEL_PATH = REPO_ROOT / "models" / "mano" / "MANO_RIGHT.pkl"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_OBJ = OUTPUT_DIR / "hand.obj"


def load_mano_model():
    """Load MANO model using smplx."""
    try:
        from smplx import MANO
    except ImportError:
        raise ImportError("smplx is required. Install with: pip install smplx")

    if not MANO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MANO model not found at {MANO_MODEL_PATH}. "
            "Download from: https://mano.is.tue.mpg.de/"
        )

    # Load MANO model (right hand)
    mano_model = MANO(
        str(MANO_MODEL_PATH),
        model_type="mano",
        use_pca=False,
        is_rhand=True,
        num_pca_comps=45,
    )

    return mano_model


def get_mano_faces(mano_model):
    """Extract faces from MANO model."""
    # MANO model has faces stored in the model
    # For smplx MANO, faces are typically available via .faces attribute
    # or we can generate them from the mesh topology

    # Try to get faces from the model
    if hasattr(mano_model, 'faces'):
        faces = mano_model.faces
    elif hasattr(mano_model, 'face_indices'):
        faces = mano_model.face_indices
    else:
        # Alternative: load from the pickle file directly
        import pickle
        with open(MANO_MODEL_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            if 'f' in data:
                faces = data['f']
            elif 'faces' in data:
                faces = data['faces']
            else:
                raise ValueError("Cannot find faces in MANO model file")

    return faces


def render_mano_mesh():
    """Render MANO mesh to .obj file."""
    print("=" * 60)
    print("Rendering MANO Mesh to .obj")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1] Output directory: {OUTPUT_DIR}")

    # Load MANO model
    print("\n[2] Loading MANO model...")
    mano_model = load_mano_model()
    print(f"    MANO model loaded from: {MANO_MODEL_PATH}")

    # Get faces from MANO model
    print("\n[3] Extracting faces from MANO model...")
    faces = get_mano_faces(mano_model)
    print(f"    Faces shape: {faces.shape}")

    # Create zero inputs
    print("\n[4] Creating zero inputs...")
    batch_size = 1
    pose = torch.zeros(batch_size, 45, dtype=torch.float32)
    betas = torch.zeros(batch_size, 10, dtype=torch.float32)
    global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)

    # Forward pass
    print("\n[5] Running MANO forward pass...")
    with torch.no_grad():
        output = mano_model(
            betas=betas,
            hand_pose=pose,
            global_orient=global_orient,
        )

    # Get vertices
    vertices = output.vertices.numpy()  # shape: [1, 778, 3]
    vertices = vertices[0]  # Remove batch dimension: [778, 3]

    print(f"\n[6] Vertices shape: {vertices.shape}")
    print(f"    Faces shape: {faces.shape}")

    # Create trimesh object
    print("\n[7] Creating trimesh mesh...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save to .obj file
    print(f"\n[8] Saving to {OUTPUT_OBJ}...")
    mesh.export(str(OUTPUT_OBJ))
    print(f"    ✓ Saved successfully!")

    # Also export as GLB for better compatibility
    output_glb = OUTPUT_DIR / "hand.glb"
    mesh.export(str(output_glb))
    print(f"    ✓ Also saved as GLB: {output_glb}")

    # Print summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.2 COMPLETED!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_OBJ}")
    print(f"  - {output_glb}")
    print(f"\nOpen these files in a 3D viewer to see the hand mesh.")
    print(f"(Preview on Mac / 3D Viewer on Windows)")


if __name__ == "__main__":
    try:
        render_mano_mesh()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
