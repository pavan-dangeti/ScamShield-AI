"""Download the published int8 ONNX model and verify every file's SHA-256.

    python -m scripts.download_model

Repository, commit and checksums come from ``configs/model_hub.json``. Files that
are already present and correct are not downloaded again. Serve the model with
``pip install -r requirements-onnx.txt`` and ``SCAMSHIELD_MODEL=xlmr-aug-int8``.
"""

from __future__ import annotations

import os

from src.model_hub import MANIFEST, ensure_model, load_manifest


def main() -> None:
    manifest = load_manifest(MANIFEST)
    print(f"fetching {manifest['repo_id']} at {manifest['revision'][:12]} -> {manifest['local_dir']}")
    onnx_path = ensure_model(MANIFEST)
    for name in manifest["files"]:
        size = os.path.getsize(os.path.join(manifest["local_dir"], name)) / 1e6
        print(f"  verified {name} ({size:.1f} MB)")
    print(f"ready: {onnx_path}")


if __name__ == "__main__":
    main()
