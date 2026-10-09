"""Publish the int8 ONNX model to Hugging Face Hub as a public model repository.

Run in your own terminal with a write token exported (it is read from the
environment, never prompted for or printed):

    export HF_TOKEN=...            # https://huggingface.co/settings/tokens, "Write"
    uv run --with huggingface_hub python -m scripts.publish_model

Uploads exactly the files the app needs, the model card (docs/hub_model_card.md
as README.md) and a SHA256SUMS file, then records the repository and the commit
it created in configs/model_hub.json so `scripts/download_model.py` and the app
fetch that exact revision. Commit that file afterwards.

After retraining (for example with the Colab notebook), the new files cannot match the
published checksums. Pass ``--new-artefact`` to record the new checksums and publish them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

from src.model_hub import MANIFEST, sha256

MODEL_CARD = "docs/hub_model_card.md"


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-name", default="scamshield-xlmr-int8")
    parser.add_argument("--new-artefact", action="store_true", help="publish retrained files with new checksums")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set. Export a write token in this terminal and run again.")
    with open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)

    # Refuse to publish anything that differs from what the manifest promises.
    for name, expected in manifest["files"].items():
        path = os.path.join(manifest["local_dir"], name)
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; regenerate it first (see the README)")
        actual = sha256(path)
        if actual != expected:
            if not args.new_artefact:
                raise SystemExit(f"{path} does not match {MANIFEST}; pass --new-artefact if it was retrained")
            manifest["files"][name] = actual

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.repo_name}"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        for name in manifest["files"]:
            shutil.copy2(os.path.join(manifest["local_dir"], name), staging)
        rows = ["| File | Size | SHA-256 |", "|---|---|---|"]
        for name, digest in manifest["files"].items():
            size = os.path.getsize(os.path.join(manifest["local_dir"], name))
            shown = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.1f} KB"
            rows.append(f"| `{name}` | {shown} | `{digest}` |")
        with open(MODEL_CARD, encoding="utf-8") as card:
            readme = card.read().replace("REPO_ID", repo_id).replace("FILES_TABLE", "\n".join(rows))
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as handle:
            handle.write(readme)
        with open(os.path.join(staging, "SHA256SUMS"), "w", encoding="utf-8") as sums:
            for name, digest in manifest["files"].items():
                sums.write(f"{digest}  {name}\n")
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=staging,
            commit_message="Publish ScamShield int8 ONNX model",
        )

    manifest["repo_id"] = repo_id
    manifest["revision"] = commit.oid
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"published https://huggingface.co/{repo_id} at {commit.oid}")
    print(f"updated {MANIFEST}; commit it so downloads use this revision")


if __name__ == "__main__":
    main()
