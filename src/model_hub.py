"""Fetch the published int8 ONNX model from Hugging Face Hub, verified by SHA-256.

The manifest (``configs/model_hub.json``) pins the repository, the exact commit
and a checksum for every file, so a download is reproducible and a tampered or
truncated file is rejected rather than loaded. Files go to the directory the
model registry already reads, so nothing else needs to know where they came from.

    python -m scripts.download_model
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import requests

MANIFEST = "configs/model_hub.json"
log = logging.getLogger("scamshield.model_hub")


class ModelHubError(RuntimeError):
    pass


def load_manifest(path: str = MANIFEST) -> dict:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("repo_id") or not manifest.get("revision"):
        raise ModelHubError(f"{path} has no repo_id/revision yet; the model has not been published")
    return manifest


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: str, expected: str, attempts: int = 4) -> None:
    # Download into ``<target>.part``, resuming with an HTTP range request when a
    # large file is cut off mid-transfer, then check the whole file and rename it.
    # A failed or corrupt download never replaces a good file or stays on disk.
    partial = target + ".part"
    name = os.path.basename(target)
    try:
        for attempt in range(1, attempts + 1):
            have = os.path.getsize(partial) if os.path.exists(partial) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                    response.raise_for_status()
                    # 206 continues the partial file; a 200 means the server sent it all again.
                    mode = "ab" if have and response.status_code == 206 else "wb"
                    with open(partial, mode) as handle:
                        for block in response.iter_content(chunk_size=1 << 20):
                            handle.write(block)
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as error:
                if attempt == attempts:
                    raise ModelHubError(f"{name}: download failed after {attempts} attempts: {error}") from error
                log.warning("download interrupted, resuming", extra={"file": name, "attempt": attempt})
                time.sleep(attempt)
        actual = sha256(partial)
        if actual != expected:
            raise ModelHubError(f"{name}: checksum {actual} does not match {expected}")
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def ensure_model(manifest_path: str = MANIFEST) -> str:
    """Make sure every model file is present and verified; return the ONNX path."""
    manifest = load_manifest(manifest_path)
    directory = manifest["local_dir"]
    os.makedirs(directory, exist_ok=True)
    base = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}"
    for name, expected in manifest["files"].items():
        target = os.path.join(directory, name)
        if os.path.exists(target) and sha256(target) == expected:
            continue
        log.info("downloading model file", extra={"file": name, "repo": manifest["repo_id"]})
        _download(f"{base}/{name}", target, expected)
    return os.path.join(directory, manifest["onnx_file"])
