"""Downloading the published model: checksums are enforced, nothing touches the network."""

from __future__ import annotations

import hashlib
import json

import pytest

from src import model_hub

FILES = {"model.int8.onnx": b"onnx-bytes", "tokenizer.json": b"{}"}


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, **_kwargs):
        yield self.body


def write_manifest(tmp_path, files=FILES, repo_id="someone/scamshield", revision="abc123"):
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": str(tmp_path / "model"),
        "onnx_file": "model.int8.onnx",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }
    path = tmp_path / "model_hub.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def serve(monkeypatch, bodies: dict[str, bytes]) -> list[str]:
    requested: list[str] = []

    def fake_get(url, **_kwargs):
        requested.append(url)
        return FakeResponse(bodies[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(model_hub.requests, "get", fake_get)
    return requested


def test_downloads_pinned_revision_and_verifies(tmp_path, monkeypatch):
    requested = serve(monkeypatch, FILES)
    onnx_path = model_hub.ensure_model(write_manifest(tmp_path))
    assert onnx_path.endswith("model.int8.onnx")
    assert (tmp_path / "model" / "tokenizer.json").read_bytes() == b"{}"
    assert all("/resolve/abc123/" in url for url in requested)


def test_corrupt_download_is_rejected_and_not_left_behind(tmp_path, monkeypatch):
    serve(monkeypatch, {**FILES, "model.int8.onnx": b"tampered"})
    with pytest.raises(model_hub.ModelHubError, match="checksum"):
        model_hub.ensure_model(write_manifest(tmp_path))
    assert not (tmp_path / "model" / "model.int8.onnx").exists()
    assert list((tmp_path / "model").iterdir()) == [], "no partial or temporary files"


def test_verified_files_are_not_downloaded_again(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    serve(monkeypatch, FILES)
    model_hub.ensure_model(manifest)
    requested = serve(monkeypatch, FILES)
    model_hub.ensure_model(manifest)
    assert requested == []


def test_unpublished_manifest_fails_clearly(tmp_path):
    with pytest.raises(model_hub.ModelHubError, match="not been published"):
        model_hub.ensure_model(write_manifest(tmp_path, repo_id=None, revision=None))
