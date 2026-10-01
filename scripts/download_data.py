"""Download the external ScamShield data sources.

Every source is pinned by URL and (after the first run) by SHA-256 in
``data/raw_manifest.json``.  Raw files are never committed: they are
re-downloaded by this script.  Licences are recorded both here and in
``docs/data_sources.md``; a source with an unverified licence must not be added
to this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any

import requests

RAW_DIR = "data/raw"
MANIFEST_PATH = "data/raw_manifest.json"

SOURCES: dict[str, dict[str, Any]] = {
    "uci_sms_spam": {
        "homepage": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "licence": "CC-BY-4.0",
        "provenance": "real",
        "citation": (
            "Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). "
            "Contributions to the Study of SMS Spam Filtering: New Collection "
            "and Results. Proceedings of the 2011 ACM Symposium on Document Engineering."
        ),
        "files": ["SMSSpamCollection"],
    },
    "indian_scam_sms_synthetic": {
        "homepage": "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited",
        "url": (
            "https://huggingface.co/datasets/Ridham115/"
            "indian-scam-sms-synthetic-audited/resolve/main/data/synthetic_audited.csv"
        ),
        "licence": "CC-BY-4.0",
        "provenance": "synthetic",
        "citation": "Ridham115. Indian Scam SMS (synthetic, audited). Hugging Face, 2025.",
        "files": ["synthetic_audited.csv"],
    },
    "bengali_sms_smishing": {
        "homepage": "https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset",
        "url": (
            "https://huggingface.co/datasets/shariul-islam/"
            "bengali-sms-smishing-dataset/resolve/main/data/{split}-00000-of-00001.parquet"
        ),
        "licence": "MIT",
        "provenance": "unverified",
        "citation": (
            "Islam, S. (2025). SmishDetect-LLM: Multilingual SMS Phishing "
            "Detection Using Large Language Models. Murdoch University."
        ),
        "files": [
            "train-00000-of-00001.parquet",
            "validation-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
        ],
    },
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def download(url: str, destination: str) -> None:
    print(f"  GET {url}")
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)


def fetch_source(name: str, force: bool, manifest: dict) -> None:
    spec = SOURCES[name]
    target_dir = os.path.join(RAW_DIR, name)
    os.makedirs(target_dir, exist_ok=True)
    entry = manifest.setdefault(name, {"licence": spec["licence"], "files": {}})
    entry["url"] = spec["url"]
    entry["homepage"] = spec["homepage"]
    entry["licence"] = spec["licence"]
    entry["provenance"] = spec["provenance"]

    is_zip = spec["url"].endswith(".zip")
    for filename in spec["files"]:
        path = os.path.join(target_dir, filename)
        expected = entry["files"].get(filename, {}).get("sha256")
        if os.path.exists(path) and not force:
            actual = sha256(path)
            if expected and actual != expected:
                raise SystemExit(
                    f"{path} hash mismatch: expected {expected}, got {actual}. "
                    "Delete the file or re-run with --force."
                )
            if not expected:
                entry["files"][filename] = {"sha256": actual}
                print(f"  pinned {filename} sha256={actual[:12]}...")
            continue
        url = spec["url"].format(split=filename.split("-")[0]) if "{split}" in spec["url"] else spec["url"]
        if is_zip:
            zip_path = path + ".zip"
            download(spec["url"], zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(target_dir)
            os.remove(zip_path)
        else:
            download(url, path)
        digest = sha256(path)
        entry["files"][filename] = {"sha256": digest}
        entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        print(f"  downloaded {filename} sha256={digest[:12]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="all", choices=["all", *SOURCES])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    manifest = load_manifest()
    names = list(SOURCES) if args.source == "all" else [args.source]
    for name in names:
        print(f"[{name}]")
        fetch_source(name, args.force, manifest)
    save_manifest(manifest)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
