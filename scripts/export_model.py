"""Export a fine-tuned encoder to ONNX and optionally quantise it.

The production decision in Phase 5 is between a 10 MB TF-IDF model and a 1.1 GB
fp32 encoder. Quantisation is what decides whether the encoder is shippable, so it
is measured rather than assumed: the script exports, quantises to int8, and reports
size, CPU latency and the accuracy cost on the validation split.

Usage::

    python -m scripts.export_model --model xlm-roberta-base-aug --quantize int8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
from sklearn.metrics import f1_score

RESULTS_DIR = "results"


def export_onnx(model_name: str, directory: str, max_length: int) -> str:
    import torch
    from transformers import AutoTokenizer

    from scripts.evaluate_models import MODEL_REGISTRY
    from src.models.transformer import ScamEncoderModel
    from src.models.transformer_infer import _model_dir

    hf_name = MODEL_REGISTRY.get(model_name) or model_name
    checkpoint = _model_dir(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    model = ScamEncoderModel(config["model_name"], dropout=0.0)
    model.load_state_dict(torch.load(os.path.join(checkpoint, "model.pt"), map_location="cpu"))
    model.eval()

    encoded = tokenizer(["dummy message"], return_tensors="pt", truncation=True, max_length=max_length)
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    args: tuple = (encoded["input_ids"], encoded["attention_mask"])
    kwargs: dict = {}
    if "token_type_ids" in encoded:
        args, kwargs = args + (encoded["token_type_ids"],), {"token_type_ids": encoded["token_type_ids"]}

    out_dir = os.path.join("models", "onnx", model_name)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    # The legacy TorchScript exporter is used deliberately: the torch.export-based
    # exporter emits a graph that onnxruntime's dynamic quantiser rejects (it wants
    # to overwrite a declared dimension of 1 with an inferred 768 during shape
    # inference). The legacy graph quantises cleanly, and the accuracy of both the
    # fp32 and the int8 copy is measured below rather than assumed.
    torch.onnx.export(
        model,
        args,
        onnx_path,
        input_names=input_names + (["token_type_ids"] if kwargs else []),
        output_names=["binary_logits", "tactic_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
        **kwargs,
    )
    tokenizer.save_pretrained(out_dir)
    return onnx_path


def quantise(onnx_path: str) -> str:
    """Dynamic int8 quantisation (weights), the standard path for a CPU encoder."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = onnx_path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(onnx_path, target, weight_type=QuantType.QInt8)
    return target


class OnnxPredictor:
    """Minimal ONNX Runtime scorer with the same signature as the eval harness expects."""

    def __init__(self, path: str, tokenizer_dir: str, max_length: int = 160) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = max_length
        self.input_names = {i.name for i in self.session.get_inputs()}

    def scores(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="np"
        )
        feed = {name: encoded[name] for name in self.input_names if name in encoded}
        binary_logits, _ = self.session.run(None, feed)
        return 1.0 / (1.0 + np.exp(-binary_logits))


def time_onnx(predictor: OnnxPredictor, texts: list[str], warmup: int = 5) -> dict:
    for text in texts[:warmup]:
        predictor.scores([text])
    timings = []
    for text in texts:
        start = time.perf_counter()
        predictor.scores([text])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xlm-roberta-base-aug")
    parser.add_argument("--quantize", action="store_true", help="also produce an int8 dynamic-quantised copy")
    parser.add_argument("--n", type=int, default=200, help="messages used for the latency measurement")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "export.json"))
    args = parser.parse_args()

    from src.data import load_split
    from src.models.transformer_infer import _model_dir

    rows = load_split("val")
    texts = [row["text"] for row in rows[: args.n]]
    all_texts = [row["text"] for row in rows]
    labels = np.array([1 if row["label"] == "scam" else 0 for row in rows])
    checkpoint = _model_dir(args.model if os.path.exists(_model_dir(args.model)) else args.model.replace("-aug", ""))
    max_length = json.load(open(os.path.join(checkpoint, "config.json"), encoding="utf-8")).get("max_length", 160)

    onnx_path = export_onnx(args.model, checkpoint, max_length)
    out_dir = os.path.dirname(onnx_path)
    report = {
        "model": args.model,
        "onnx_mb": round(os.path.getsize(onnx_path) / 1e6, 1),
        "fp32_torch_mb": round(
            sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(checkpoint) for f in files) / 1e6, 1
        ),
        # Accuracy is measured on the whole validation split, not the latency sample:
        # the first N rows are a single language cell and would flatter the model.
        "accuracy_split": f"val ({len(rows)} rows)",
        "latency_sample": args.n,
    }
    predictor = OnnxPredictor(onnx_path, out_dir, max_length)
    report["fp32_onnx_latency"] = time_onnx(predictor, texts)
    report["fp32_onnx_f1_full_val"] = float(
        f1_score(labels, (predictor.scores(all_texts) >= 0.5).astype(int), zero_division=0)
    )

    if args.quantize:
        int8_path = quantise(onnx_path)
        int8 = OnnxPredictor(int8_path, out_dir, max_length)
        report["int8_mb"] = round(os.path.getsize(int8_path) / 1e6, 1)
        report["int8_latency"] = time_onnx(int8, texts)
        report["int8_f1_full_val"] = float(
            f1_score(labels, (int8.scores(all_texts) >= 0.5).astype(int), zero_division=0)
        )
        report["int8_path"] = int8_path

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
