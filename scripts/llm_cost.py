"""Cost per 1,000 messages for the prompted LLM arm.

Two currencies, both from measured quantities rather than guesses:

* **Local** - the model runs on the owner's own hardware, so the cost is wall
  time and electricity, not API spend.
* **API** - the same prompts sent to a hosted model. Pass the provider's blended
  price per million tokens and the measured token counts from a real run; this
  script only does arithmetic and never calls a service.

Token counts are measured with ``scripts/llm_cost.py --measure`` on this machine
(one representative message per shot setting, greedy decoding), and are recorded
in ``results/llm_cost.json`` next to the cost they produce.

Usage::

    python -m scripts.llm_cost --measure --shots 3
    python -m scripts.llm_cost --shots 3 --price-per-million 0.30 --watts 25
"""

from __future__ import annotations

import argparse
import json
import os
import time

RESULTS_DIR = "results"

# Measured on this repository's validation messages with Qwen2.5-7B-Instruct-4bit.
# Re-measure with --measure if you change the model or the prompt.
MEASURED = {
    0: {"input_tokens": 100, "output_tokens": 15, "seconds": 22.56},
    3: {"input_tokens": 440, "output_tokens": 13, "seconds": 3.61},
}


def measure(shots: int, model: str) -> dict:
    from src.data import load_split
    from src.models.llm_prompted import (
        GUIDELINES,
        SYSTEM_PROMPT,
        USER_TEMPLATE,
        MlxChat,
        few_shot_examples,
    )

    chat = MlxChat(model)
    train, val = load_split("train"), load_split("val")
    message = val[0]
    prefix = few_shot_examples(train, shots, 42) if shots else ""
    system = SYSTEM_PROMPT if not prefix else f"{SYSTEM_PROMPT}\n\n{GUIDELINES}\n\nExamples:\n{prefix}"
    user = USER_TEMPLATE.format(text=message["text"], language=message["language"], script=message["script"])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    input_tokens = len(chat.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    start = time.time()
    output = chat(system, user)
    seconds = time.time() - start
    return {
        "input_tokens": input_tokens,
        "output_tokens": len(chat.tokenizer.encode(output)),
        "seconds": round(seconds, 2),
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=3, choices=[0, 3])
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--price-per-million", type=float, default=0.30, help="blended USD per 1M tokens")
    parser.add_argument("--watts", type=float, default=25.0, help="measured draw while generating")
    parser.add_argument("--kwh-price", type=float, default=0.15, help="USD per kWh")
    parser.add_argument("--measure", action="store_true", help="re-measure token counts and wall time")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "llm_cost.json"))
    args = parser.parse_args()

    measured = measure(args.shots, args.model) if args.measure else dict(MEASURED[args.shots])
    per_1000_input = measured["input_tokens"] * 1000
    per_1000_output = measured["output_tokens"] * 1000
    per_1000_seconds = measured["seconds"] * 1000
    kwh = (args.watts / 1000.0) * (per_1000_seconds / 3600.0)
    api_usd = (per_1000_input + per_1000_output) / 1_000_000.0 * args.price_per_million

    report = {
        "shots": args.shots,
        "model": measured.get("model", args.model),
        "measured_per_message": measured,
        "per_1000_messages": {
            "input_tokens": per_1000_input,
            "output_tokens": per_1000_output,
            "total_tokens": per_1000_input + per_1000_output,
        },
        "local": {
            "usd_per_1000_messages": round(kwh * args.kwh_price, 4),
            "kwh_per_1000_messages": round(kwh, 5),
            "minutes_per_1000_messages": round(per_1000_seconds / 60, 1),
            "assumptions": {
                "watts_while_generating": args.watts,
                "usd_per_kwh": args.kwh_price,
                "note": "one machine, no API spend; the currency is time and electricity",
            },
        },
        "api": {
            "usd_per_1000_messages": round(api_usd, 4),
            "price_per_million_usd": args.price_per_million,
            "note": "arithmetic on the measured token counts; this script makes no API call",
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
