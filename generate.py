"""
MiniTruth-8M text generation.

Generates text from a trained checkpoint and records
the model probability of each generated token.

Important:
    Token probability is NOT the same as factual truth probability.
    We call these measurements "model confidence".
"""

import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from model import MiniTruthLM
from config import ModelConfig


# ============================================================
# Configuration
# ============================================================

DEFAULT_CHECKPOINT = "checkpoints/final.pt"
DEFAULT_TOKENIZER = "data/tokenizer.json"

DEVICE = "cpu"

DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9

MIN_PROBABILITY = 1e-12


# ============================================================
# Sampling utilities
# ============================================================

def top_k_filter(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))

    values, _ = torch.topk(logits, top_k)

    threshold = values[..., -1, None]

    return torch.where(
        logits < threshold,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def top_p_filter(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    remove = cumulative_probs > top_p

    # Keep at least one token.
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf"),
    )

    filtered_logits = torch.full_like(logits, float("-inf"))

    filtered_logits.scatter_(
        -1,
        sorted_indices,
        sorted_logits,
    )

    return filtered_logits


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path, tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = ByteLevelDecoder()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MiniTruthLM(ModelConfig())

    # Support both our checkpoint format and plain state_dict.
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        step = checkpoint.get("step", None)
    else:
        model.load_state_dict(checkpoint)
        step = None

    model.to(DEVICE)
    model.eval()

    return model, tokenizer, step


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.0,
):
    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated_ids = input_ids.clone()

    token_records = []

    for _ in range(max_new_tokens):

        # Only feed the model's context window.
        context = generated_ids[:, -ModelConfig().block_size:]

        logits, _ = model(context)

        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # RAW MODEL DISTRIBUTION
        #
        # This is what we use for confidence.
        # ----------------------------------------------------

        raw_probs = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # ----------------------------------------------------
        # Repetition penalty
        # ----------------------------------------------------

        if repetition_penalty != 1.0:

            seen_tokens = set(
                generated_ids[0].tolist()
            )

            for token_id in seen_tokens:

                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

        # ----------------------------------------------------
        # Sampling distribution
        # ----------------------------------------------------

        sampling_logits = next_token_logits.clone()

        if temperature <= 0:
            temperature = 1.0

        sampling_logits /= temperature

        sampling_logits = top_k_filter(
            sampling_logits,
            top_k,
        )

        sampling_logits = top_p_filter(
            sampling_logits,
            top_p,
        )

        sampling_probs = torch.softmax(
            sampling_logits,
            dim=-1,
        )

        # ----------------------------------------------------
        # Select next token
        # ----------------------------------------------------

        if do_sample:
            next_token = torch.multinomial(
                sampling_probs,
                num_samples=1,
            )
        else:
            next_token = torch.argmax(
                raw_probs,
                dim=-1,
                keepdim=True,
            )

        token_id = next_token.item()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Record the ORIGINAL model probability of the
        # selected token, NOT the sampling probability.
        # ----------------------------------------------------

        raw_probability = raw_probs[
            0,
            token_id,
        ].item()

        raw_probability = max(
            raw_probability,
            MIN_PROBABILITY,
        )

        log_probability = math.log(
            raw_probability
        )

        token_text = tokenizer.decode(
            [token_id]
        )

        token_records.append(
            {
                "token_id": token_id,
                "token": token_text,
                "probability": raw_probability,
                "log_probability": log_probability,
            }
        )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token,
            ],
            dim=1,
        )

        # Stop at EOS.
        eos_id = tokenizer.token_to_id("<eos>")

        if eos_id is not None and token_id == eos_id:
            break

    # Decode the complete generated sequence.
    full_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    # Decode only generated continuation.
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1]:].tolist()
    )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "token_records": token_records,
    }


# ============================================================
# Confidence statistics
# ============================================================

def calculate_confidence(token_records):

    if not token_records:
        return {
            "num_generated_tokens": 0,
            "arithmetic_mean_probability": None,
            "geometric_mean_probability": None,
            "mean_log_probability": None,
            "minimum_probability": None,
            "maximum_probability": None,
            "median_probability": None,
        }

    probabilities = [
        x["probability"]
        for x in token_records
    ]

    log_probabilities = [
        x["log_probability"]
        for x in token_records
    ]

    arithmetic_mean = sum(probabilities) / len(
        probabilities
    )

    mean_log_probability = sum(
        log_probabilities
    ) / len(log_probabilities)

    geometric_mean = math.exp(
        mean_log_probability
    )

    sorted_probabilities = sorted(probabilities)

    middle = len(sorted_probabilities) // 2

    if len(sorted_probabilities) % 2 == 0:
        median_probability = (
            sorted_probabilities[middle - 1]
            + sorted_probabilities[middle]
        ) / 2
    else:
        median_probability = sorted_probabilities[middle]

    return {
        "num_generated_tokens": len(probabilities),

        "arithmetic_mean_probability":
            arithmetic_mean,

        "geometric_mean_probability":
            geometric_mean,

        "mean_log_probability":
            mean_log_probability,

        "minimum_probability":
            min(probabilities),

        "maximum_probability":
            max(probabilities),

        "median_probability":
            median_probability,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("Loading model...")

    model, tokenizer, checkpoint_step = load_model(
        args.checkpoint,
        args.tokenizer,
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    if checkpoint_step is not None:
        print(
            f"Training step: {checkpoint_step}"
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        repetition_penalty=args.repetition_penalty,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        result["token_records"]
    )

    result["confidence"] = confidence

    result["generation_settings"] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "repetition_penalty":
            args.repetition_penalty,
    }

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROMPT")
    print("=" * 70)

    print(args.prompt)

    print()
    print("=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)

    print(result["generated_text"])

    print()
    print("=" * 70)
    print("MODEL CONFIDENCE")
    print("=" * 70)

    print(
        f"Generated tokens: "
        f"{confidence['num_generated_tokens']}"
    )

    print(
        f"Arithmetic mean: "
        f"{confidence['arithmetic_mean_probability']:.6f}"
    )

    print(
        f"Geometric mean:  "
        f"{confidence['geometric_mean_probability']:.6f}"
    )

    print(
        f"Mean logprob:    "
        f"{confidence['mean_log_probability']:.6f}"
    )

    print(
        f"Minimum:         "
        f"{confidence['minimum_probability']:.6f}"
    )

    print(
        f"Maximum:         "
        f"{confidence['maximum_probability']:.6f}"
    )

    print(
        f"Median:          "
        f"{confidence['median_probability']:.6f}"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    if args.save:

        output_path = Path(args.save)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"Results saved to {output_path}"
        )


if __name__ == "__main__":
    main()