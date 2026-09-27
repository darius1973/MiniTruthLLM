from pathlib import Path

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

OUTPUT_FILE = "data/raw/corpus.txt"

# This is deliberately small for our first experiment.
# 5 million characters is roughly 1-1.5 million English tokens,
# depending on the tokenizer.
MAX_CHARACTERS = 50_000_000

# Wikipedia dataset configuration.
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"


# ============================================================
# Download and build corpus
# ============================================================

def build_corpus():
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Wikipedia dataset...")
    print("Using streaming mode so we don't download everything.")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    total_characters = 0
    article_count = 0

    print("Building corpus...")

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for article in dataset:

            text = article.get("text", "").strip()

            if not text:
                continue

            # Separate articles clearly.
            f.write(text)
            f.write("\n\n")

            total_characters += len(text)
            article_count += 1

            if article_count % 100 == 0:
                print(
                    f"Articles: {article_count:,} | "
                    f"Characters: {total_characters:,}"
                )

            if total_characters >= MAX_CHARACTERS:
                break

    print()
    print("=" * 60)
    print("Corpus created")
    print("=" * 60)
    print(f"Articles:   {article_count:,}")
    print(f"Characters: {total_characters:,}")
    print(f"File:       {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    build_corpus()