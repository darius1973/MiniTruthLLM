from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


# ============================================================
# Configuration
# ============================================================

CORPUS_FILE = "data/raw/corpus.txt"
TOKENIZER_FILE = "data/tokenizer.json"
OUTPUT_DIR = Path("data/processed")

TRAIN_SPLIT = 0.90
SEED = 1337


# ============================================================
# Prepare corpus
# ============================================================

def prepare_data():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniTruth-8M data preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    vocab_size = tokenizer.get_vocab_size()

    print(f"Vocabulary size: {vocab_size:,}")

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    corpus_path = Path(CORPUS_FILE)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}"
        )

    text = corpus_path.read_text(
        encoding="utf-8"
    )

    print(f"Characters: {len(text):,}")

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing corpus...")
    print("This may take a little while on your CPU.")

    encoded = tokenizer.encode(text)

    token_ids = np.asarray(
        encoded.ids,
        dtype=np.uint16,
    )

    print(f"Tokens: {len(token_ids):,}")

    # --------------------------------------------------------
    # Check that uint16 is sufficient
    # --------------------------------------------------------

    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            "Vocabulary is too large for uint16."
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split_index = int(
        len(token_ids) * TRAIN_SPLIT
    )

    train_tokens = token_ids[:split_index]
    val_tokens = token_ids[split_index:]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_file = output_dir / "train.bin"
    val_file = output_dir / "val.bin"

    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Data preparation complete")
    print("=" * 60)

    print(f"Total tokens:       {len(token_ids):,}")
    print(f"Training tokens:    {len(train_tokens):,}")
    print(f"Validation tokens:  {len(val_tokens):,}")

    print()
    print(f"Training file:      {train_file}")
    print(f"Validation file:    {val_file}")

    print()
    print(
        f"Training size:      "
        f"{train_file.stat().st_size / 1024 / 1024:.2f} MB"
    )

    print(
        f"Validation size:    "
        f"{val_file.stat().st_size / 1024 / 1024:.2f} MB"
    )

    print("=" * 60)


if __name__ == "__main__":
    prepare_data()