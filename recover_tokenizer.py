from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


INPUT_FILE = "data/raw/corpus.txt"
OUTPUT_FILE = "data/tokenizer_recovered.json"

VOCAB_SIZE = 8_000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


print("Training recovery tokenizer...")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)

tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
    add_prefix_space=False
)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
)

tokenizer.train(
    [INPUT_FILE],
    trainer
)

# Add decoder AFTER training.
tokenizer.decoder = ByteLevelDecoder()

tokenizer.save(OUTPUT_FILE)

print()
print("Recovery tokenizer saved.")
print("Vocabulary size:", tokenizer.get_vocab_size())

test = tokenizer.encode(
    "American politician and author"
)

print()
print("Test IDs:")
print(test.ids)

print("Test tokens:")
print(test.tokens)

print("Decoded:")
print(repr(tokenizer.decode(test.ids)))