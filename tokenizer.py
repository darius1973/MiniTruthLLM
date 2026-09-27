from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


VOCAB_SIZE = 8_000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


def train_tokenizer(
    input_file="data/raw/corpus.txt",
    output_file="data/tokenizer.json",
):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train(
        [input_file],
        trainer
    )

    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(output_file)

    print(f"Saved tokenizer: {output_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    test = tokenizer.encode(
        "American politician and author"
    )

    print("Test IDs:", test.ids)
    print("Test tokens:", test.tokens)
    print("Decoded:", repr(tokenizer.decode(test.ids)))


if __name__ == "__main__":
    train_tokenizer()