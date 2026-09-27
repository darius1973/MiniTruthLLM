
"""
Little Language Model configuration

CPU-first configuration for:
    Intel Core i5 dual-core, 8 GB RAM

Llama-inspired decoder-only language model.
"""

from dataclasses import dataclass, field


# ============================================================
# Model configuration
# ============================================================

@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 8_000

    # Maximum sequence length
    block_size: int = 256

    # Transformer
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256

    # SwiGLU intermediate dimension
    n_ff: int = 704

    # Regularization
    dropout: float = 0.0

    # Llama-style models generally don't use biases
    bias: bool = False

    # Rotary positional embedding
    rope_theta: float = 10_000.0

    # Share token embedding and output projection
    tie_embeddings: bool = True


# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainingConfig:
     
    batch_size: int = 1

    gradient_accumulation_steps: int = 8

    learning_rate: float = 1.5e-4

    beta1: float = 0.9
    beta2: float = 0.95

    weight_decay: float = 0.1

    max_grad_norm: float = 1.0

    max_steps: int = 30_000

    eval_interval: int = 500
    eval_batches: int = 10

    checkpoint_interval: int = 1_000

    warmup_steps: int = 1_000

    num_workers: int = 0

    # Resume training from a checkpoint.
    resume_from: str | None = None


# ============================================================
# Data configuration
# ============================================================

@dataclass
class DataConfig:
    data_dir: str = "data"

    raw_dir: str = "data/raw"

    processed_dir: str = "data/processed"

    train_split: float = 0.90

    max_tokens: int = 5_000_000

    tokenizer_path: str = "data/tokenizer.json"

    seed: int = 1337


# ============================================================
# Generation configuration
# ============================================================

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128

    temperature: float = 0.8

    top_k: int = 40

    top_p: float = 0.9

    do_sample: bool = True


# ============================================================
# Confidence configuration
# ============================================================

@dataclass
class ConfidenceConfig:
    calculate_arithmetic: bool = True

    calculate_geometric: bool = True

    store_token_probabilities: bool = True

    min_probability: float = 1e-12


# ============================================================
# Runtime configuration
# ============================================================

@dataclass
class RuntimeConfig:
    device: str = "cpu"

    dtype: str = "float32"

    verbose: bool = True

    checkpoint_dir: str = "checkpoints"

    results_dir: str = "results"


# ============================================================
# Complete configuration
# ============================================================

@dataclass
class Config:
    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    data: DataConfig = field(
        default_factory=DataConfig
    )

    generation: GenerationConfig = field(
        default_factory=GenerationConfig
    )

    confidence: ConfidenceConfig = field(
        default_factory=ConfidenceConfig
    )

    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )


config = Config()


# ============================================================
# Display configuration
# ============================================================

def print_config():

    print("=" * 60)
    print("MiniTruth-8M configuration")
    print("=" * 60)

    print("\nMODEL")

    print(
        f"  Vocabulary:       "
        f"{config.model.vocab_size:,}"
    )

    print(
        f"  Layers:           "
        f"{config.model.n_layer}"
    )

    print(
        f"  Embedding size:   "
        f"{config.model.n_embd}"
    )

    print(
        f"  Attention heads:  "
        f"{config.model.n_head}"
    )

    print(
        f"  FFN size:         "
        f"{config.model.n_ff}"
    )

    print(
        f"  Context length:   "
        f"{config.model.block_size}"
    )

    print("\nTRAINING")

    print(
        f"  Batch size:       "
        f"{config.training.batch_size}"
    )

    print(
        f"  Gradient accum:   "
        f"{config.training.gradient_accumulation_steps}"
    )

    print(
        f"  Learning rate:    "
        f"{config.training.learning_rate}"
    )

    print(
        f"  Max steps:        "
        f"{config.training.max_steps}"
    )

    print("\nDATA")

    print(
        f"  Maximum tokens:   "
        f"{config.data.max_tokens:,}"
    )

    print(
        f"  Train split:      "
        f"{config.data.train_split:.0%}"
    )

    print("\nGENERATION")

    print(
        f"  Max new tokens:   "
        f"{config.generation.max_new_tokens}"
    )

    print(
        f"  Temperature:      "
        f"{config.generation.temperature}"
    )

    print(
        f"  Top-k:            "
        f"{config.generation.top_k}"
    )

    print(
        f"  Top-p:            "
        f"{config.generation.top_p}"
    )

    print("\nRUNTIME")

    print(
        f"  Device:           "
        f"{config.runtime.device}"
    )

    print(
        f"  Data type:        "
        f"{config.runtime.dtype}"
    )

    print("=" * 60)


if __name__ == "__main__":
    print_config()