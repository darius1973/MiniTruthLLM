# MiniTruthLLM

**A small, CPU-friendly language model built from scratch in PyTorch.**

MiniTruthLLM is an experimental lightweight Large Language Model project focused on making language-model training and inference accessible on ordinary CPU machines.

The goal is simple: build a useful, understandable language model without requiring a GPU cluster.

> **Status:** 🚧 Experimental / actively developed

## ✨ Goals

MiniTruthLLM is designed around a few principles:

* 🖥️ **CPU-first** — designed to run on consumer CPUs
* 🪶 **Lightweight** — small enough to train and experiment with locally
* 🔍 **Understandable** — simple PyTorch code rather than a large framework stack
* 🧪 **Experiment-friendly** — easy to modify architectures, datasets, and training strategies
* 📦 **Self-contained** — minimal dependencies
* 📚 **Educational** — useful for learning how modern language models work

The project is not intended to compete with large production LLMs. Instead, it aims to make the mechanics of an LLM small enough to inspect, understand, and experiment with.

---

## 🧠 What is MiniTruthLLM?

MiniTruthLLM is a transformer-based language model implemented in PyTorch.

At a high level, the pipeline looks like this:

```text
Text
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Embedding
  │
  ▼
Transformer Blocks
  │
  ├── Self-Attention
  ├── Feed-Forward Network
  ├── Layer Normalization
  └── Residual Connections
  │
  ▼
Language-Model Head
  │
  ▼
Next-Token Probabilities
  │
  ▼
Generated Text
```

The model learns to predict the next token given the tokens that came before it.

---

## 🚀 Features

Current/planned features include:

* [x] PyTorch implementation
* [x] CPU inference
* [x] Autoregressive text generation
* [x] Transformer architecture
* [x] Configurable model size
* [x] Configurable context length
* [x] Checkpoint saving/loading
* [ ] Improved tokenizer
* [ ] Training from larger datasets
* [ ] Quantized inference
* [ ] Better sampling strategies
* [ ] Instruction tuning
* [ ] Evaluation benchmarks
* [ ] More efficient CPU kernels

---


## ⚙️ Requirements

MiniTruthLLM is intended to run on machines without a dedicated GPU.

Recommended environment:

* Python 3.10+
* PyTorch
* NumPy
* Optional: tqdm

Install the dependencies:

```bash
git clone https://github.com/yourusername/MiniTruthLLM.git
cd MiniTruthLLM

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## 🏃 Running on CPU

MiniTruthLLM automatically targets the CPU when no accelerator is available.

For example:

```python
import torch

device = torch.device("cpu")

model = MiniTruthLLM(...)
model.to(device)
```

For explicit CPU execution:

```bash
python scripts/generate.py --device cpu
```

This makes it possible to experiment with the model on machines such as:

* laptops
* desktops
* small home servers
* Raspberry Pi-class hardware
* older machines without CUDA-capable GPUs

Training speed will depend heavily on CPU performance, model size, sequence length, and dataset size.

---

## 🏋️ Training

Prepare the training data:

```bash
python scripts/prepare_data.py
```

Then start training:

```bash
python scripts/train.py
```

A typical training configuration might look like:

```yaml
model:
  vocab_size: 32000
  context_length: 256
  embedding_dim: 256
  num_layers: 6
  num_heads: 8

training:
  batch_size: 8
  learning_rate: 0.0003
  epochs: 5

device: cpu
```

For CPU experimentation, start with a small model and short context length. Once the complete training pipeline works, scale individual components gradually.

---

## 💬 Text Generation

After training, generate text from a checkpoint:

```bash
python scripts/generate.py \
    --checkpoint checkpoints/latest.pt \
    --prompt "The future of artificial intelligence"
```

Example Python usage:

```python
from minitruthllm import MiniTruthLLM

model = MiniTruthLLM.from_checkpoint(
    "checkpoints/latest.pt"
)

text = model.generate(
    "The future of artificial intelligence",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)

print(text)
```

---

## 🎛️ Generation Parameters

MiniTruthLLM supports common autoregressive sampling controls.

### Temperature

Controls randomness:

```text
temperature < 1.0
    More deterministic

temperature = 1.0
    Default sampling

temperature > 1.0
    More random
```

Example:

```bash
--temperature 0.7
```

### Top-k

Limits sampling to the most probable `k` tokens:

```bash
--top-k 40
```

### Top-p

Alternatively, nucleus sampling can restrict sampling to a probability mass:

```bash
--top-p 0.9
```

---

## 🧩 Model Architecture

MiniTruthLLM uses the standard autoregressive Transformer approach.

A simplified Transformer block is:

```text
Input
  │
  ├───────────────┐
  │               │
  ▼               │
LayerNorm         │
  │               │
  ▼               │
Self-Attention    │
  │               │
  └───────┬───────┘
          │
          ▼
       Residual
          │
  ├───────────────┐
  │               │
  ▼               │
LayerNorm         │
  │               │
  ▼               │
Feed Forward      │
  │               │
  └───────┬───────┘
          │
          ▼
       Residual
```

The model is trained using next-token prediction:

```text
Input:

"The cat sat on the"

Target:

"cat sat on the mat"
                    ↑
              predict this
```

During training, the model receives many examples of token sequences and learns to estimate:

```text
P(next_token | previous_tokens)
```

---

## 📊 Model Size

MiniTruthLLM is intended to support multiple configurations.

For example:

| Configuration | Parameters | Intended Use               |
| ------------- | ---------: | -------------------------- |
| Tiny          |     ~1–10M | Fast experiments           |
| Small         |    ~10–50M | CPU experimentation        |
| Medium        |   ~50–200M | More capable local model   |
| Custom        |   Variable | Research / experimentation |

Actual parameter counts depend on the architecture and configuration used.

The project prioritizes **practical experimentation over maximizing parameter count**.

---

## 🗃️ Dataset

MiniTruthLLM separates data preparation from model training.

The general pipeline is:

```text
Raw Dataset
     │
     ▼
Cleaning
     │
     ▼
Tokenization
     │
     ▼
Train / Validation Split
     │
     ▼
Training Batches
     │
     ▼
MiniTruthLLM
```

Datasets should be used only when their licensing and terms permit the intended use.

When adding a dataset to an experiment, record:

* Dataset name
* Source
* License
* Approximate size
* Preprocessing steps
* Tokenizer used
* Train/validation split

---

## 📈 Evaluation

Training loss alone does not fully describe a language model's behavior.

MiniTruthLLM can be evaluated using:

* Training loss
* Validation loss
* Perplexity
* Generation samples
* Token throughput
* CPU memory usage
* Inference latency

For example:

```text
Step:        10,000
Train loss:  2.84
Val loss:    3.01
Perplexity:  20.3
Tokens/sec:  145
```

Evaluation results should always be interpreted relative to the dataset, tokenizer, model configuration, and evaluation procedure.

---

## 🖥️ CPU Performance

CPU performance is one of the main constraints this project is designed around.

Training a Transformer entirely on CPU can be substantially slower than training on a modern GPU, particularly as model size and sequence length increase.

For CPU experiments, useful optimizations include:

* Smaller batch sizes
* Shorter context lengths
* Gradient accumulation
* Mixed precision where supported
* Efficient tokenization
* Efficient data loading
* Checkpointing
* Gradient clipping
* Appropriate PyTorch thread settings

For example:

```python
import torch

torch.set_num_threads(8)
```

The optimal number of threads depends on the hardware.

---

## 🔬 Why Build Another Small LLM?

Large language models can make the underlying technology difficult to study.

MiniTruthLLM takes the opposite approach:

> **Make the model small enough that you can understand the whole thing.**

Instead of requiring expensive hardware, the project aims to let developers experiment with:

* Attention mechanisms
* Tokenization
* Embeddings
* Positional information
* Transformer blocks
* Optimization
* Training dynamics
* Sampling
* Quantization
* Dataset preparation

on ordinary hardware.

---


## ⚠️ Limitations

MiniTruthLLM is an experimental small language model.

It should not be expected to match the capabilities of modern large-scale models trained with enormous datasets and compute budgets.

Generated text may be:

* incorrect
* repetitive
* incoherent
* biased by the training data
* factually unreliable
* sensitive to prompting and sampling parameters

The model should therefore be treated as an experimental research/learning project rather than a source of authoritative information.

---

## 🗺️ Roadmap

### Phase 1 — Foundation

* [x] Basic Transformer
* [x] Training loop
* [x] CPU inference
* [x] Checkpointing
* [x] Text generation

### Phase 2 — Efficiency

* [ ] Better CPU utilization
* [ ] KV caching
* [ ] Quantized weights
* [ ] Memory-efficient attention
* [ ] Faster tokenizer

### Phase 3 — Training

* [ ] Larger datasets
* [ ] Improved data pipeline
* [ ] Learning-rate scheduling
* [ ] Better evaluation
* [ ] Experiment tracking

### Phase 4 — Capabilities

* [ ] Instruction tuning
* [ ] Chat-style interface
* [ ] Fine-tuning support
* [ ] Retrieval experiments
* [ ] Tool-use experiments

---

## 📜 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## 🌱 Philosophy

MiniTruthLLM is built around a simple idea:

**You don't need a datacenter to learn how an LLM works.**

A small model running on a CPU is enough to explore the fundamental ideas behind modern language models—and that's what this project is about.

---

## ⭐ If You Find It Useful

If MiniTruthLLM helps you learn, experiment, or build something interesting, consider starring the repository and sharing your experiments.

Pull requests, issues, benchmarks, and architectural experiments are welcome.
