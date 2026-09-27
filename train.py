"""
MiniTruth-8M training script.

CPU-first training loop.
"""
"""
MiniTruth-8M training script.

CPU-first training loop with checkpoint/resume support.
"""

import math
import os
import random
import time

import numpy as np
import torch

from config import config
from dataset import TokenDataset, get_batch
from model import MiniTruthLM


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Learning-rate schedule
# ============================================================

def get_learning_rate(step):

    warmup_steps = config.training.warmup_steps
    max_steps = config.training.max_steps
    max_lr = config.training.learning_rate

    if step < warmup_steps:
        return max_lr * (
            (step + 1) / warmup_steps
        )

    # Cosine decay
    decay_ratio = (
        (step - warmup_steps)
        / max(
            1,
            max_steps - warmup_steps,
        )
    )

    decay_ratio = min(
        1.0,
        decay_ratio,
    )

    min_lr = max_lr * 0.1

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * decay_ratio
        )
    )

    return (
        min_lr
        + cosine * (
            max_lr - min_lr
        )
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(
    model,
    train_dataset,
    val_dataset,
    device,
):

    model.eval()

    results = {}

    for name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
    ]:

        losses = []

        for _ in range(
            config.training.eval_batches
        ):

            x, y = get_batch(
                dataset,
                config.training.batch_size,
                device,
            )

            _, loss = model(x, y)

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    step,
):

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,

        # Random-number-generator states.
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
):

    print(
        f"Loading checkpoint: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    # Restore RNG states when available.
    #
    # Older checkpoints from the first training run
    # won't contain these, so we handle that gracefully.

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

    if "python_rng_state" in checkpoint:
        random.setstate(
            checkpoint["python_rng_state"]
        )

    checkpoint_step = checkpoint.get(
        "step",
        0,
    )

    print(
        f"Checkpoint was saved at step "
        f"{checkpoint_step}"
    )

    return checkpoint_step


# ============================================================
# Main training function
# ============================================================

def train():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        config.runtime.device
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        config.data.seed
    )

    print("=" * 60)
    print("MiniTruth-8M training")
    print("=" * 60)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = TokenDataset(
        "train"
    )

    val_dataset = TokenDataset(
        "val"
    )

    print(
        f"Training tokens: "
        f"{len(train_dataset.data):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(val_dataset.data):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = MiniTruthLM()

    model.to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(
            config.training.beta1,
            config.training.beta2,
        ),
        weight_decay=config.training.weight_decay,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_step = 0

    resume_from = (
        config.training.resume_from
    )

    if resume_from is not None:

        checkpoint_step = load_checkpoint(
            resume_from,
            model,
            optimizer,
        )

        # Checkpoints store the completed step.
        # The loop uses zero-based step numbers,
        # so continue at the next step.
        start_step = checkpoint_step

        print(
            f"Resuming training at step "
            f"{start_step + 1}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    model.train()

    start_time = time.time()

    running_loss = 0.0

    for step in range(
        start_step,
        config.training.max_steps,
    ):

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        learning_rate = get_learning_rate(
            step
        )

        for param_group in (
            optimizer.param_groups
        ):

            param_group["lr"] = (
                learning_rate
            )

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0

        for _ in range(
            config.training
            .gradient_accumulation_steps
        ):

            x, y = get_batch(
                train_dataset,
                config.training.batch_size,
                device,
            )

            _, loss = model(
                x,
                y,
            )

            loss = (
                loss
                / config.training
                .gradient_accumulation_steps
            )

            loss.backward()

            accumulated_loss += (
                loss.item()
            )

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.max_grad_norm,
        )

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        optimizer.step()

        running_loss += (
            accumulated_loss
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step == start_step
            or (
                step + 1
            )
            % config.training.eval_interval
            == 0
        ):

            losses = estimate_loss(
                model,
                train_dataset,
                val_dataset,
                device,
            )

            elapsed = (
                time.time()
                - start_time
            )

            if step == start_step:
                average_loss = (
                    running_loss
                )
            else:
                average_loss = (
                    running_loss
                    / config.training
                    .eval_interval
                )

            print(
                f"Step {step + 1:5d} | "
                f"Train loss {losses['train']:.4f} | "
                f"Val loss {losses['val']:.4f} | "
                f"LR {learning_rate:.2e} | "
                f"Time {elapsed / 60:.1f} min"
            )

            running_loss = 0.0

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            step + 1
        ) % config.training.checkpoint_interval == 0:

            os.makedirs(
                config.runtime.checkpoint_dir,
                exist_ok=True,
            )

            checkpoint_path = os.path.join(
                config.runtime.checkpoint_dir,
                f"step_{step + 1}.pt",
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step + 1,
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    os.makedirs(
        config.runtime.checkpoint_dir,
        exist_ok=True,
    )

    final_path = os.path.join(
        config.runtime.checkpoint_dir,
        "final.pt",
    )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        config.training.max_steps,
    )

    print(
        f"\nFinal model saved to: "
        f"{final_path}"
    )

    print("\nTraining complete.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train()