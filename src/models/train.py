"""Deterministic training pipeline for the modular-addition task.

The loop is a textbook *full-batch* AdamW run with decoupled weight decay --
the configuration in which grokking is reliably observed (Power et al., 2022;
Nanda et al., 2023).  Everything that could introduce non-determinism is
explicit: the seed controls initialisation *and* the train/test split, the batch
is the whole training set, and no data augmentation or dropout is used.

Usage from Python::

    from src.models.train import TrainConfig, train_modular_addition
    result = train_modular_addition(TrainConfig(p=59, epochs=20000))
    transformer = result.model

Usage from the CLI::

    python demo/cli.py train --p 59 --epochs 20000 --out artifacts/checkpoints/add59.pt
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.transformer import (
    ModelConfig,
    Transformer,
    modular_addition_dataset,
)

__all__ = [
    "TrainConfig",
    "TrainResult",
    "evaluate",
    "train_modular_addition",
]


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyper-parameters of the training run.

    Defaults reproduce the canonical grokking regime on ``p = 59``: 30% of the
    ``p * p`` pairs for training, full-batch AdamW, ``weight_decay = 1.0`` and
    ``betas = (0.9, 0.98)``.
    """

    p: int = 59
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    mlp_width: int = 0
    init_std: float = 0.02
    train_fraction: float = 0.3
    epochs: int = 15000
    lr: float = 1e-3
    #: Learning-rate schedule: ``"constant"`` or a cosine decay to
    #: ``lr_final_frac * lr`` over the run.  Cosine decay is the standard way to
    #: anneal into the generalising basin once the memorising solution has been
    #: found.
    lr_schedule: str = "constant"
    lr_final_frac: float = 0.1
    #: Optional gradient-norm clipping (``None`` disables it).
    grad_clip: Optional[float] = None
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    seed: int = 42
    log_every: int = 500
    #: Stop once test accuracy has been 1.0 for this many consecutive evaluations.
    grok_patience: int = 400
    #: Continue training this many epochs after the first perfect test accuracy,
    #: which tends to "clean up" the circuit (weights become more symmetric and
    #: more concentrated on a few Fourier frequencies).
    post_grok_epochs: int = 1500
    eval_every: int = 100
    checkpoint_path: Optional[str] = None
    #: If set, write an intermediate checkpoint (and its history) whenever the
    #: best test accuracy improves, so an interrupted run is inspectable.
    checkpoint_every: Optional[int] = None
    #: Restrict the number of CPU threads.  Fixing this makes the floating point
    #: reduction order -- and hence the run -- bitwise reproducible.
    torch_threads: Optional[int] = None
    verbose: bool = True

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            p=self.p,
            seq_len=3,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_width=self.mlp_width,
            init_std=self.init_std,
        )


@dataclass
class TrainResult:
    model: Transformer
    config: TrainConfig
    history: List[Dict[str, float]] = field(default_factory=list)
    final_train_acc: float = 0.0
    final_test_acc: float = 0.0
    grokking_epoch: Optional[int] = None
    epochs_run: int = 0
    elapsed_s: float = 0.0
    checkpoint_path: Optional[str] = None

    def summary(self) -> Dict[str, object]:
        return {
            "p": self.config.p,
            "lr_schedule": self.config.lr_schedule,
            "epochs_run": self.epochs_run,
            "final_train_acc": round(self.final_train_acc, 6),
            "final_test_acc": round(self.final_test_acc, 6),
            "grokking_epoch": self.grokking_epoch,
            "elapsed_s": round(self.elapsed_s, 2),
            "n_params": self.model.n_params,
            "checkpoint_path": self.checkpoint_path,
        }

    def save_history(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"config": asdict(self.config), "summary": self.summary(), "history": self.history},
                handle,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: Transformer,
    dataset: Dict[str, torch.Tensor],
    index: torch.Tensor,
) -> tuple[float, float]:
    """Return ``(loss, accuracy)`` of the model on the subset ``index``."""
    model.eval()
    tokens = dataset["tokens"][index]
    target = dataset["target"][index]
    logits = model(tokens)[0][:, model.config.answer_position, :]
    loss = F.cross_entropy(logits, target).item()
    acc = (logits.argmax(dim=-1) == target).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_modular_addition(cfg: TrainConfig = TrainConfig()) -> TrainResult:
    """Train a Transformer to add modulo ``cfg.p``.

    The model is trained with cross-entropy on the answer position, full-batch,
    and restored to the parameters that achieved the *best test accuracy* (with
    the smallest loss as tie-breaker) so that the returned checkpoint is the
    grokked solution and not a noisy post-grok iteration.
    """
    if cfg.torch_threads is not None:
        torch.set_num_threads(int(cfg.torch_threads))
    torch.manual_seed(cfg.seed)
    # The dataset split is driven by its own generator, so it is independent of
    # the parameter initialisation generator.
    dataset = modular_addition_dataset(cfg.p, train_fraction=cfg.train_fraction, seed=cfg.seed)
    model = Transformer(cfg.model_config(), seed=cfg.seed)

    train_idx = dataset["train_idx"]
    test_idx = dataset["test_idx"]
    # With ``train_fraction == 1.0`` there is no held-out data; evaluate the
    # "test" metrics on the training set instead of producing NaNs.
    eval_on_train_instead = test_idx.numel() == 0
    if eval_on_train_instead:
        test_idx = train_idx
    train_tokens = dataset["tokens"][train_idx]
    train_target = dataset["target"][train_idx]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_acc = -1.0
    best_loss = math.inf
    grokking_epoch: Optional[int] = None
    grok_hold = 0
    last_checkpoint_epoch = -10**9
    start = time.time()
    epochs_run = 0

    if cfg.lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")

    for epoch in range(1, cfg.epochs + 1):
        if cfg.lr_schedule == "cosine":
            # Half-cosine from ``lr`` to ``lr_final_frac * lr`` over the run.
            phase = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, cfg.epochs)))
            lr_now = cfg.lr * (cfg.lr_final_frac + (1.0 - cfg.lr_final_frac) * phase)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
        model.train()
        logits = model(train_tokens)[0][:, model.config.answer_position, :]
        loss = F.cross_entropy(logits, train_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        epochs_run = epoch

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            tr_loss, tr_acc = evaluate(model, dataset, train_idx)
            te_loss, te_acc = evaluate(model, dataset, test_idx)
            record = {
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": tr_loss,
                "test_loss": te_loss,
                "train_acc": tr_acc,
                "test_acc": te_acc,
            }
            history.append(record)
            improved = te_acc > best_acc or (te_acc == best_acc and te_loss < best_loss)
            if improved:
                best_acc, best_loss = te_acc, te_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                # Rate-limit intermediate checkpoints to one per
                # ``checkpoint_every`` epochs, but never skip an improvement
                # just because it did not land on a round epoch: an interrupted
                # run must always leave an inspectable model behind.
                if cfg.checkpoint_path and cfg.checkpoint_every:
                    since_last = epoch - last_checkpoint_epoch
                    if since_last >= cfg.checkpoint_every:
                        last_checkpoint_epoch = epoch
                        _write_checkpoint(
                            model,
                            cfg.checkpoint_path,
                            tr_acc,
                            te_acc,
                            grokking_epoch,
                            epoch,
                            config=cfg,
                            history=history,
                            elapsed_s=time.time() - start,
                        )

            if te_acc >= 1.0:
                grok_hold += 1
                if grokking_epoch is None:
                    grokking_epoch = epoch
            else:
                grok_hold = 0

            if cfg.verbose and epoch % cfg.log_every == 0:
                print(
                    f"[epoch {epoch:6d}] train {tr_loss:8.5f}/{tr_acc:6.3f} "
                    f"test {te_loss:8.5f}/{te_acc:6.3f}",
                    flush=True,
                )

            if grokking_epoch is not None:
                # Keep training for a while after grokking, then stop.
                if epoch - grokking_epoch >= cfg.post_grok_epochs:
                    if cfg.verbose:
                        print(
                            f"[train] grokked at epoch {grokking_epoch}; "
                            f"stopping after {cfg.post_grok_epochs} post-grok epochs",
                            flush=True,
                        )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tr_loss, tr_acc = evaluate(model, dataset, train_idx)
    te_loss, te_acc = evaluate(model, dataset, test_idx)
    history.append(
        {
            "epoch": float(epochs_run),
            "train_loss": tr_loss,
            "test_loss": te_loss,
            "train_acc": tr_acc,
            "test_acc": te_acc,
        }
    )

    result = TrainResult(
        model=model,
        config=cfg,
        history=history,
        final_train_acc=tr_acc,
        final_test_acc=te_acc,
        grokking_epoch=grokking_epoch,
        epochs_run=epochs_run,
        elapsed_s=time.time() - start,
    )

    if cfg.checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.checkpoint_path)), exist_ok=True)
        # Always persist the trajectory next to the checkpoint: the figures in
        # the paper and README are generated from this file, never retyped.
        result.save_history(os.path.splitext(cfg.checkpoint_path)[0] + ".history.json")
        model.save(
            cfg.checkpoint_path,
            extra={
                "train": {
                    "final_train_acc": tr_acc,
                    "final_test_acc": te_acc,
                    "grokking_epoch": grokking_epoch,
                    "epochs_run": epochs_run,
                    "seed": cfg.seed,
                }
            },
        )
        result.checkpoint_path = cfg.checkpoint_path
    return result


def _write_checkpoint(
    model: Transformer,
    path: str,
    train_acc: float,
    test_acc: float,
    grokking_epoch: Optional[int],
    epoch: int,
    *,
    config: Optional[TrainConfig] = None,
    history: Optional[List[Dict[str, float]]] = None,
    elapsed_s: float = 0.0,
) -> None:
    """Persist the weights and, when given, the trajectory seen so far.

    The trajectory is flushed with every intermediate checkpoint so that a run
    that is interrupted (or a machine that dies) still leaves a usable history
    next to the weights, with ``"summary": {"final": false}``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.save(
        path,
        extra={
            "train": {
                "final_train_acc": train_acc,
                "final_test_acc": test_acc,
                "grokking_epoch": grokking_epoch,
                "epochs_run": epoch,
            }
        },
    )
    if config is not None and history is not None:
        payload = {
            "config": asdict(config),
            "summary": {
                "p": config.p,
                "lr_schedule": config.lr_schedule,
                "epochs_run": epoch,
                "final_train_acc": round(train_acc, 6),
                "final_test_acc": round(test_acc, 6),
                "grokking_epoch": grokking_epoch,
                "elapsed_s": round(elapsed_s, 2),
                "n_params": model.n_params,
                "checkpoint_path": path,
                "final": False,
            },
            "history": history,
        }
        with open(os.path.splitext(path)[0] + ".history.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Train the Circuit2Code reference model.")
    parser.add_argument("--p", type=int, default=59)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="artifacts/checkpoints/add59.pt")
    args = parser.parse_args(argv)

    result = train_modular_addition(
        TrainConfig(
            p=args.p,
            epochs=args.epochs,
            d_model=args.d_model,
            n_heads=args.n_heads,
            seed=args.seed,
            checkpoint_path=args.out,
        )
    )
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
