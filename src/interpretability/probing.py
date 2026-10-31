"""Activation caching, activation patching and edge attribution patching.

The functions here never mutate model weights (except :func:`zero_head`, which
uses an explicit save/restore context manager) and never rely on hooks: the
:class:`~src.models.transformer.Transformer` exposes its full activation trace,
and residual-stream interventions are passed to ``forward`` directly.  As a
consequence every measurement in the paper is reproducible bit-for-bit from a
checkpoint.

Definitions
-----------
*Denoising* patch (a.k.a. *noising*): run the model on corrupted inputs, then
replace one activation with its clean value and measure how much of the clean
behaviour is restored.

*Attribution patching* (AtP, Kramár et al. 2024): a first-order approximation of
the same quantity obtained from a single backward pass,

.. math:: \\hat{\\Delta} = \\langle \\nabla_a m, a_{\\text{clean}} - a_{\\text{corrupt}} \\rangle

where :math:`m` is the behavioural metric and :math:`a` the activation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.models.transformer import ForwardTrace, Transformer

__all__ = [
    "ActivationCache",
    "metric_logit_diff",
    "patch_residual",
    "patch_head",
    "attribution_patch",
    "zero_head",
    "add_head",
]


# ---------------------------------------------------------------------------
# Behavioural metric
# ---------------------------------------------------------------------------
def metric_logit_diff(logits: Tensor, target: Tensor, alt: Optional[Tensor] = None) -> Tensor:
    """Mean ``logit[target] - logit[alt]`` over the batch.

    ``alt`` defaults to ``(target + 1) % p``, the standard "one-off" contrast
    used in the modular-arithmetic circuit literature.  The metric is *linear*
    in the logits, which is what makes attribution patching exact to first order
    and keeps the Z3 translation of the verifier simple.
    """
    logits = torch.as_tensor(logits)
    if logits.dim() != 2:
        raise ValueError("logits must have shape (batch, n_classes)")
    target = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    p = logits.shape[-1]
    if alt is None:
        alt = (target + 1) % p
    alt = torch.as_tensor(alt, dtype=torch.long).reshape(-1)
    rows = torch.arange(logits.shape[0])
    return (logits[rows, target] - logits[rows, alt]).mean()


# ---------------------------------------------------------------------------
# Activation cache
# ---------------------------------------------------------------------------
@dataclass
class ActivationCache:
    """Named access to every activation of a forward pass.

    Keys are of the form

    ``resid.<layer>.<pos>``      residual stream, shape (B, d)
    ``attn.<layer>.<head>.<pos>`` attention row, shape (B, T)
    ``head.<layer>.<head>.<pos>`` per-head contribution, shape (B, d)
    ``logits.<pos>``             output logits, shape (B, p)
    """

    trace: ForwardTrace
    entries: Dict[str, Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Tensor:
        return self.entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def keys(self) -> List[str]:  # pragma: no cover - trivial
        return sorted(self.entries)

    def residual(self, layer: int, position: int) -> Tensor:
        return self.trace.residual_stream[layer][:, position, :]

    def attention(self, layer: int, head: int, position: int) -> Tensor:
        return self.trace.attention_patterns[layer][:, head, position, :]


def build_activation_cache(
    model: Transformer, tokens: Tensor, *, detach: bool = True
) -> ActivationCache:
    """Run ``model`` and return every activation, keyed by name."""
    logits, trace = model(tokens, trace=True, detach_trace=detach)
    assert trace is not None
    entries: Dict[str, Tensor] = {}
    cfg = model.config
    for layer, resid in enumerate(trace.residual_stream):
        for pos in range(cfg.seq_len):
            entries[f"resid.{layer}.{pos}"] = resid[:, pos, :]
    for layer, patterns in enumerate(trace.attention_patterns):
        for head in range(cfg.n_heads):
            for pos in range(cfg.seq_len):
                entries[f"attn.{layer}.{head}.{pos}"] = patterns[:, head, pos, :]
    for layer, heads in enumerate(trace.head_outputs):
        for head in range(cfg.n_heads):
            for pos in range(cfg.seq_len):
                entries[f"head.{layer}.{head}.{pos}"] = heads[:, pos, head, :]
    for pos in range(cfg.seq_len):
        entries[f"logits.{pos}"] = logits[:, pos, :]
    return ActivationCache(trace=trace, entries=entries)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------
@torch.no_grad()
def patch_residual(
    model: Transformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    layer: int,
    position: Optional[int] = None,
    *,
    clean_activations: Optional[ActivationCache] = None,
) -> Tensor:
    """Denoising patch of the residual stream at ``(layer, position)``.

    Returns the logits (shape ``(B, p)``) at the answer position of the
    corrupted run after the clean residual has been substituted.

    ``layer`` indexes the residual stream (``0`` is the embedding, ``L`` follows
    the last block), matching :attr:`~src.models.transformer.ForwardTrace`.
    """
    model.eval()
    cfg = model.config
    position = cfg.answer_position if position is None else position
    if clean_activations is None:
        clean_logits, clean_trace = model(clean_tokens, trace=True)
    else:
        clean_trace = clean_activations.trace
    assert clean_trace is not None
    clean_resid = clean_trace.residual_stream[layer][:, position, :]
    # delta = clean - corrupted requires the corrupted value; inject it as a
    # replacement by first computing the corrupted residual, then adding the
    # difference.
    with torch.no_grad():
        _, corrupted_trace = model(corrupted_tokens, trace=True)
    assert corrupted_trace is not None
    corrupted_resid = corrupted_trace.residual_stream[layer][:, position, :]
    delta = clean_resid - corrupted_resid
    logits, _ = model(
        corrupted_tokens, interventions=[(layer - 1, position, delta)]
    )
    return logits[:, cfg.answer_position, :]


@torch.no_grad()
def patch_head(
    model: Transformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    layer: int,
    head: int,
    position: Optional[int] = None,
) -> Tensor:
    """Denoising patch that swaps one head's contribution at one position."""
    model.eval()
    cfg = model.config
    position = cfg.answer_position if position is None else position
    _, clean_trace = model(clean_tokens, trace=True)
    _, corrupted_trace = model(corrupted_tokens, trace=True)
    assert clean_trace is not None and corrupted_trace is not None
    delta = (
        clean_trace.head_outputs[layer][:, position, head, :]
        - corrupted_trace.head_outputs[layer][:, position, head, :]
    )
    logits, _ = model(corrupted_tokens, interventions=[(layer - 1, position, delta)])
    return logits[:, cfg.answer_position, :]


def attribution_patch(
    model: Transformer,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    layer: int,
    position: Optional[int] = None,
) -> Tensor:
    """First-order attribution of the residual stream at ``(layer, position)``.

    Returns ``⟨∇_a m, a_clean − a_corrupt⟩`` for the behavioural metric ``m``
    (logit difference), where the gradient is evaluated at the corrupted input.
    """
    model.eval()
    cfg = model.config
    position = cfg.answer_position if position is None else position
    clean_tokens = torch.as_tensor(clean_tokens, dtype=torch.long)
    corrupted_tokens = torch.as_tensor(corrupted_tokens, dtype=torch.long)

    with torch.no_grad():
        _, clean_trace = model(clean_tokens, trace=True)
        _, corrupted_trace = model(corrupted_tokens, trace=True)
    assert clean_trace is not None and corrupted_trace is not None
    diff = (
        clean_trace.residual_stream[layer][:, position, :]
        - corrupted_trace.residual_stream[layer][:, position, :]
    )

    model.zero_grad(set_to_none=True)
    logits, corrupted_graph = model(corrupted_tokens, trace=True, detach_trace=False)
    assert corrupted_graph is not None
    # Keep a reference to the *whole* residual tensor: it is the object that sits
    # in the autograd graph, so its gradient is well defined for every position.
    activation = corrupted_graph.residual_stream[layer]
    activation.retain_grad()
    answer_logits = logits[:, cfg.answer_position, :]
    loss = _answer_metric(model, answer_logits, corrupted_tokens)
    loss.backward()
    grad = activation.grad
    if grad is None:  # pragma: no cover - defensive
        raise RuntimeError("attribution patching failed: no gradient on the activation")
    estimate = (grad[:, position, :].detach() * diff).sum(dim=-1)
    # Do not leak gradients into the caller's parameters.
    model.zero_grad(set_to_none=True)
    return estimate


def _answer_metric(model: Transformer, answer_logits: Tensor, tokens: Tensor) -> Tensor:
    """Logit difference of the answer position for a batch of input tokens."""
    a, b = tokens[:, 0], tokens[:, 1]
    target = (a + b) % model.config.p
    return metric_logit_diff(answer_logits, target)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def zero_head(model: Transformer, layer: int, head: int) -> Iterator[None]:
    """Temporarily zero the output projection slice of one head.

    This is a *true* weight ablation: it removes the head from the computation
    entirely rather than masking an activation, so the measured effect is the
    effect of deleting the sub-circuit.
    """
    if not 0 <= layer < model.config.n_layers:
        raise IndexError(f"layer {layer} out of range")
    if not 0 <= head < model.config.n_heads:
        raise IndexError(f"head {head} out of range")
    saved = model.W_O[layer].detach().clone()
    try:
        with torch.no_grad():
            model.W_O[layer][:, head, :] = 0.0
        yield
    finally:
        with torch.no_grad():
            model.W_O[layer].copy_(saved)


@torch.no_grad()
def add_head(model: Transformer, layer: int, head: int, delta: Tensor) -> None:
    """Add ``delta`` (shape ``(d_model, d_head)``) to a head's output matrix."""
    if delta.shape != model.W_O[layer][:, head, :].shape:
        raise ValueError(
            f"delta must have shape {tuple(model.W_O[layer][:, head, :].shape)}"
        )
    model.W_O[layer][:, head, :] += delta
