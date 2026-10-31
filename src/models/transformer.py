"""A minimal, fully transparent Transformer for discrete algorithmic tasks.

Design goals
------------
1. **No black boxes.**  Every parameter is an explicit ``torch.nn.Parameter``
   and every op is a named tensor product, so the forward pass can be read off
   the code and translated to SMT / symbolic algebra without guessing.
2. **No LayerNorm.**  Layer normalisation destroys the clean bilinear algebra
   of the attention circuits (it rescales the residual stream by a
   data-dependent factor) and is hostile to SMT translation.  The architecture
   therefore uses a plain pre-norm-free residual stream, exactly as in the
   canonical grokking setup of Power et al. (2022) / Nanda et al. (2023).
3. **Full activation trace.**  :meth:`Transformer.forward` can return a
   :class:`ForwardTrace` holding the residual stream, per-head attention
   patterns, queries/keys/values and per-layer outputs for *every* token, which
   is what the interpretability stage consumes.

Tokenisation convention
-----------------------
A modular-arithmetic example is encoded as the fixed-length sequence::

    tokens = [a, b, EQ]        with   EQ = p (a reserved token)

and the prediction is read from the final position (``EQ``).  Because the query
position is a *constant* token, the residual stream at that position is a pure
function of ``(a, b)``, which makes the extracted ``W_eff`` / ``W_OV`` matrices
exactly interpretable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "ModelConfig",
    "ForwardTrace",
    "Transformer",
    "modular_addition_dataset",
    "modular_addition_tokens",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Static hyper-parameters of the transparent Transformer.

    Parameters
    ----------
    p:
        The modulus.  The vocabulary is ``{0, ..., p-1, EQ}`` with
        ``vocab_size = p + 1`` and ``n_classes = p``.
    seq_len:
        Number of input tokens.  ``3`` for the ``[a, b, EQ]`` encoding.
    d_model:
        Residual-stream width.
    n_heads:
        Number of attention heads per layer.
    n_layers:
        Number of blocks (1 for the canonical grokking circuit).
    mlp_width:
        Width of the optional ReLU MLP inside each block.  ``0`` disables the
        MLP, yielding the purely attention-based architecture whose circuits
        are analysed in the paper.  When ``> 0`` the model is piecewise linear
        after attention and therefore *exactly* translatable to Z3.
    causal:
        Whether attention is causally masked.
    attn_scale:
        ``"sqrt"`` divides the logits by ``sqrt(d_head)`` (standard), ``"none"``
        leaves them unscaled.  Keeping the scale explicit is important because
        the SMT translation must reproduce it exactly.
    init_std:
        Standard deviation used for the (seeded) normal initialisation.
    """

    p: int = 59
    seq_len: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    mlp_width: int = 0
    causal: bool = False
    attn_scale: str = "sqrt"
    init_std: float = 0.02

    # -- derived quantities -------------------------------------------------
    @property
    def n_classes(self) -> int:
        return self.p

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        return self.d_model // self.n_heads

    @property
    def answer_position(self) -> int:
        """Index of the token whose residual stream is unembedded."""
        return self.seq_len - 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "p": self.p,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "mlp_width": self.mlp_width,
            "causal": self.causal,
            "attn_scale": self.attn_scale,
            "init_std": self.init_std,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# Activation trace
# ---------------------------------------------------------------------------
@dataclass
class ForwardTrace:
    """All intermediate activations of one forward pass.

    Shapes are given for a batch of ``B`` examples with ``T`` tokens, ``d``
    model width and ``H`` heads of width ``d_h``.
    """

    tokens: Tensor                      # (B, T) int64
    embedding: Tensor                   # (B, T, d)
    attention_patterns: List[Tensor]    # L x (B, H, T, T)
    attention_scores: List[Tensor]      # L x (B, H, T, T)
    head_outputs: List[Tensor]          # L x (B, T, H, d)  (per-head residual contribution)
    layer_outputs: List[Tensor]         # L x (B, T, d)       (block output)
    residual_stream: List[Tensor]       # (L+1) x (B, T, d)
    mlp_hidden: List[Tensor] = field(default_factory=list)  # L x (B, T, w)

    def answer_residual(self, position: int) -> Tensor:
        """Residual stream at the answer position after the last block."""
        return self.residual_stream[-1][:, position, :]

    def head_attribution(self, layer: int, position: int) -> Tensor:
        """Attention pattern of layer ``layer`` at the query ``position``."""
        return self.attention_patterns[layer][:, :, position, :]


# ---------------------------------------------------------------------------
# The Transformer
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    """Attention-only (or attention + ReLU MLP) Transformer with explicit weights.

    Parameter inventory (all bias-free so that the network is a composition of
    linear maps and softmaxes):

    ``W_E``   : ``(vocab_size, d_model)``     token embedding / unembedding basis
    ``W_Q[l]``: ``(d_model, n_heads, d_head)``
    ``W_K[l]``: ``(d_model, n_heads, d_head)``
    ``W_V[l]``: ``(d_model, n_heads, d_head)``
    ``W_O[l]``: ``(d_model, n_heads, d_head)``  (contracts head space back)
    ``W_in[l]`` / ``W_out[l]``: optional MLP
    ``W_U``   : ``(d_model, n_classes)``      unembedding
    """

    def __init__(self, config: ModelConfig, seed: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        cfg = config
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))

        def normal(*shape: int) -> nn.Parameter:
            if gen is None:
                t = torch.randn(*shape) * cfg.init_std
            else:
                t = torch.randn(*shape, generator=gen) * cfg.init_std
            return nn.Parameter(t)

        self.W_E = normal(cfg.vocab_size, cfg.d_model)
        self.W_U = normal(cfg.d_model, cfg.n_classes)

        self.W_Q = nn.ParameterList()
        self.W_K = nn.ParameterList()
        self.W_V = nn.ParameterList()
        self.W_O = nn.ParameterList()
        self.W_in = nn.ParameterList()
        self.W_out = nn.ParameterList()
        for _ in range(cfg.n_layers):
            self.W_Q.append(normal(cfg.d_model, cfg.n_heads, cfg.d_head))
            self.W_K.append(normal(cfg.d_model, cfg.n_heads, cfg.d_head))
            self.W_V.append(normal(cfg.d_model, cfg.n_heads, cfg.d_head))
            self.W_O.append(normal(cfg.d_model, cfg.n_heads, cfg.d_head))
            if cfg.mlp_width > 0:
                self.W_in.append(normal(cfg.d_model, cfg.mlp_width))
                self.W_out.append(normal(cfg.mlp_width, cfg.d_model))

    # -- properties --------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def named_weight_summary(self) -> Dict[str, Tuple[int, ...]]:
        return {name: tuple(p.shape) for name, p in self.named_parameters()}

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        tokens: Tensor,
        *,
        trace: bool = False,
        positions: Optional[Sequence[int]] = None,
        detach_trace: bool = True,
        interventions: Optional[Sequence[Tuple[int, int, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[ForwardTrace]]:
        """Run the network.

        Parameters
        ----------
        tokens:
            ``(B, T)`` integer tensor of token ids.
        trace:
            When ``True`` a :class:`ForwardTrace` is returned as well.
        positions:
            Optional subset of token positions to keep in the trace (the
            attention patterns are sliced accordingly).  ``None`` keeps all.
        detach_trace:
            Detach the stored activations from the autograd graph.  Set to
            ``False`` when gradient attribution is required.
        interventions:
            Sequence of ``(layer, token_position, delta)`` triples.  ``delta``
            (shape ``(B, d_model)``) is **added** to the residual stream at that
            position.  ``layer`` indexes the residual stream like
            :attr:`ForwardTrace.residual_stream`: ``-1`` means "before the first
            block" and ``l >= 0`` means "after block ``l``".  This is the hook
            used by :mod:`src.interpretability.probing` for activation patching.

        Returns
        -------
        (logits, trace)
            ``logits`` has shape ``(B, T, n_classes)``.
        """
        cfg = self.config
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape (batch, seq_len)")
        if tokens.shape[1] != cfg.seq_len:
            raise ValueError(
                f"expected seq_len={cfg.seq_len} tokens, got {tokens.shape[1]}"
            )
        B, T = tokens.shape
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.d_head
        scale = 1.0 / math.sqrt(dh) if cfg.attn_scale == "sqrt" else 1.0

        x = self.W_E[tokens]  # (B, T, d)

        def apply_interventions(current: Tensor, stream_index: int) -> Tensor:
            """Add every intervention that targets residual-stream ``stream_index``."""
            for (iv_layer, iv_pos, iv_delta) in interventions or ():
                if iv_layer != stream_index:
                    continue
                if not 0 <= iv_pos < T:
                    raise ValueError(f"intervention position {iv_pos} out of range")
                delta = torch.as_tensor(iv_delta, dtype=current.dtype, device=current.device)
                if delta.shape != (B, cfg.d_model):
                    raise ValueError(
                        f"intervention delta must have shape {(B, cfg.d_model)}, "
                        f"got {tuple(delta.shape)}"
                    )
                current = current.clone()
                current[:, iv_pos, :] = current[:, iv_pos, :] + delta
            return current

        x = apply_interventions(x, -1)
        residuals: List[Tensor] = [x]
        patterns: List[Tensor] = []
        scores_all: List[Tensor] = []
        heads: List[Tensor] = []
        layers: List[Tensor] = []
        mlps: List[Tensor] = []

        for layer in range(cfg.n_layers):
            q = torch.einsum("btd,dhk->bthk", x, self.W_Q[layer])
            k = torch.einsum("btd,dhk->bthk", x, self.W_K[layer])
            v = torch.einsum("btd,dhk->bthk", x, self.W_V[layer])

            scores = torch.einsum("bthk,bshk->bhts", q, k) * scale  # (B,H,T,T)
            if cfg.causal:
                mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1
                )
                scores = scores.masked_fill(mask, float("-inf"))
            pattern = torch.softmax(scores, dim=-1)  # (B,H,T,T)

            z = torch.einsum("bhts,bshk->bthk", pattern, v)     # (B,T,H,dh)
            # Per-head output written back into the residual stream.
            # Per-head residual contribution: (B, T, H, d) from z (B, T, H, d_h).
            w_o = self.W_O[layer].permute(1, 2, 0)  # (H, d_h, d)
            per_head = torch.einsum("bthk,hkd->bthd", z, w_o)
            x = x + per_head.sum(dim=2)

            if cfg.mlp_width > 0:
                pre = x @ self.W_in[layer]
                hid = F.relu(pre)
                x = x + hid @ self.W_out[layer]
                mlps.append(hid)

            x = apply_interventions(x, layer)

            if trace:
                sl = slice(None) if positions is None else list(positions)
                post = (lambda t: t.detach()) if detach_trace else (lambda t: t)
                patterns.append(post(pattern[:, :, sl, :]))
                scores_all.append(post(scores[:, :, sl, :]))
                heads.append(post(per_head[:, sl, :, :]))
                layers.append(post(x[:, sl, :]))
            residuals.append(x)

        logits = x @ self.W_U  # (B, T, n_classes)

        if not trace:
            return logits, None

        post = (lambda t: t.detach()) if detach_trace else (lambda t: t)
        trace_obj = ForwardTrace(
            tokens=tokens.detach(),
            embedding=post(self.W_E[tokens]),
            attention_patterns=patterns,
            attention_scores=scores_all,
            head_outputs=heads,
            layer_outputs=layers,
            residual_stream=[post(r) for r in residuals],
            mlp_hidden=mlps,
        )
        return logits, trace_obj

    # -- convenience -------------------------------------------------------
    def answer_logits(self, a: Tensor, b: Tensor) -> Tensor:
        """Logits at the answer position for integer inputs ``a`` and ``b``.

        This is the entry point used by the synthesizer and the verifier, and
        deliberately bypasses batching concerns so that it can be called with a
        single scalar pair.
        """
        a = torch.as_tensor(a, dtype=torch.long).reshape(-1)
        b = torch.as_tensor(b, dtype=torch.long).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("a and b must have the same number of elements")
        tokens = modular_addition_tokens(a, b, self.config.p)
        logits, _ = self.forward(tokens)
        return logits[:, self.config.answer_position, :]

    @torch.no_grad()
    def predict(self, a: Tensor, b: Tensor) -> Tensor:
        return self.answer_logits(a, b).argmax(dim=-1)

    def save(self, path: str, *, extra: Optional[Dict[str, object]] = None) -> None:
        """Serialise weights *and* config (never pickle the module itself)."""
        payload: Dict[str, object] = {
            "config": self.config.to_dict(),
            "state_dict": {k: v.detach().clone() for k, v in self.state_dict().items()},
            "format_version": 1,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, map_location: str = "cpu") -> "Transformer":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError(f"{path} does not contain a Circuit2Code checkpoint")
        model = cls(ModelConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def modular_addition_tokens(a: Tensor, b: Tensor, p: int) -> Tensor:
    """Build the ``[a, b, EQ]`` token sequence for a batch of pairs."""
    a = torch.as_tensor(a, dtype=torch.long).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.long).reshape(-1)
    eq = torch.full_like(a, p)
    return torch.stack([a, b, eq], dim=1)


def modular_addition_dataset(
    p: int = 59,
    *,
    train_fraction: float = 0.4,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Complete ``a + b mod p`` dataset plus a fixed train/test split.

    The split is *deterministic*: a permutation of ``p*p`` examples generated by
    a seeded CPU generator, then cut at ``train_fraction``.

    Returns
    -------
    dict with keys
        ``tokens``     (N, 3)   full dataset
        ``a``, ``b``   (N,)
        ``target``     (N,)     ``(a + b) % p``
        ``train_idx``, ``test_idx`` (M,), (N-M,)
    """
    grid = torch.arange(p)
    a = grid.repeat_interleave(p)
    b = grid.repeat(p)
    target = (a + b) % p
    tokens = modular_addition_tokens(a, b, p)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    n_train = int(round(train_fraction * p * p))
    return {
        "tokens": tokens,
        "a": a,
        "b": b,
        "target": target,
        "train_idx": perm[:n_train],
        "test_idx": perm[n_train:],
    }
