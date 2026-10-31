"""Shared fixtures: small, fast, deterministic models."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import ModelConfig, Transformer, modular_addition_dataset  # noqa: E402

SEED = 42


@pytest.fixture(scope="session", autouse=True)
def _deterministic_torch():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    yield


@pytest.fixture(scope="session")
def small_config() -> ModelConfig:
    """A tiny model (p=7) used by the structural tests."""
    return ModelConfig(p=7, d_model=16, n_heads=2, n_layers=1)


@pytest.fixture(scope="session")
def small_model(small_config) -> Transformer:
    return Transformer(small_config, seed=SEED).eval()


@pytest.fixture(scope="session")
def small_dataset(small_config) -> dict:
    return modular_addition_dataset(small_config.p, train_fraction=0.5, seed=SEED)


@pytest.fixture(scope="session")
def medium_config() -> ModelConfig:
    return ModelConfig(p=11, d_model=32, n_heads=2)


@pytest.fixture(scope="session")
def medium_model(medium_config) -> Transformer:
    return Transformer(medium_config, seed=SEED).eval()


@pytest.fixture(scope="session")
def medium_dataset(medium_config) -> dict:
    return modular_addition_dataset(medium_config.p, train_fraction=0.5, seed=SEED)


@pytest.fixture(scope="session")
def trained_checkpoint() -> str:
    """Path to the reference p=59 checkpoint, if it has been produced."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "checkpoints",
        "add59.pt",
    )
    if not os.path.exists(path):
        pytest.skip("reference checkpoint missing (run `make train`)")
    return path
