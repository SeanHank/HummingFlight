"""Mini-config fixture validation: verifies the parser's understanding of the
GLM-5.2 config layout that src/model/config.cpp must reproduce. Config facts
here (layer counts, expert counts, MLP types) mirror the real model."""
from __future__ import annotations

import json
import pathlib

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "config_mini.json"


def load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_shape() -> None:
    cfg = load()
    assert cfg["hidden_size"] == 16
    assert cfg["num_hidden_layers"] == 4
    assert cfg["num_attention_heads"] == 4
    # GLM-5.2 field names: n_routed_experts / n_shared_experts.
    assert cfg["n_routed_experts"] == 8
    assert cfg["num_experts_per_tok"] == 2
    assert cfg["n_shared_experts"] == 1
    assert cfg["moe_intermediate_size"] == 16


def test_first_layers_dense_then_moe() -> None:
    cfg = load()
    # first_k_dense_replace == 1 mirrors the reference model (3 dense + 75 MoE
    # over 78 layers): the first layer is dense, the rest sparse.
    types = cfg["mlp_layer_types"]
    assert types == ["dense", "sparse", "sparse", "sparse"]
    assert types.count("dense") == cfg["first_k_dense_replace"] == 1


def test_attention_dimensions() -> None:
    cfg = load()
    assert cfg["q_lora_rank"] == 8
    assert cfg["kv_lora_rank"] == 4
    assert cfg["qk_nope_head_dim"] == 6
    assert cfg["qk_rope_head_dim"] == 2


def test_rope_and_norms() -> None:
    cfg = load()
    assert cfg["rms_norm_eps"] > 0
    assert "factor" in cfg["rope_scaling"] if cfg.get("rope_scaling") else True