#!/usr/bin/env python3
"""Convert Qwen3 MoE test checkpoints (MLA-test and DSA-test) from Megatron to HuggingFace format.

Usage (requires a SLURM allocation with GPU):

    srun --jobid=<JOBID> --nodes=1 --ntasks-per-node=1 --mpi=pmix \
      --environment=/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps1.toml \
      bash -c "PYTHONPATH=/iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/src:\$PYTHONPATH \
               python /iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/convert_mla_test_to_hf.py"

Uses the Qwen3MoEModelProvider directly (with MLATransformerConfig inheritance) to
build a Megatron model matching the training config, loads checkpoint weights,
then remaps the state dict to HF naming conventions and saves as safetensors.

The standard Qwen3MoEBridge mapping doesn't handle MLA attention weights, so we
do the Megatron->HF weight name remapping manually (following DeepSeek conventions
for MLA weight names: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj).

Checkpoints can specify a wandb_config.json path to load model config from a
Megatron-Bridge training run.
"""

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.model_load_save import temporary_distributed_context


@dataclass
class ModelConfig:
    """Model architecture config for Megatron <-> HF conversion."""

    num_layers: int
    num_experts: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    kv_channels: int
    q_lora_rank: Optional[int]
    hf_model_id: str


# Default config for small test checkpoints (mla-test, dsa-test)
DEFAULT_CONFIG = ModelConfig(
    num_layers=3,
    num_experts=8,
    hidden_size=128,
    num_attention_heads=32,
    num_query_groups=4,
    ffn_hidden_size=6144,
    moe_ffn_hidden_size=768,
    moe_router_topk=8,
    kv_channels=128,
    q_lora_rank=32,
    hf_model_id="Qwen/Qwen3-30B-A3B",
)

CHECKPOINTS = {
    "mla-test": {
        "megatron": "/capstor/scratch/cscs/ntazi/checkpoints/1n_tp1_ep4_fsdp0_pp1_vpp0_nl3_ne8_hs128_mffn768_topk8_gbs12_mbs3_vocswissai_ngc_25-11-nemo-alps1_recomp-sel_mla-test",
        "hf": "/capstor/scratch/cscs/ntazi/checkpoints/qwen3_moe_mla_test_hf",
        "mla": True,  # trained before MLA Hydra fix — standard attention
        "dsa": False,
    },
    "dsa-test": {
        "megatron": "/capstor/scratch/cscs/ntazi/checkpoints/1n_tp1_ep4_fsdp0_pp1_vpp0_nl3_ne8_hs128_mffn768_topk8_gbs12_mbs3_vocswissai_ngc_25-11-nemo-alps1_recomp-sel_dsa-test",
        "hf": "/capstor/scratch/cscs/ntazi/checkpoints/qwen3_moe_dsa_test_hf",
        "mla": True,
        "dsa": True,
    },
    "256n_gbs4096_muon_localattn": {
        "megatron": "/capstor/store/cscs/swissai/a139/checkpoints/moe_runs/256n_gbs4096_muon_localattn",
        "hf": "/capstor/scratch/cscs/mariagrandury/checkpoints/qwen3_moe_256n_gbs4096_muon_localattn_hf",
        "wandb_config": "scripts/wandb_config.json",
    },
}

def load_config_from_wandb(wandb_path: str) -> tuple[ModelConfig, bool, bool]:
    """Load model config, mla, and dsa from a wandb_config.json export.

    Returns:
        (ModelConfig, mla, dsa)
    """
    path = Path(wandb_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / wandb_path
    data = json.loads(path.read_text())

    model_cfg = data.get("model", {}).get("value", {})

    mla = model_cfg.get("multi_latent_attention", False)
    dsa = model_cfg.get("experimental_attention_variant") == "dsa"

    config = ModelConfig(
        num_layers=model_cfg["num_layers"],
        num_experts=model_cfg["num_moe_experts"],
        hidden_size=model_cfg["hidden_size"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_query_groups=model_cfg["num_query_groups"],
        ffn_hidden_size=model_cfg["ffn_hidden_size"],
        moe_ffn_hidden_size=model_cfg["moe_ffn_hidden_size"],
        moe_router_topk=model_cfg["moe_router_topk"],
        kv_channels=model_cfg["kv_channels"],
        q_lora_rank=model_cfg["q_lora_rank"],
        hf_model_id=model_cfg.get("hf_model_id", "Qwen/Qwen3-30B-A3B"),
    )
    print(config, mla, dsa)
    return config, mla, dsa


def get_checkpoint_config(cfg: dict) -> tuple[ModelConfig, bool, bool]:
    """Resolve ModelConfig, mla, and dsa for a checkpoint entry."""
    if "wandb_config" in cfg:
        return load_config_from_wandb(cfg["wandb_config"])
    return (
        DEFAULT_CONFIG,
        cfg.get("mla", True),
        cfg.get("dsa", False),
    )


def find_latest_iter(ckpt_dir: str) -> Path:
    """Find the latest *completed* iter_* subdirectory in a checkpoint."""
    ckpt_path = Path(ckpt_dir)
    latest_file = ckpt_path / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        iteration = int(latest_file.read_text().strip())
        return ckpt_path / f"iter_{iteration:07d}"
    iter_folders = [f for f in ckpt_path.iterdir() if f.is_dir() and f.name.startswith("iter_")]
    if iter_folders:
        return max(iter_folders, key=lambda f: int(f.name.replace("iter_", "")))
    return ckpt_path


def build_provider(config: ModelConfig, mla: bool = True, dsa: bool = False) -> Qwen3MoEModelProvider:
    """Build a Qwen3MoEModelProvider matching the training config."""
    provider = Qwen3MoEModelProvider(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        ffn_hidden_size=config.ffn_hidden_size,
        moe_ffn_hidden_size=config.moe_ffn_hidden_size,
        num_moe_experts=config.num_experts,
        moe_router_topk=config.moe_router_topk,
        kv_channels=config.kv_channels,
        seq_length=4096,
        multi_latent_attention=mla,
        q_lora_rank=config.q_lora_rank if mla else None,
    )
    if dsa:
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_n_heads = 16
        provider.dsa_indexer_head_dim = 128
        provider.dsa_indexer_topk = 256
        provider.dsa_indexer_loss_coeff = 0.001
    return provider


# ---------------------------------------------------------------------------
# Megatron -> HF weight name remapping
# ---------------------------------------------------------------------------

# MLA attention weight mapping (Megatron suffix -> HF suffix)
MLA_ATTENTION_MAP = {
    "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
    "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
    "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
    "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
    "self_attention.kv_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
}

# Standard (non-MLA) attention mapping
STANDARD_ATTENTION_MAP = {
    "self_attention.linear_qkv.weight": None,  # Special: needs QKV split
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
}

# Common (non-attention) layer weight mapping
COMMON_MAP = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "input_layernorm.weight": "input_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
}

# Top-level (non-layer) mapping
TOP_LEVEL_MAP = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
}


def _remap_expert_key(suffix: str) -> Optional[str]:
    """Remap expert MLP weight names.

    Megatron: mlp.experts.linear_fc1.weight{idx}  (gate+up fused)
              mlp.experts.linear_fc2.weight{idx}  (down_proj)
    HF:       mlp.experts.{idx}.gate_proj.weight
              mlp.experts.{idx}.up_proj.weight
              mlp.experts.{idx}.down_proj.weight
    """
    m = re.match(r"mlp\.experts\.linear_fc2\.weight(\d+)", suffix)
    if m:
        return f"mlp.experts.{m.group(1)}.down_proj.weight"

    m = re.match(r"mlp\.experts\.linear_fc1\.weight(\d+)", suffix)
    if m:
        return f"mlp.experts.{m.group(1)}._gated_fc1"  # sentinel for split

    return None


def _split_qkv(
    hf_state: dict,
    hf_prefix: str,
    qkv_weight: torch.Tensor,
    config: ModelConfig,
) -> None:
    """Split a fused QKV weight into separate Q, K, V weights."""
    q_size = config.num_attention_heads * config.kv_channels
    k_size = config.num_query_groups * config.kv_channels
    v_size = config.num_query_groups * config.kv_channels
    q, k, v = qkv_weight.split([q_size, k_size, v_size], dim=0)
    hf_state[f"{hf_prefix}.self_attn.q_proj.weight"] = q
    hf_state[f"{hf_prefix}.self_attn.k_proj.weight"] = k
    hf_state[f"{hf_prefix}.self_attn.v_proj.weight"] = v


def remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    mla: bool,
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    """Remap a Megatron model state dict to HF naming conventions."""
    attn_map = MLA_ATTENTION_MAP if mla else STANDARD_ATTENTION_MAP
    hf_state: dict[str, torch.Tensor] = OrderedDict()
    unmapped: list[str] = []

    for mcore_key, tensor in state_dict.items():
        # Top-level weights
        if mcore_key in TOP_LEVEL_MAP:
            hf_state[TOP_LEVEL_MAP[mcore_key]] = tensor
            continue

        # Layer weights: decoder.layers.{i}.{suffix}
        m = re.match(r"decoder\.layers\.(\d+)\.(.*)", mcore_key)
        if not m:
            unmapped.append(mcore_key)
            continue

        layer_idx, suffix = m.group(1), m.group(2)
        hf_prefix = f"model.layers.{layer_idx}"
        mapped = False

        # Common (non-attention) mappings
        for mcore_pattern, hf_pattern in COMMON_MAP.items():
            if suffix == mcore_pattern:
                hf_state[f"{hf_prefix}.{hf_pattern}"] = tensor
                mapped = True
                break
        if mapped:
            continue

        # Attention mappings
        for mcore_pattern, hf_pattern in attn_map.items():
            if suffix == mcore_pattern:
                if hf_pattern is None:
                    _split_qkv(hf_state, hf_prefix, tensor, config)
                else:
                    hf_state[f"{hf_prefix}.{hf_pattern}"] = tensor
                mapped = True
                break
        if mapped:
            continue

        # Expert weights
        expert_hf = _remap_expert_key(suffix)
        if expert_hf is not None:
            if expert_hf.endswith("._gated_fc1"):
                expert_prefix = expert_hf.replace("._gated_fc1", "")
                gate, up = tensor.chunk(2, dim=0)
                hf_state[f"{hf_prefix}.{expert_prefix}.gate_proj.weight"] = gate
                hf_state[f"{hf_prefix}.{expert_prefix}.up_proj.weight"] = up
            else:
                hf_state[f"{hf_prefix}.{expert_hf}"] = tensor
            continue

        unmapped.append(mcore_key)

    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped keys:")
        for k in unmapped[:20]:
            print(f"    {k}")
        if len(unmapped) > 20:
            print(f"    ... and {len(unmapped) - 20} more")

    return hf_state


def save_hf_checkpoint(
    hf_state: dict[str, torch.Tensor],
    output_dir: str,
    model_config: ModelConfig,
    mla: bool,
    dsa: bool,
) -> None:
    """Save remapped weights as safetensors + HF config."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    # safetensors requires contiguous tensors
    hf_state_contiguous = {k: v.contiguous() for k, v in hf_state.items()}
    save_file(hf_state_contiguous, out / "model.safetensors")

    # Save weight index
    weight_map = {k: "model.safetensors" for k in hf_state}
    total_size = sum(t.numel() * t.element_size() for t in hf_state.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # Save HF config (Qwen3 MoE base + MLA/DSA extensions)
    hf_config = AutoConfig.from_pretrained(model_config.hf_model_id, trust_remote_code=True)
    hf_config.num_hidden_layers = model_config.num_layers
    hf_config.hidden_size = model_config.hidden_size
    hf_config.intermediate_size = model_config.ffn_hidden_size
    hf_config.moe_intermediate_size = model_config.moe_ffn_hidden_size
    hf_config.num_attention_heads = model_config.num_attention_heads
    hf_config.num_key_value_heads = model_config.num_query_groups
    hf_config.num_experts = model_config.num_experts
    hf_config.num_experts_per_tok = model_config.moe_router_topk
    hf_config.vocab_size = 151936
    hf_config.max_position_embeddings = 40960
    hf_config.head_dim = model_config.kv_channels
    if mla:
        hf_config.multi_latent_attention = True
        hf_config.q_lora_rank = model_config.q_lora_rank
        hf_config.kv_lora_rank = 512  # MLATransformerConfig default
        hf_config.qk_nope_head_dim = 128  # qk_head_dim (DeepSeek HF naming)
        hf_config.qk_rope_head_dim = 64  # qk_pos_emb_head_dim
        hf_config.v_head_dim = 128
    if dsa:
        hf_config.experimental_attention_variant = "dsa"
        hf_config.dsa_indexer_n_heads = 16
        hf_config.dsa_indexer_head_dim = 128
        hf_config.dsa_indexer_topk = 256
        hf_config.dsa_indexer_loss_coeff = 0.001
    hf_config.save_pretrained(out)

    # Try to save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config.hf_model_id, trust_remote_code=True)
        tokenizer.save_pretrained(out)
        print(f"  Saved tokenizer")
    except Exception as e:
        print(f"  WARNING: Could not save tokenizer: {e}")

    print(f"  Saved {len(hf_state)} weight tensors ({total_size / 1e6:.1f} MB) to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Convert Megatron checkpoints to HF format.")
parser.add_argument(
    "--checkpoint",
    choices=sorted(CHECKPOINTS.keys()),
    help="Convert only one checkpoint key from CHECKPOINTS.",
)
args = parser.parse_args()

selected_checkpoints = (
    {args.checkpoint: CHECKPOINTS[args.checkpoint]} if args.checkpoint else CHECKPOINTS
)

with temporary_distributed_context(backend="gloo"):
    for name, cfg in selected_checkpoints.items():
        megatron_dir = cfg["megatron"]
        if not Path(megatron_dir).exists():
            print(f"\nSkipping {name}: checkpoint dir not found at {megatron_dir}")
            continue

        model_config, mla, dsa = get_checkpoint_config(cfg)
        print(f"\n{'='*60}")
        print(f"Converting {name} (MLA={mla}, DSA={dsa})")
        print(f"  Config: {model_config.num_layers}L, {model_config.hidden_size}H, {model_config.num_experts}E")
        print(f"{'='*60}")

        # 1) Build provider matching the training config
        provider = build_provider(model_config, mla=mla, dsa=dsa)
        provider.finalize()
        print(f"  Provider: {type(provider).__name__}, MLA={provider.multi_latent_attention}")

        # 2) Build Megatron model
        model = provider.provide()
        print(f"  Model built: {sum(p.numel() for p in model.parameters()):,} params")

        # 3) Load distributed checkpoint weights
        iter_path = find_latest_iter(megatron_dir)
        print(f"  Loading weights from {iter_path}")
        _load_model_weights_from_checkpoint(
            str(iter_path),
            [model],
            dist_ckpt_strictness="assume_ok_unexpected",
        )

        # 4) Remap state dict to HF naming
        megatron_sd = model.state_dict()
        print(f"  Remapping {len(megatron_sd)} Megatron keys to HF format...")
        hf_sd = remap_state_dict(megatron_sd, mla=mla, config=model_config)

        # 5) Save as HF checkpoint
        print(f"  Saving to {cfg['hf']}")
        save_hf_checkpoint(hf_sd, cfg["hf"], model_config, mla=mla, dsa=dsa)

        print(f"  Done: {name}")
