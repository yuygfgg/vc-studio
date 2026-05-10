from __future__ import annotations

from pathlib import Path
import sys

import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosyvoice.flow.DiT.dit import DiT
from cosyvoice.flow.flow_matching import CausalConditionalCFM


def test_uncached_grouped_prompt_flow_does_not_require_prepared_steps() -> None:
    torch.manual_seed(3)
    decoder = _tiny_decoder()
    prompt_len = 8
    source_len = 6
    grouped_prompt = {
        "branch_mus": [torch.randn(1, 80, prompt_len), torch.randn(1, 80, prompt_len + 2)],
        "branch_conds": [torch.randn(1, 80, prompt_len), torch.randn(1, 80, prompt_len + 2)],
        "branch_spks": [torch.randn(1, 80), torch.randn(1, 80)],
        "branch_weights": torch.tensor([0.4, 0.6]),
        "dominant_branch_position": 0,
        "attention_temperature": 1.0,
    }

    with torch.inference_mode():
        output, cache = decoder(
            mu=torch.randn(1, 80, prompt_len + source_len),
            mask=torch.ones(1, 1, prompt_len + source_len, dtype=torch.bool),
            n_timesteps=2,
            spks=torch.randn(1, 80),
            cond=torch.randn(1, 80, prompt_len + source_len),
            streaming=True,
            prompt_len=prompt_len,
            grouped_prompt_uncached=grouped_prompt,
        )

    assert output.shape == (1, 80, prompt_len + source_len)
    assert cache is None


def test_prepare_prompt_cache_can_store_step_kv_in_half_without_prompt_inputs() -> None:
    torch.manual_seed(4)
    decoder = _tiny_decoder()

    with torch.inference_mode():
        cache = decoder.prepare_prompt_cache(
            mu=torch.randn(1, 80, 8),
            mask=torch.ones(1, 1, 8, dtype=torch.bool),
            n_timesteps=2,
            spks=torch.randn(1, 80),
            cond=torch.randn(1, 80, 8),
            streaming=True,
            cache_storage_dtype=torch.float16,
            keep_prompt_inputs=False,
        )

    assert cache["steps"][0]["prompt_inputs"] is None
    key, value = cache["steps"][0]["prompt_cache"]["kv"][0]
    assert key.dtype == torch.float16
    assert value.dtype == torch.float16


def test_streaming_prompt_cache_preserves_padding_mask() -> None:
    torch.manual_seed(12)
    estimator = DiT(
        dim=16,
        depth=2,
        heads=2,
        dim_head=8,
        ff_mult=1,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        static_chunk_size=16,
        num_decoding_left_chunks=-1,
    ).eval()
    batch = 2
    seq_len = 5
    mask = torch.ones(batch, 1, seq_len, dtype=torch.bool)
    mask[0, :, -1] = False

    x = torch.randn(batch, 80, seq_len)
    mu = torch.randn(batch, 80, seq_len)
    cond = torch.randn(batch, 80, seq_len)
    t = torch.randn(batch)
    spks = torch.randn(batch, 80)

    with torch.inference_mode():
        streaming_output, streaming_cache = estimator.forward_prompt_cache(
            x,
            mask,
            mu,
            t,
            spks=spks,
            cond=cond,
            streaming=True,
        )
        full_output, full_cache = estimator.forward_prompt_cache(
            x,
            mask,
            mu,
            t,
            spks=spks,
            cond=cond,
            streaming=False,
        )

    assert torch.allclose(streaming_output, full_output, atol=1e-6, rtol=1e-6)
    for (streaming_key, streaming_value), (full_key, full_value) in zip(
        streaming_cache["kv"],
        full_cache["kv"],
    ):
        assert torch.allclose(streaming_key, full_key, atol=1e-6, rtol=1e-6)
        assert torch.allclose(streaming_value, full_value, atol=1e-6, rtol=1e-6)


def test_grouped_prompt_source_cache_builds_history_kv() -> None:
    decoder = _tiny_decoder()
    base_len = 4
    source_len = 5
    history_len = 2
    branch_count = 2
    cfg_batch = 2
    heads = 2
    head_dim = 8
    prompt_cache_steps = {
        "steps": [],
        "final_prompt_x": torch.randn(1, 80, base_len),
        "base_cache_len": base_len,
    }
    source_step_caches = []
    for _ in range(2):
        prompt_cache_steps["steps"].append(
            {
                "prompt_cache": {
                    "grouped_branch_attention": True,
                    "grouped_kv": [
                        (
                            torch.randn(branch_count, cfg_batch, heads, base_len, head_dim, dtype=torch.float16),
                            torch.randn(branch_count, cfg_batch, heads, base_len, head_dim, dtype=torch.float16),
                        )
                    ],
                    "grouped_prompt_mask": torch.ones(branch_count, base_len, dtype=torch.bool),
                    "branch_weights": torch.tensor([0.5, 0.5]),
                    "branch_indices": [0, 1],
                    "branch_cache_lens": [base_len, base_len],
                    "prompt_len": base_len,
                    "base_prompt_len": base_len,
                },
                "prompt_inputs": {
                    "x_prompt_in": torch.randn(cfg_batch, 80, base_len),
                    "prompt_mu_in": torch.randn(cfg_batch, 80, base_len),
                    "prompt_cond_in": torch.randn(cfg_batch, 80, base_len),
                    "t_in": torch.randn(cfg_batch),
                    "spks_in": torch.randn(cfg_batch, 80),
                },
            }
        )
        source_step_caches.append(
            {
                "source_cache": {
                    "kv": [
                        (
                            torch.randn(cfg_batch, heads, source_len, head_dim),
                            torch.randn(cfg_batch, heads, source_len, head_dim),
                        )
                    ],
                },
                "source_inputs": {
                    "x_source_in": torch.randn(cfg_batch, 80, source_len),
                    "source_mu_in": torch.randn(cfg_batch, 80, source_len),
                    "source_cond_in": torch.randn(cfg_batch, 80, source_len),
                },
            }
        )
    source_step_caches = [
        decoder.prepare_source_step_cache_for_history_storage(
            source_step_cache,
            prompt_cache_steps["steps"][index]["prompt_cache"],
            source_len - history_len,
            source_len,
        )
        for index, source_step_cache in enumerate(source_step_caches)
    ]
    assert source_step_caches[0]["source_cache_start"] == source_len - history_len
    assert source_step_caches[0]["source_cache"]["kv"][0][0].shape[2] == history_len
    assert source_step_caches[0]["source_cache"]["kv"][0][0].dtype == torch.float16
    assert source_step_caches[0]["source_inputs"]["x_source_in"].shape[2] == history_len

    updated = decoder.build_bounded_source_cache(
        prompt_cache_steps=prompt_cache_steps,
        source_step_caches=source_step_caches,
        source_x=torch.randn(1, 80, source_len),
        source_cache_len=history_len,
        source_cache_end=source_len,
    )

    first_prompt_cache = updated["steps"][0]["prompt_cache"]
    assert first_prompt_cache["grouped_branch_attention"] is True
    assert first_prompt_cache["prompt_len"] == base_len + history_len
    assert first_prompt_cache["base_prompt_len"] == base_len
    assert first_prompt_cache["history_kv"][0][0].shape[2] == history_len
    assert first_prompt_cache["history_kv"][0][0].dtype == torch.float16
    assert updated["history_cache_len"] == history_len


def test_sequential_grouped_prompt_source_cache_builds_history_kv() -> None:
    decoder = _tiny_decoder()
    base_len = 4
    source_len = 5
    history_len = 2
    branch_count = 2
    cfg_batch = 2
    heads = 2
    head_dim = 8
    prompt_cache_steps = {
        "steps": [],
        "final_prompt_x": torch.randn(1, 80, base_len),
        "base_cache_len": base_len,
    }
    source_step_caches = []
    for _ in range(2):
        prompt_cache_steps["steps"].append(
            {
                "prompt_cache": {
                    "grouped_branch_attention": True,
                    "grouped_attention_mode": "sequential",
                    "sequential_branch_caches": [
                        {
                            "branch_position": index,
                            "branch_index": index,
                            "cache_len": base_len,
                            "prompt_mask": torch.ones(base_len, dtype=torch.bool),
                            "kv": [
                                (
                                    torch.randn(cfg_batch, heads, base_len, head_dim, dtype=torch.float16),
                                    torch.randn(cfg_batch, heads, base_len, head_dim, dtype=torch.float16),
                                )
                            ],
                        }
                        for index in range(branch_count)
                    ],
                    "branch_weights": torch.tensor([0.5, 0.5]),
                    "branch_indices": [0, 1],
                    "branch_cache_lens": [base_len, base_len],
                    "prompt_len": base_len,
                    "base_prompt_len": base_len,
                },
                "prompt_inputs": {
                    "x_prompt_in": torch.randn(cfg_batch, 80, base_len),
                    "prompt_mu_in": torch.randn(cfg_batch, 80, base_len),
                    "prompt_cond_in": torch.randn(cfg_batch, 80, base_len),
                    "t_in": torch.randn(cfg_batch),
                    "spks_in": torch.randn(cfg_batch, 80),
                },
            }
        )
        source_step_caches.append(
            {
                "source_cache": {
                    "kv": [
                        (
                            torch.randn(cfg_batch, heads, source_len, head_dim),
                            torch.randn(cfg_batch, heads, source_len, head_dim),
                        )
                    ],
                },
                "source_inputs": {
                    "x_source_in": torch.randn(cfg_batch, 80, source_len),
                    "source_mu_in": torch.randn(cfg_batch, 80, source_len),
                    "source_cond_in": torch.randn(cfg_batch, 80, source_len),
                },
            }
        )

    updated = decoder.build_bounded_source_cache(
        prompt_cache_steps=prompt_cache_steps,
        source_step_caches=source_step_caches,
        source_x=torch.randn(1, 80, source_len),
        source_cache_len=history_len,
        source_cache_end=source_len,
    )

    first_prompt_cache = updated["steps"][0]["prompt_cache"]
    assert first_prompt_cache["grouped_attention_mode"] == "sequential"
    assert len(first_prompt_cache["sequential_branch_caches"]) == branch_count
    assert first_prompt_cache["prompt_len"] == base_len + history_len
    assert first_prompt_cache["history_kv"][0][0].shape[2] == history_len
    assert first_prompt_cache["history_kv"][0][0].dtype == torch.float16


def test_dit_forward_source_with_sequential_grouped_prompt_cache() -> None:
    torch.manual_seed(5)
    estimator = _tiny_decoder().estimator
    prompt_len = 4
    source_len = 33
    branch_count = 2
    cfg_batch = 2
    heads = 2
    head_dim = 8
    depth = len(estimator.transformer_blocks)
    prompt_cache = {
        "grouped_branch_attention": True,
        "grouped_attention_mode": "sequential",
        "sequential_branch_caches": [
            {
                "branch_position": branch_index,
                "branch_index": branch_index,
                "cache_len": prompt_len,
                "prompt_mask": torch.ones(prompt_len, dtype=torch.bool),
                "kv": [
                    (
                        torch.randn(cfg_batch, heads, prompt_len, head_dim),
                        torch.randn(cfg_batch, heads, prompt_len, head_dim),
                    )
                    for _ in range(depth)
                ],
            }
            for branch_index in range(branch_count)
        ],
        "branch_weights": torch.tensor([0.45, 0.55]),
        "branch_indices": [0, 1],
        "branch_cache_lens": [prompt_len, prompt_len],
        "prompt_len": prompt_len,
        "base_prompt_len": prompt_len,
    }

    with torch.inference_mode():
        output, source_cache = estimator.forward_source_with_prompt_cache(
            x=torch.randn(cfg_batch, 80, source_len),
            mask=torch.ones(cfg_batch, 1, prompt_len + source_len, dtype=torch.bool),
            mu=torch.randn(cfg_batch, 80, source_len),
            t=torch.randn(cfg_batch),
            spks=torch.randn(cfg_batch, 80),
            cond=torch.randn(cfg_batch, 80, source_len),
            prompt_x=torch.randn(cfg_batch, 80, prompt_len),
            prompt_mu=torch.randn(cfg_batch, 80, prompt_len),
            prompt_cond=torch.randn(cfg_batch, 80, prompt_len),
            prompt_cache=prompt_cache,
            streaming=True,
            return_source_cache=True,
        )

    assert output.shape == (cfg_batch, 80, source_len)
    assert source_cache is not None
    assert len(source_cache["kv"]) == depth
    assert source_cache["kv"][0][0].shape[2] == source_len


def _tiny_decoder() -> CausalConditionalCFM:
    estimator = DiT(
        dim=16,
        depth=2,
        heads=2,
        dim_head=8,
        ff_mult=1,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        static_chunk_size=4,
        num_decoding_left_chunks=-1,
    ).eval()
    return CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
        estimator=estimator,
    ).eval()
