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
