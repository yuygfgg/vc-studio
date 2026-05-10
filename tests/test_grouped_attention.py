from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosyvoice.flow.DiT.modules import DiTBlock


def test_grouped_prompt_attention_single_branch_matches_normal_prompt_kv() -> None:
    torch.manual_seed(0)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 4, 4)
    mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)

    with torch.inference_mode():
        normal = block.forward_with_prompt_kv(x.clone(), t, prompt_key, prompt_value, mask=mask)
        grouped = block.forward_with_grouped_prompt_kv(
            x.clone(),
            t,
            prompt_key.unsqueeze(0),
            prompt_value.unsqueeze(0),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1),
            mask=mask,
        )

    assert torch.allclose(grouped, normal, atol=1e-6, rtol=1e-6)


def test_grouped_prompt_attention_one_hot_selects_matching_branch() -> None:
    torch.manual_seed(1)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 2, 4, 4)
    mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)

    with torch.inference_mode():
        branch_1 = block.forward_with_prompt_kv(x.clone(), t, prompt_key[1], prompt_value[1], mask=mask)
        grouped = block.forward_with_grouped_prompt_kv(
            x.clone(),
            t,
            prompt_key,
            prompt_value,
            torch.ones(2, 4, dtype=torch.bool),
            torch.tensor([0.0, 1.0]),
            mask=mask,
        )

    assert torch.allclose(grouped, branch_1, atol=1e-6, rtol=1e-6)
