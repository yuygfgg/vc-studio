from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosyvoice.flow.DiT.modules import DiTBlock
from cosyvoice.utils.mask import add_optional_chunk_mask


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


def test_prompt_kv_query_chunking_matches_full_attention() -> None:
    torch.manual_seed(11)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 4, 4)
    mask = torch.ones(2, 1, 5, 9, dtype=torch.bool)

    with torch.inference_mode():
        full = block.forward_with_prompt_kv(x.clone(), t, prompt_key, prompt_value, mask=mask)
        chunked = block.forward_with_prompt_kv(
            x.clone(),
            t,
            prompt_key,
            prompt_value,
            mask=mask,
            query_chunk_size=2,
        )

    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)


def test_prompt_cache_chunked_prefix_attention_matches_static_chunk_mask() -> None:
    torch.manual_seed(10)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 7, 8)
    t = torch.randn(2, 8)
    mask = add_optional_chunk_mask(
        x,
        torch.ones(2, 1, 7, dtype=torch.bool),
        False,
        False,
        0,
        3,
        -1,
    ).unsqueeze(1)

    with torch.inference_mode():
        masked, masked_key, masked_value = block.forward_return_kv(x.clone(), t, mask=mask)
        chunked, chunked_key, chunked_value = block.forward_return_kv(x.clone(), t, prefix_chunk_size=3)

    assert torch.allclose(chunked, masked, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_key, masked_key, atol=1e-6, rtol=1e-6)
    assert torch.allclose(chunked_value, masked_value, atol=1e-6, rtol=1e-6)


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


def test_grouped_prompt_attention_single_branch_with_history_matches_normal_prompt_kv() -> None:
    torch.manual_seed(2)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 4, 4)
    history_key = torch.randn(2, 2, 2, 4)
    history_value = torch.randn(2, 2, 2, 4)
    mask = torch.ones(2, 1, 3, 9, dtype=torch.bool)

    with torch.inference_mode():
        normal = block.forward_with_prompt_kv(
            x.clone(),
            t,
            torch.cat([prompt_key, history_key], dim=2),
            torch.cat([prompt_value, history_value], dim=2),
            mask=mask,
        )
        grouped = block.forward_with_grouped_prompt_kv(
            x.clone(),
            t,
            prompt_key.unsqueeze(0),
            prompt_value.unsqueeze(0),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1),
            mask=mask,
            history_key=history_key,
            history_value=history_value,
        )

    assert torch.allclose(grouped, normal, atol=1e-6, rtol=1e-6)


def test_sequential_grouped_prompt_attention_matches_vectorized_grouped_attention() -> None:
    torch.manual_seed(3)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 2, 4, 4)
    prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    branch_weights = torch.tensor([0.35, 0.65])
    mask = torch.ones(2, 1, 3, 7, dtype=torch.bool)

    with torch.inference_mode():
        vectorized = block.forward_with_grouped_prompt_kv(
            x.clone(),
            t,
            prompt_key,
            prompt_value,
            prompt_mask,
            branch_weights,
            mask=mask,
        )
        sequential = block.forward_with_sequential_grouped_prompt_kv(
            x.clone(),
            t,
            [
                (prompt_key[0], prompt_value[0], prompt_mask[0]),
                (prompt_key[1], prompt_value[1], prompt_mask[1]),
            ],
            branch_weights,
            mask=mask,
        )
        sequential_chunked = block.forward_with_sequential_grouped_prompt_kv(
            x.clone(),
            t,
            [
                (prompt_key[0], prompt_value[0], prompt_mask[0]),
                (prompt_key[1], prompt_value[1], prompt_mask[1]),
            ],
            branch_weights,
            mask=mask,
            query_chunk_size=2,
        )

    assert torch.allclose(sequential, vectorized, atol=1e-6, rtol=1e-6)
    assert torch.allclose(sequential_chunked, vectorized, atol=1e-6, rtol=1e-6)


def test_sequential_grouped_prompt_attention_with_history_matches_vectorized() -> None:
    torch.manual_seed(4)
    block = DiTBlock(dim=8, heads=2, dim_head=4, ff_mult=2, dropout=0.0).eval()
    x = torch.randn(2, 3, 8)
    t = torch.randn(2, 8)
    prompt_key = torch.randn(2, 2, 2, 4, 4)
    prompt_value = torch.randn(2, 2, 2, 4, 4)
    prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    branch_weights = torch.tensor([0.35, 0.65])
    history_key = torch.randn(2, 2, 2, 4)
    history_value = torch.randn(2, 2, 2, 4)
    mask = torch.ones(2, 1, 3, 9, dtype=torch.bool)

    with torch.inference_mode():
        vectorized = block.forward_with_grouped_prompt_kv(
            x.clone(),
            t,
            prompt_key,
            prompt_value,
            prompt_mask,
            branch_weights,
            mask=mask,
            history_key=history_key,
            history_value=history_value,
        )
        sequential = block.forward_with_sequential_grouped_prompt_kv(
            x.clone(),
            t,
            [
                (prompt_key[0], prompt_value[0], prompt_mask[0]),
                (prompt_key[1], prompt_value[1], prompt_mask[1]),
            ],
            branch_weights,
            mask=mask,
            history_key=history_key,
            history_value=history_value,
        )

    assert torch.allclose(sequential, vectorized, atol=1e-6, rtol=1e-6)
