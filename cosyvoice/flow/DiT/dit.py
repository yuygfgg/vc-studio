
"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import repeat
from x_transformers.x_transformers import RotaryEmbedding
from cosyvoice.utils.mask import add_optional_chunk_mask
from cosyvoice.flow.DiT.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    CausalConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
    ):
        x = self.project_inputs(x, cond, text_embed, spks)
        x = self.conv_pos_embed(x) + x
        return x

    def forward_with_cache(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
            cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.project_inputs(x, cond, text_embed, spks)
        pos, cache = self.conv_pos_embed.forward_with_cache(x, cache=cache)
        return pos + x, cache

    def forward_return_cache(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.project_inputs(x, cond, text_embed, spks)
        pos, cache = self.conv_pos_embed.forward_return_cache(x)
        return pos + x, cache

    def project_inputs(
            self,
            x: float["b n d"],
            cond: float["b n d"],
            text_embed: float["b n d"],
            spks: float["b d"],
    ) -> torch.Tensor:
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)

        return self.proj(torch.cat(to_cat, dim=-1))


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(
        self,
        x,
        mask,
        mu,
        t,
        spks=None,
        cond=None,
        streaming=False,
        prompt_len=0,
        source_mel_offset=0,
    ):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        positions = self.source_window_positions(seq_len, prompt_len, source_mel_offset, x.device)
        rope = self.rotary_embed(positions)

        if self.long_skip_connection is not None:
            residual = x

        if streaming is True:
            attn_mask = self.global_chunk_mask(mask.bool(), positions)
        else:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1).unsqueeze(dim=1)

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output

    def source_window_positions(self, seq_len, prompt_len=0, source_mel_offset=0, device=None):
        prompt_len = min(max(int(prompt_len), 0), int(seq_len))
        source_mel_offset = max(int(source_mel_offset or 0), 0)
        source_len = int(seq_len) - prompt_len
        prompt_pos = torch.arange(prompt_len, device=device)
        source_pos = torch.arange(source_len, device=device) + prompt_len + source_mel_offset
        return torch.cat([prompt_pos, source_pos], dim=0)

    def cached_source_positions(self, seq_len, prompt_len, base_prompt_len, source_mel_offset=0, device=None):
        prompt_len = min(max(int(prompt_len), 0), int(seq_len))
        base_prompt_len = min(max(int(base_prompt_len), 0), prompt_len)
        source_mel_offset = max(int(source_mel_offset or 0), 0)
        source_len = int(seq_len) - prompt_len
        history_len = prompt_len - base_prompt_len
        prompt_pos = torch.arange(base_prompt_len, device=device)
        source_start = base_prompt_len + source_mel_offset
        history_start = max(base_prompt_len, source_start - history_len)
        history_pos = torch.arange(history_start, source_start, device=device)
        source_pos = torch.arange(source_len, device=device) + source_start
        return torch.cat([prompt_pos, history_pos, source_pos], dim=0)

    def global_chunk_mask(self, mask, positions):
        if self.static_chunk_size <= 0:
            return mask.repeat(1, positions.numel(), 1).unsqueeze(dim=1)
        block_end = (torch.div(positions, self.static_chunk_size, rounding_mode='trunc') + 1) * self.static_chunk_size
        chunk_mask = positions.unsqueeze(0) < block_end.unsqueeze(1)
        return (mask & chunk_mask.unsqueeze(0)).unsqueeze(dim=1)

    def source_attention_query_chunk_size(self, query_len, prompt_len):
        if int(prompt_len) <= 0 or int(query_len) <= 32:
            return None
        if self.static_chunk_size > 0:
            return max(1, min(32, int(self.static_chunk_size)))
        return 32

    def forward_prompt_cache(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x, input_embed_cache = self.input_embed.forward_return_cache(x, cond, mu, spks.squeeze(1))
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        prompt_mask = mask.bool()
        prefix_chunk_size = None
        if streaming is True and self.static_chunk_size > 0 and bool(prompt_mask.all().item()):
            prefix_chunk_size = self.static_chunk_size

        if prefix_chunk_size is not None:
            attn_mask = None
        elif streaming is True:
            attn_mask = add_optional_chunk_mask(
                x,
                prompt_mask,
                False,
                False,
                0,
                self.static_chunk_size,
                -1,
            ).unsqueeze(dim=1)
        else:
            attn_mask = (
                add_optional_chunk_mask(x, prompt_mask, False, False, 0, 0, -1)
                .repeat(1, x.size(1), 1)
                .unsqueeze(dim=1)
            )

        cache = []
        for block in self.transformer_blocks:
            x, key, value = block.forward_return_kv(
                x,
                t,
                mask=attn_mask.bool() if attn_mask is not None else None,
                rope=rope,
                prefix_chunk_size=prefix_chunk_size,
            )
            cache.append((key, value))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output, {
            "kv": cache,
            "prompt_len": seq_len,
            "base_prompt_len": seq_len,
            "input_embed_cache": input_embed_cache,
        }

    def extend_input_embed_cache(
        self,
        cache: dict[str, torch.Tensor] | None,
        x,
        mu,
        cond,
        spks,
    ) -> dict[str, torch.Tensor] | None:
        if cache is None:
            return None
        if x.shape[2] == 0:
            return cache
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        if spks.ndim == 3:
            spks = spks.squeeze(1)
        _, cache = self.input_embed.forward_with_cache(x, cond, mu, spks, cache=cache)
        return cache

    def forward_source_with_prompt_cache(
        self,
        x,
        mask,
        mu,
        t,
        spks=None,
        cond=None,
        prompt_x=None,
        prompt_mu=None,
        prompt_cond=None,
        prompt_cache=None,
        streaming=False,
        return_source_cache=False,
        source_mel_offset=0,
    ):
        prompt_len = prompt_cache["prompt_len"]
        base_prompt_len = prompt_cache.get("base_prompt_len", prompt_len)
        x_len = x.shape[2]
        spks = spks.unsqueeze(dim=1)
        batch = x.shape[0]
        seq_len = prompt_len + x_len
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        input_embed_cache = prompt_cache.get("input_embed_cache")
        if input_embed_cache is None:
            x_full = torch.cat([prompt_x, x], dim=2).transpose(1, 2)
            mu_full = torch.cat([prompt_mu, mu], dim=2).transpose(1, 2)
            cond_full = torch.cat([prompt_cond, cond], dim=2).transpose(1, 2)
            x = self.input_embed(x_full, cond_full, mu_full, spks.squeeze(1))[:, prompt_len:]
        else:
            x, _ = self.input_embed.forward_with_cache(
                x.transpose(1, 2),
                cond.transpose(1, 2),
                mu.transpose(1, 2),
                spks.squeeze(1),
                cache=input_embed_cache,
            )
        source_position_base = int(base_prompt_len) + max(int(source_mel_offset), 0)
        source_positions = torch.arange(x_len, device=x.device) + source_position_base
        rope = self.rotary_embed(source_positions)
        positions = self.cached_source_positions(seq_len, prompt_len, base_prompt_len, source_mel_offset, x.device)

        if streaming is True:
            attn_mask = self.global_chunk_mask(mask.bool(), positions)
        else:
            attn_mask = mask.bool().repeat(1, seq_len, 1).unsqueeze(dim=1)
        attn_mask = attn_mask[:, :, prompt_len:, :]
        source_query_chunk_size = self.source_attention_query_chunk_size(x_len, prompt_len)

        source_cache = []
        if prompt_cache.get("grouped_branch_attention"):
            branch_weights = prompt_cache["branch_weights"]
            attention_temperature = float(prompt_cache.get("attention_temperature", 1.0))
            history_kv = prompt_cache.get("history_kv")
            if prompt_cache.get("grouped_attention_mode") == "sequential":
                sequential_branch_caches = prompt_cache["sequential_branch_caches"]
                for block_index, block in enumerate(self.transformer_blocks):
                    history_pair = history_kv[block_index] if history_kv is not None else None
                    history_key, history_value = history_pair if history_pair is not None else (None, None)
                    branch_kv = [
                        (
                            branch_cache["kv"][block_index][0],
                            branch_cache["kv"][block_index][1],
                            branch_cache["prompt_mask"],
                        )
                        for branch_cache in sequential_branch_caches
                    ]
                    if return_source_cache:
                        x, source_key, source_value = block.forward_with_sequential_grouped_prompt_kv(
                            x,
                            t,
                            branch_kv,
                            branch_weights,
                            mask=attn_mask.bool(),
                            rope=rope,
                            return_kv=True,
                            attention_temperature=attention_temperature,
                            history_key=history_key,
                            history_value=history_value,
                            query_chunk_size=source_query_chunk_size,
                        )
                        source_cache.append((source_key, source_value))
                    else:
                        x = block.forward_with_sequential_grouped_prompt_kv(
                            x,
                            t,
                            branch_kv,
                            branch_weights,
                            mask=attn_mask.bool(),
                            rope=rope,
                            attention_temperature=attention_temperature,
                            history_key=history_key,
                            history_value=history_value,
                            query_chunk_size=source_query_chunk_size,
                        )
            else:
                grouped_kv = prompt_cache["grouped_kv"]
                grouped_prompt_mask = prompt_cache["grouped_prompt_mask"]
                if history_kv is None:
                    grouped_kv_iter = ((block, prompt_kv, None) for block, prompt_kv in zip(self.transformer_blocks, grouped_kv))
                else:
                    grouped_kv_iter = (
                        (block, prompt_kv, hist_kv)
                        for block, prompt_kv, hist_kv in zip(self.transformer_blocks, grouped_kv, history_kv)
                    )
                for block, (prompt_key, prompt_value), history_pair in grouped_kv_iter:
                    history_key, history_value = history_pair if history_pair is not None else (None, None)
                    if return_source_cache:
                        x, source_key, source_value = block.forward_with_grouped_prompt_kv(
                            x,
                            t,
                            prompt_key,
                            prompt_value,
                            grouped_prompt_mask,
                            branch_weights,
                            mask=attn_mask.bool(),
                            rope=rope,
                            return_kv=True,
                            attention_temperature=attention_temperature,
                            history_key=history_key,
                            history_value=history_value,
                            query_chunk_size=source_query_chunk_size,
                        )
                        source_cache.append((source_key, source_value))
                    else:
                        x = block.forward_with_grouped_prompt_kv(
                            x,
                            t,
                            prompt_key,
                            prompt_value,
                            grouped_prompt_mask,
                            branch_weights,
                            mask=attn_mask.bool(),
                            rope=rope,
                            attention_temperature=attention_temperature,
                            history_key=history_key,
                            history_value=history_value,
                            query_chunk_size=source_query_chunk_size,
                        )

            x = self.norm_out(x, t)
            output = self.proj_out(x).transpose(1, 2)
            assert output.shape[2] == x_len
            if return_source_cache:
                return output, {"kv": source_cache, "source_len": x_len}
            return output, None

        history_kv = prompt_cache.get("history_kv")
        if history_kv is None:
            prompt_kv_iter = zip(self.transformer_blocks, prompt_cache["kv"])
        else:
            prompt_kv_iter = (
                (
                    block,
                    (
                        torch.cat([base_key.to(device=hist_key.device, dtype=hist_key.dtype), hist_key], dim=2),
                        torch.cat([base_value.to(device=hist_value.device, dtype=hist_value.dtype), hist_value], dim=2),
                    ),
                )
                for block, (base_key, base_value), (hist_key, hist_value) in zip(self.transformer_blocks, prompt_cache["kv"], history_kv)
            )

        for block, (prompt_key, prompt_value) in prompt_kv_iter:
            if return_source_cache:
                x, source_key, source_value = block.forward_with_prompt_kv(
                    x,
                    t,
                    prompt_key,
                    prompt_value,
                    mask=attn_mask.bool(),
                    rope=rope,
                    return_kv=True,
                    query_chunk_size=source_query_chunk_size,
                )
                source_cache.append((source_key, source_value))
            else:
                x = block.forward_with_prompt_kv(
                    x,
                    t,
                    prompt_key,
                    prompt_value,
                    mask=attn_mask.bool(),
                    rope=rope,
                    query_chunk_size=source_query_chunk_size,
                )

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        assert output.shape[2] == x_len
        if return_source_cache:
            return output, {"kv": source_cache, "source_len": x_len}
        return output, None

    def forward_grouped_prompt_source_uncached(
        self,
        branch_prompt_x,
        branch_prompt_mu,
        branch_prompt_cond,
        branch_spks,
        source_x,
        source_mu,
        t,
        source_spks,
        source_cond,
        branch_weights,
        dominant_branch_position=0,
        streaming=False,
        source_mel_offset=0,
        attention_temperature=1.0,
        source_position_prompt_len=None,
    ):
        if not branch_prompt_x:
            raise ValueError("grouped prompt inputs have no active branches")

        source_len = source_x.shape[2]
        branch_count = len(branch_prompt_x)
        branch_weights = branch_weights.to(device=source_x.device, dtype=source_x.dtype)
        branch_hidden = []
        branch_t = []
        branch_masks = []
        branch_ropes = []
        branch_input_embed_caches = []

        for prompt_x, prompt_mu, prompt_cond, prompt_spk in zip(
            branch_prompt_x,
            branch_prompt_mu,
            branch_prompt_cond,
            branch_spks,
        ):
            prompt_len = prompt_x.shape[2]
            batch = prompt_x.size(0)
            x_prompt_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_mu_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_cond_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_mask_in = torch.ones([2 * batch, 1, prompt_len], device=prompt_x.device, dtype=torch.bool)
            t_in = torch.zeros([2 * batch], device=prompt_x.device, dtype=source_spks.dtype)
            spks_in = torch.zeros([2 * batch, 80], device=prompt_x.device, dtype=source_spks.dtype)

            x_prompt_in[:] = prompt_x
            prompt_mu_in[:batch] = prompt_mu
            prompt_cond_in[:batch] = prompt_cond
            t_in[:] = t.reshape(1)
            spks_in[:batch] = prompt_spk

            hidden, input_embed_cache = self.input_embed.forward_return_cache(
                x_prompt_in.transpose(1, 2),
                prompt_cond_in.transpose(1, 2),
                prompt_mu_in.transpose(1, 2),
                spks_in,
            )
            t_emb = self.time_embed(t_in)
            rope = self.rotary_embed.forward_from_seq_len(prompt_len)
            if streaming is True:
                attn_mask = add_optional_chunk_mask(
                    hidden,
                    prompt_mask_in.bool(),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                ).unsqueeze(dim=1)
            else:
                attn_mask = add_optional_chunk_mask(
                    hidden,
                    prompt_mask_in.bool(),
                    False,
                    False,
                    0,
                    0,
                    -1,
                ).repeat(1, hidden.size(1), 1).unsqueeze(dim=1)
            branch_hidden.append(hidden)
            branch_t.append(t_emb)
            branch_masks.append(attn_mask.bool())
            branch_ropes.append(rope)
            branch_input_embed_caches.append(input_embed_cache)

        dominant_branch_position = max(0, min(int(dominant_branch_position), branch_count - 1))
        prompt_len = branch_prompt_x[dominant_branch_position].shape[2]
        position_prompt_len = prompt_len if source_position_prompt_len is None else int(source_position_prompt_len)
        position_prompt_len = max(prompt_len, position_prompt_len)
        source_batch = source_x.size(0)
        x_source_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_mu_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_cond_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        full_mask_in = torch.ones([2 * source_batch, 1, position_prompt_len + source_len], device=source_x.device, dtype=torch.bool)
        t_in = torch.zeros([2 * source_batch], device=source_x.device, dtype=source_spks.dtype)
        spks_in = torch.zeros([2 * source_batch, 80], device=source_x.device, dtype=source_spks.dtype)
        x_source_in[:] = source_x
        source_mu_in[:source_batch] = source_mu
        source_cond_in[:source_batch] = source_cond
        t_in[:] = t.reshape(1)
        spks_in[:source_batch] = source_spks
        source_t = self.time_embed(t_in)
        source_hidden, _ = self.input_embed.forward_with_cache(
            x_source_in.transpose(1, 2),
            source_cond_in.transpose(1, 2),
            source_mu_in.transpose(1, 2),
            spks_in,
            cache=branch_input_embed_caches[dominant_branch_position],
        )

        source_position_base = int(position_prompt_len) + max(int(source_mel_offset), 0)
        source_positions = torch.arange(source_len, device=source_hidden.device) + source_position_base
        source_rope = self.rotary_embed(source_positions)
        positions = self.cached_source_positions(
            position_prompt_len + source_len,
            position_prompt_len,
            position_prompt_len,
            source_mel_offset,
            source_hidden.device,
        )
        if streaming is True:
            source_attn_mask = self.global_chunk_mask(full_mask_in.bool(), positions)
        else:
            source_attn_mask = full_mask_in.bool().repeat(1, position_prompt_len + source_len, 1).unsqueeze(dim=1)
        source_attn_mask = source_attn_mask[:, :, position_prompt_len:, :]

        for block in self.transformer_blocks:
            prompt_keys = []
            prompt_values = []
            for branch_index in range(branch_count):
                hidden, key, value = block.forward_return_kv(
                    branch_hidden[branch_index],
                    branch_t[branch_index],
                    mask=branch_masks[branch_index],
                    rope=branch_ropes[branch_index],
                )
                branch_hidden[branch_index] = hidden
                prompt_keys.append(key)
                prompt_values.append(value)
            grouped_key, grouped_value, grouped_mask = self._stack_current_grouped_prompt_kv(prompt_keys, prompt_values)
            source_hidden = block.forward_with_grouped_prompt_kv(
                source_hidden,
                source_t,
                grouped_key,
                grouped_value,
                grouped_mask,
                branch_weights,
                mask=source_attn_mask.bool(),
                rope=source_rope,
                attention_temperature=attention_temperature,
            )

        prompt_outputs = [
            self.proj_out(self.norm_out(hidden, branch_t[index])).transpose(1, 2)
            for index, hidden in enumerate(branch_hidden)
        ]
        source_output = self.proj_out(self.norm_out(source_hidden, source_t)).transpose(1, 2)
        assert source_output.shape[2] == source_len
        return prompt_outputs, source_output

    def forward_soft_prompt_source_until_layer(
        self,
        prompt_x,
        prompt_mu,
        prompt_cond,
        prompt_spks,
        source_x,
        source_mu,
        t,
        source_spks,
        source_cond,
        distill_layer=6,
        streaming=False,
        source_mel_offset=0,
        checkpoint_segments=0,
    ):
        prompt_len = prompt_x.shape[2]
        source_len = source_x.shape[2]
        source_batch = source_x.size(0)
        layer_count = min(max(int(distill_layer), 1), len(self.transformer_blocks))

        x_prompt_in = torch.zeros([2 * source_batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
        prompt_mu_in = torch.zeros([2 * source_batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
        prompt_cond_in = torch.zeros([2 * source_batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
        prompt_mask_in = torch.ones([2 * source_batch, 1, prompt_len], device=prompt_x.device, dtype=torch.bool)
        x_source_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_mu_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_cond_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        full_mask_in = torch.ones([2 * source_batch, 1, prompt_len + source_len], device=source_x.device, dtype=torch.bool)
        t_in = torch.zeros([2 * source_batch], device=source_x.device, dtype=source_spks.dtype)
        prompt_spks_in = torch.zeros([2 * source_batch, 80], device=prompt_x.device, dtype=source_spks.dtype)
        source_spks_in = torch.zeros([2 * source_batch, 80], device=source_x.device, dtype=source_spks.dtype)

        x_prompt_in[:] = prompt_x
        prompt_mu_in[:source_batch] = prompt_mu
        prompt_cond_in[:source_batch] = prompt_cond
        x_source_in[:] = source_x
        source_mu_in[:source_batch] = source_mu
        source_cond_in[:source_batch] = source_cond
        t_in[:] = t.reshape(1)
        prompt_spks_in[:source_batch] = prompt_spks
        source_spks_in[:source_batch] = source_spks

        prompt_hidden, input_embed_cache = self.input_embed.forward_return_cache(
            x_prompt_in.transpose(1, 2),
            prompt_cond_in.transpose(1, 2),
            prompt_mu_in.transpose(1, 2),
            prompt_spks_in,
        )
        source_hidden, _ = self.input_embed.forward_with_cache(
            x_source_in.transpose(1, 2),
            source_cond_in.transpose(1, 2),
            source_mu_in.transpose(1, 2),
            source_spks_in,
            cache=input_embed_cache,
        )
        t_emb = self.time_embed(t_in)
        prompt_rope = self.rotary_embed.forward_from_seq_len(prompt_len)
        source_position_base = int(prompt_len) + max(int(source_mel_offset), 0)
        source_positions = torch.arange(source_len, device=source_hidden.device) + source_position_base
        source_rope = self.rotary_embed(source_positions)
        positions = self.cached_source_positions(
            prompt_len + source_len,
            prompt_len,
            prompt_len,
            source_mel_offset,
            source_hidden.device,
        )

        if streaming is True:
            if self.static_chunk_size > 0 and bool(prompt_mask_in.all().item()):
                prompt_attn_mask = None
                prompt_prefix_chunk_size = self.static_chunk_size
            else:
                prompt_attn_mask = add_optional_chunk_mask(
                    prompt_hidden,
                    prompt_mask_in.bool(),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                ).unsqueeze(dim=1)
                prompt_prefix_chunk_size = None
            source_attn_mask = self.global_chunk_mask(full_mask_in.bool(), positions)
        else:
            prompt_attn_mask = add_optional_chunk_mask(
                prompt_hidden,
                prompt_mask_in.bool(),
                False,
                False,
                0,
                0,
                -1,
            ).repeat(1, prompt_hidden.size(1), 1).unsqueeze(dim=1)
            prompt_prefix_chunk_size = None
            source_attn_mask = full_mask_in.bool().repeat(1, prompt_len + source_len, 1).unsqueeze(dim=1)
        source_attn_mask = source_attn_mask[:, :, prompt_len:, :]
        query_chunk_size = self.source_attention_query_chunk_size(source_len, prompt_len)

        def run_layers(start, end, current_prompt_hidden, current_source_hidden):
            for block in self.transformer_blocks[start:end]:
                current_prompt_hidden, prompt_key, prompt_value = block.forward_return_kv(
                    current_prompt_hidden,
                    t_emb,
                    mask=prompt_attn_mask.bool() if prompt_attn_mask is not None else None,
                    rope=prompt_rope,
                    prefix_chunk_size=prompt_prefix_chunk_size,
                )
                current_source_hidden = block.forward_with_prompt_kv(
                    current_source_hidden,
                    t_emb,
                    prompt_key,
                    prompt_value,
                    mask=source_attn_mask.bool(),
                    rope=source_rope,
                    query_chunk_size=query_chunk_size,
                )
            return current_prompt_hidden, current_source_hidden

        segments = max(0, int(checkpoint_segments or 0))
        if segments > 0 and torch.is_grad_enabled():
            segment_size = max(1, (layer_count + segments - 1) // segments)
            start = 0
            while start < layer_count:
                end = min(layer_count, start + segment_size)
                prompt_hidden, source_hidden = checkpoint(
                    lambda prompt_arg, source_arg, layer_start=start, layer_end=end: run_layers(
                        layer_start,
                        layer_end,
                        prompt_arg,
                        source_arg,
                    ),
                    prompt_hidden,
                    source_hidden,
                    use_reentrant=False,
                )
                start = end
        else:
            prompt_hidden, source_hidden = run_layers(0, layer_count, prompt_hidden, source_hidden)

        return source_hidden

    def forward_grouped_prompt_source_until_layer(
        self,
        branch_prompt_x,
        branch_prompt_mu,
        branch_prompt_cond,
        branch_spks,
        source_x,
        source_mu,
        t,
        source_spks,
        source_cond,
        branch_weights,
        distill_layer=6,
        dominant_branch_position=0,
        streaming=False,
        source_mel_offset=0,
        attention_temperature=1.0,
        source_position_prompt_len=None,
    ):
        if not branch_prompt_x:
            raise ValueError("grouped prompt inputs have no active branches")

        source_len = source_x.shape[2]
        branch_count = len(branch_prompt_x)
        layer_count = min(max(int(distill_layer), 1), len(self.transformer_blocks))
        branch_weights = branch_weights.to(device=source_x.device, dtype=source_x.dtype)
        branch_hidden = []
        branch_t = []
        branch_masks = []
        branch_ropes = []
        branch_prompt_masks = []
        branch_input_embed_caches = []

        for prompt_x, prompt_mu, prompt_cond, prompt_spk in zip(
            branch_prompt_x,
            branch_prompt_mu,
            branch_prompt_cond,
            branch_spks,
        ):
            prompt_len = prompt_x.shape[2]
            batch = prompt_x.size(0)
            x_prompt_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_mu_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_cond_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=source_spks.dtype)
            prompt_mask_in = torch.ones([2 * batch, 1, prompt_len], device=prompt_x.device, dtype=torch.bool)
            t_in = torch.zeros([2 * batch], device=prompt_x.device, dtype=source_spks.dtype)
            spks_in = torch.zeros([2 * batch, 80], device=prompt_x.device, dtype=source_spks.dtype)

            x_prompt_in[:] = prompt_x
            prompt_mu_in[:batch] = prompt_mu
            prompt_cond_in[:batch] = prompt_cond
            t_in[:] = t.reshape(1)
            spks_in[:batch] = prompt_spk

            hidden, input_embed_cache = self.input_embed.forward_return_cache(
                x_prompt_in.transpose(1, 2),
                prompt_cond_in.transpose(1, 2),
                prompt_mu_in.transpose(1, 2),
                spks_in,
            )
            t_emb = self.time_embed(t_in)
            rope = self.rotary_embed.forward_from_seq_len(prompt_len)
            if streaming is True:
                attn_mask = add_optional_chunk_mask(
                    hidden,
                    prompt_mask_in.bool(),
                    False,
                    False,
                    0,
                    self.static_chunk_size,
                    -1,
                ).unsqueeze(dim=1)
            else:
                attn_mask = add_optional_chunk_mask(
                    hidden,
                    prompt_mask_in.bool(),
                    False,
                    False,
                    0,
                    0,
                    -1,
                ).repeat(1, hidden.size(1), 1).unsqueeze(dim=1)
            branch_hidden.append(hidden)
            branch_t.append(t_emb)
            branch_masks.append(attn_mask.bool())
            branch_ropes.append(rope)
            branch_prompt_masks.append(torch.ones(prompt_len, dtype=torch.bool, device=prompt_x.device))
            branch_input_embed_caches.append(input_embed_cache)

        dominant_branch_position = max(0, min(int(dominant_branch_position), branch_count - 1))
        prompt_len = branch_prompt_x[dominant_branch_position].shape[2]
        position_prompt_len = prompt_len if source_position_prompt_len is None else int(source_position_prompt_len)
        position_prompt_len = max(prompt_len, position_prompt_len)
        source_batch = source_x.size(0)
        x_source_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_mu_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        source_cond_in = torch.zeros([2 * source_batch, 80, source_len], device=source_x.device, dtype=source_spks.dtype)
        full_mask_in = torch.ones([2 * source_batch, 1, position_prompt_len + source_len], device=source_x.device, dtype=torch.bool)
        t_in = torch.zeros([2 * source_batch], device=source_x.device, dtype=source_spks.dtype)
        spks_in = torch.zeros([2 * source_batch, 80], device=source_x.device, dtype=source_spks.dtype)
        x_source_in[:] = source_x
        source_mu_in[:source_batch] = source_mu
        source_cond_in[:source_batch] = source_cond
        t_in[:] = t.reshape(1)
        spks_in[:source_batch] = source_spks
        source_t = self.time_embed(t_in)
        source_hidden, _ = self.input_embed.forward_with_cache(
            x_source_in.transpose(1, 2),
            source_cond_in.transpose(1, 2),
            source_mu_in.transpose(1, 2),
            spks_in,
            cache=branch_input_embed_caches[dominant_branch_position],
        )

        source_position_base = int(position_prompt_len) + max(int(source_mel_offset), 0)
        source_positions = torch.arange(source_len, device=source_hidden.device) + source_position_base
        source_rope = self.rotary_embed(source_positions)
        positions = self.cached_source_positions(
            position_prompt_len + source_len,
            position_prompt_len,
            position_prompt_len,
            source_mel_offset,
            source_hidden.device,
        )
        if streaming is True:
            source_attn_mask = self.global_chunk_mask(full_mask_in.bool(), positions)
        else:
            source_attn_mask = full_mask_in.bool().repeat(1, position_prompt_len + source_len, 1).unsqueeze(dim=1)
        source_attn_mask = source_attn_mask[:, :, position_prompt_len:, :]
        query_chunk_size = self.source_attention_query_chunk_size(source_len, position_prompt_len)

        for block in self.transformer_blocks[:layer_count]:
            branch_kv = []
            for branch_index in range(branch_count):
                hidden, key, value = block.forward_return_kv(
                    branch_hidden[branch_index],
                    branch_t[branch_index],
                    mask=branch_masks[branch_index],
                    rope=branch_ropes[branch_index],
                )
                branch_hidden[branch_index] = hidden
                branch_kv.append((key, value, branch_prompt_masks[branch_index]))
            source_hidden = block.forward_with_sequential_grouped_prompt_kv(
                source_hidden,
                source_t,
                branch_kv,
                branch_weights,
                mask=source_attn_mask.bool(),
                rope=source_rope,
                attention_temperature=attention_temperature,
                query_chunk_size=query_chunk_size,
            )
            del branch_kv

        return source_hidden

    def _stack_current_grouped_prompt_kv(self, keys, values):
        max_len = max(key.shape[2] for key in keys)
        first_key = keys[0]
        first_value = values[0]
        key_stack = first_key.new_zeros((len(keys), first_key.shape[0], first_key.shape[1], max_len, first_key.shape[3]))
        value_stack = first_value.new_zeros(
            (len(values), first_value.shape[0], first_value.shape[1], max_len, first_value.shape[3])
        )
        mask = torch.zeros((len(keys), max_len), dtype=torch.bool, device=first_key.device)
        for branch_pos, (key, value) in enumerate(zip(keys, values)):
            length = key.shape[2]
            key_stack[branch_pos, :, :, :length] = key
            value_stack[branch_pos, :, :, :length] = value
            mask[branch_pos, :length] = True
        return key_stack, value_stack, mask
