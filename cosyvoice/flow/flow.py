# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize,
                  use_prompt_cache=False,
                  prompt_cache_steps=None,
                  grouped_prompt_inputs=None,
                  soft_prompt_inputs=None,
                  prompt_cache_len=0,
                  source_cache_len=0,
                  source_cache_end=0,
                  source_mel_offset=0):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        if soft_prompt_inputs is not None:
            return self._inference_soft_prompt(
                token=token,
                token_len=token_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
                soft_prompt_inputs=soft_prompt_inputs,
                use_prompt_cache=use_prompt_cache,
                prompt_cache_steps=prompt_cache_steps,
                prompt_cache_len=prompt_cache_len,
                source_cache_len=source_cache_len,
                source_cache_end=source_cache_end,
                source_mel_offset=source_mel_offset,
            )

        base_prompt_mel_len = prompt_feat.shape[1]
        if grouped_prompt_inputs is not None and prompt_cache_steps is None:
            return self._inference_grouped_prompt_uncached(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=streaming,
                finalize=finalize,
                grouped_prompt_inputs=grouped_prompt_inputs,
                source_mel_offset=source_mel_offset,
            )

        decoder_prompt_len = prompt_cache_len if prompt_cache_steps is not None else base_prompt_mel_len
        cached_history_mel = max(0, decoder_prompt_len - base_prompt_mel_len)
        if (
            prompt_cache_steps is not None
            and isinstance(prompt_cache_steps, dict)
            and prompt_cache_steps.get("flow_token_tail") is not None
            and cached_history_mel % self.token_mel_ratio == 0
        ):
            cached_history_tokens = cached_history_mel // self.token_mel_ratio
            has_source_context = finalize is True or token.shape[1] - cached_history_tokens > self.pre_lookahead_len
            if token.shape[1] >= cached_history_tokens and has_source_context:
                source_token = token[:, cached_history_tokens:]
                h, source_token_embed = self._source_conditioning_from_tail(
                    source_token,
                    prompt_cache_steps["flow_token_tail"],
                    finalize=finalize,
                )
                source_mu = h.transpose(1, 2).contiguous()
                source_cond = torch.zeros(
                    [1, self.output_size, h.shape[1]],
                    device=token.device,
                    dtype=h.dtype,
                )
                feat, updated_cache = self.decoder.forward_prepared_prompt_cache_source(
                    source_mu=source_mu,
                    source_cond=source_cond,
                    n_timesteps=10,
                    spks=embedding,
                    streaming=streaming,
                    prompt_cache_steps=prompt_cache_steps,
                    source_cache_len=source_cache_len,
                    source_cache_end=source_cache_end,
                    source_mel_offset=source_mel_offset,
                )
                self._attach_flow_conditioning_tail(
                    updated_cache,
                    prompt_cache_steps,
                    source_token_embed,
                    source_cache_len,
                    source_cache_end,
                )
                feat = feat[:, :, base_prompt_mel_len:]
                assert feat.shape[2] == cached_history_mel + h.shape[1]
                return feat.float(), updated_cache

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = base_prompt_mel_len, h.shape[1] - base_prompt_mel_len

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, updated_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
            prompt_len=decoder_prompt_len,
            use_prompt_cache=use_prompt_cache,
            prompt_cache_steps=prompt_cache_steps,
            source_cache_len=source_cache_len,
            source_cache_end=source_cache_end,
            source_mel_offset=source_mel_offset,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), updated_cache

    def _inference_soft_prompt(
            self,
            token,
            token_len,
            embedding,
            streaming,
            finalize,
            soft_prompt_inputs,
            use_prompt_cache=False,
            prompt_cache_steps=None,
            prompt_cache_len=0,
            source_cache_len=0,
            source_cache_end=0,
            source_mel_offset=0):
        soft_prompt_mu = soft_prompt_inputs["soft_prompt_mu"].to(device=token.device)
        soft_prompt_feat = soft_prompt_inputs["soft_prompt_feat"].to(device=token.device)
        base_prompt_mel_len = soft_prompt_mu.shape[1]
        if soft_prompt_feat.shape[1] != base_prompt_mel_len:
            raise RuntimeError("soft_prompt_mu and soft_prompt_feat must have the same mel length")

        decoder_prompt_len = prompt_cache_len if prompt_cache_steps is not None else base_prompt_mel_len
        cached_history_mel = max(0, decoder_prompt_len - base_prompt_mel_len)
        if cached_history_mel > 0 and cached_history_mel % self.token_mel_ratio == 0:
            cached_history_tokens = cached_history_mel // self.token_mel_ratio
            if token.shape[1] >= cached_history_tokens:
                token = token[:, cached_history_tokens:]
                token_len = torch.clamp(token_len - cached_history_tokens, min=0)

        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token_embed = self.input_embedding(torch.clamp(token, min=0)) * mask
        if finalize is True:
            h = self.pre_lookahead_layer(token_embed)
        else:
            h = self.pre_lookahead_layer(
                token_embed[:, :-self.pre_lookahead_len],
                context=token_embed[:, -self.pre_lookahead_len:],
            )
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        if prompt_cache_steps is not None:
            source_mu = h.transpose(1, 2).contiguous()
            source_cond = torch.zeros(
                [1, self.output_size, h.shape[1]],
                device=token.device,
                dtype=h.dtype,
            )
            feat, updated_cache = self.decoder.forward_prepared_prompt_cache_source(
                source_mu=source_mu,
                source_cond=source_cond,
                n_timesteps=10,
                spks=embedding,
                streaming=streaming,
                prompt_cache_steps=prompt_cache_steps,
                source_cache_len=source_cache_len,
                source_cache_end=source_cache_end,
                source_mel_offset=source_mel_offset,
            )
            feat = feat[:, :, base_prompt_mel_len:]
            assert feat.shape[2] == decoder_prompt_len - base_prompt_mel_len + h.shape[1]
            return feat.float(), updated_cache

        total_mel_len = decoder_prompt_len + h.shape[1]
        mu = torch.zeros([1, total_mel_len, self.output_size], device=token.device, dtype=h.dtype)
        conds = torch.zeros([1, total_mel_len, self.output_size], device=token.device, dtype=h.dtype)
        if prompt_cache_steps is None:
            mu[:, :base_prompt_mel_len] = soft_prompt_mu.to(dtype=h.dtype)
            conds[:, :base_prompt_mel_len] = soft_prompt_feat.to(dtype=h.dtype)
        mu[:, decoder_prompt_len:] = h
        conds = conds.transpose(1, 2)
        full_mask = torch.ones([1, total_mel_len], device=token.device, dtype=torch.bool)

        feat, updated_cache = self.decoder(
            mu=mu.transpose(1, 2).contiguous(),
            mask=full_mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
            prompt_len=decoder_prompt_len,
            use_prompt_cache=use_prompt_cache,
            prompt_cache_steps=prompt_cache_steps,
            source_cache_len=source_cache_len,
            source_cache_end=source_cache_end,
            source_mel_offset=source_mel_offset,
        )
        feat = feat[:, :, base_prompt_mel_len:]
        assert feat.shape[2] == decoder_prompt_len - base_prompt_mel_len + h.shape[1]
        return feat.float(), updated_cache

    def _inference_grouped_prompt_uncached(
            self,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming,
            finalize,
            grouped_prompt_inputs,
            source_mel_offset=0):
        base_prompt_mel_len = prompt_feat.shape[1]
        grouped_prompt_uncached = self._prepare_grouped_prompt_uncached_inputs(grouped_prompt_inputs)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = base_prompt_mel_len, h.shape[1] - base_prompt_mel_len
        if mel_len2 < 0:
            raise RuntimeError("grouped prompt source window is shorter than the dominant prompt")

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat[:, :mel_len1].to(device=token.device, dtype=h.dtype)
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2], device=token.device))).to(h)
        feat, updated_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
            prompt_len=mel_len1,
            use_prompt_cache=False,
            grouped_prompt_uncached=grouped_prompt_uncached,
            source_cache_len=0,
            source_cache_end=0,
            source_mel_offset=source_mel_offset,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), updated_cache

    def _prepare_grouped_prompt_uncached_inputs(self, grouped_prompt_inputs):
        prompt_tokens = grouped_prompt_inputs["prompt_tokens"]
        prompt_feats = grouped_prompt_inputs["prompt_feats"]
        embeddings = grouped_prompt_inputs["embeddings"]
        weights = grouped_prompt_inputs["branch_weights"]
        if not prompt_tokens:
            raise ValueError("grouped prompt inputs have no active branches")

        device = self.input_embedding.weight.device
        branch_mus = []
        branch_conds = []
        branch_spks = []
        for branch_token, branch_feat, branch_embedding in zip(prompt_tokens, prompt_feats, embeddings):
            branch_token = branch_token.to(device=device)
            branch_feat = branch_feat.to(device=device)
            branch_embedding = F.normalize(branch_embedding.to(device=device), dim=1)
            branch_spk = self.spk_embed_affine_layer(branch_embedding)

            branch_token_len = torch.tensor([branch_token.shape[1]], dtype=torch.int32, device=device)
            token_mask = (~make_pad_mask(branch_token_len)).unsqueeze(-1).to(branch_spk)
            token_embed = self.input_embedding(torch.clamp(branch_token, min=0)) * token_mask
            branch_h = self.pre_lookahead_layer(token_embed)
            branch_h = branch_h.repeat_interleave(self.token_mel_ratio, dim=1)
            branch_len = min(branch_feat.shape[1], branch_h.shape[1])
            if branch_len <= 0:
                raise ValueError("grouped prompt branch is too short")
            branch_mus.append(branch_h[:, :branch_len].transpose(1, 2).contiguous())
            branch_conds.append(branch_feat[:, :branch_len].to(dtype=branch_h.dtype).transpose(1, 2).contiguous())
            branch_spks.append(branch_spk)

        return {
            "branch_mus": branch_mus,
            "branch_conds": branch_conds,
            "branch_spks": branch_spks,
            "branch_weights": torch.tensor(weights, dtype=torch.float32, device=device),
            "branch_indices": list(grouped_prompt_inputs.get("branch_indices", range(len(branch_mus)))),
            "dominant_branch_position": int(grouped_prompt_inputs.get("dominant_branch_position", 0)),
            "attention_temperature": float(grouped_prompt_inputs.get("attention_temperature", 1.0)),
        }

    @torch.inference_mode()
    def prepare_prompt_cache(self,
                             prompt_token,
                             prompt_token_len,
                             prompt_feat,
                             prompt_feat_len,
                             embedding,
                             streaming,
                             cache_storage_dtype=None,
                             cache_target_device=None,
                             keep_prompt_inputs=True):
        assert prompt_token.shape[0] == 1
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        mask = (~make_pad_mask(prompt_token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(prompt_token, min=0)) * mask
        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        static_chunk_size = self.decoder.estimator.static_chunk_size
        cache_len = int(prompt_feat.shape[1] // static_chunk_size * static_chunk_size)
        if cache_len <= 0:
            return 0, None

        mu = h[:, :cache_len].transpose(1, 2).contiguous()
        cond = prompt_feat[:, :cache_len].transpose(1, 2).contiguous()
        mask = (~make_pad_mask(torch.tensor([cache_len], device=prompt_feat.device))).to(h)
        cache = self.decoder.prepare_prompt_cache(
            mu=mu,
            mask=mask.unsqueeze(1),
            n_timesteps=10,
            spks=embedding,
            cond=cond,
            streaming=streaming,
            cache_storage_dtype=cache_storage_dtype,
            cache_target_device=cache_target_device,
            keep_prompt_inputs=keep_prompt_inputs,
        )
        cache_token_len = min(prompt_token.shape[1], cache_len // self.token_mel_ratio)
        cache["flow_token_tail"] = self._conditioning_tail_from_embedding(token[:, :cache_token_len])
        return cache_len, cache

    @torch.inference_mode()
    def prepare_soft_prompt_cache(self,
                                  soft_prompt_mu,
                                  soft_prompt_feat,
                                  soft_speaker_embedding,
                                  streaming,
                                  cache_storage_dtype=None,
                                  cache_target_device=None,
                                  keep_prompt_inputs=True):
        assert soft_prompt_mu.shape[0] == 1
        if soft_prompt_mu.shape != soft_prompt_feat.shape:
            raise ValueError(
                f"soft_prompt_mu and soft_prompt_feat must have the same shape, got "
                f"{tuple(soft_prompt_mu.shape)} and {tuple(soft_prompt_feat.shape)}"
            )
        embedding = F.normalize(soft_speaker_embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        static_chunk_size = self.decoder.estimator.static_chunk_size
        cache_len = int(soft_prompt_mu.shape[1] // static_chunk_size * static_chunk_size) if static_chunk_size > 0 else int(soft_prompt_mu.shape[1])
        if cache_len <= 0:
            return 0, None

        mu = soft_prompt_mu[:, :cache_len].transpose(1, 2).contiguous()
        cond = soft_prompt_feat[:, :cache_len].transpose(1, 2).contiguous()
        mask = (~make_pad_mask(torch.tensor([cache_len], device=soft_prompt_mu.device))).to(soft_prompt_mu)
        cache = self.decoder.prepare_prompt_cache(
            mu=mu,
            mask=mask.unsqueeze(1),
            n_timesteps=10,
            spks=embedding,
            cond=cond,
            streaming=streaming,
            cache_storage_dtype=cache_storage_dtype,
            cache_target_device=cache_target_device,
            keep_prompt_inputs=keep_prompt_inputs,
        )
        cache["soft_prompt"] = True
        return cache_len, cache

    def _source_conditioning_from_tail(
        self,
        token: torch.Tensor,
        token_tail: torch.Tensor,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_embed = self.input_embedding(torch.clamp(token, min=0))
        if finalize is True:
            emit_embed = token_embed
            context = torch.zeros(0, 0, 0, device=token.device, dtype=token_embed.dtype)
        else:
            emit_embed = token_embed[:, :-self.pre_lookahead_len]
            context = token_embed[:, -self.pre_lookahead_len:]
        token_tail = token_tail.to(device=token.device, dtype=token_embed.dtype)
        combined = torch.cat([token_tail, emit_embed], dim=1) if token_tail.shape[1] > 0 else emit_embed
        h = self.pre_lookahead_layer(combined, context=context)
        h = h[:, token_tail.shape[1]:]
        return h.repeat_interleave(self.token_mel_ratio, dim=1), token_embed

    def _conditioning_tail_from_embedding(self, token_embed: torch.Tensor) -> torch.Tensor:
        tail_size = self._conditioning_tail_size()
        return token_embed[:, -tail_size:].detach()

    def _attach_flow_conditioning_tail(
        self,
        updated_cache,
        prompt_cache_steps,
        source_token_embed: torch.Tensor,
        source_cache_len: int,
        source_cache_end: int,
    ) -> None:
        if updated_cache is None or source_cache_len <= 0:
            return
        token_tail = prompt_cache_steps.get("flow_token_tail") if isinstance(prompt_cache_steps, dict) else None
        if token_tail is None:
            return
        source_cache_tokens = source_cache_len // self.token_mel_ratio
        source_cache_end_tokens = source_cache_end // self.token_mel_ratio if source_cache_end > 0 else source_token_embed.shape[1]
        source_cache_end_tokens = min(source_cache_end_tokens, source_token_embed.shape[1])
        source_cache_start_tokens = max(0, source_cache_end_tokens - source_cache_tokens)
        selected = source_token_embed[:, source_cache_start_tokens:source_cache_end_tokens]
        updated_cache["flow_token_tail"] = self._merge_conditioning_tail(token_tail, selected)

    def _merge_conditioning_tail(self, token_tail: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        token_tail = token_tail.to(device=token_embed.device, dtype=token_embed.dtype)
        merged = torch.cat([token_tail, token_embed], dim=1) if token_embed.shape[1] > 0 else token_tail
        return merged[:, -self._conditioning_tail_size():].detach()

    def _conditioning_tail_size(self) -> int:
        return int(self.pre_lookahead_layer.conv2.kernel_size[0] - 1)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})
    model = configs['flow']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len = 10 * model.decoder.estimator.static_chunk_size
    chunk_size = model.decoder.estimator.static_chunk_size
    context_size = model.pre_lookahead_layer.pre_lookahead_len
    token = torch.randint(0, 6561, size=(1, max_len)).to(device)
    token_len = torch.tensor([max_len]).to(device)
    prompt_token = torch.randint(0, 6561, size=(1, chunk_size)).to(device)
    prompt_token_len = torch.tensor([chunk_size]).to(device)
    prompt_feat = torch.rand(1, chunk_size * 2, 80).to(device)
    prompt_feat_len = torch.tensor([chunk_size * 2]).to(device)
    prompt_embedding = torch.rand(1, 192).to(device)
    pred_gt, _ = model.inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=True)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(token[:, :i + chunk_size + context_size], torch.tensor([token[:, :i + chunk_size + context_size].shape[1]]).to(device),
                                        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=finalize)
        pred_chunk = pred_chunk[:, :, i * model.token_mel_ratio:]
        print((pred_gt[:, :, i * model.token_mel_ratio: i * model.token_mel_ratio + pred_chunk.shape[2]] - pred_chunk).abs().max().item())
