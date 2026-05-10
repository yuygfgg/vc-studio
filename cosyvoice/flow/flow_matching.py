# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
import torch
import torch.nn.functional as F
from cosyvoice.utils.common import set_all_random_seed


class BASECFM(torch.nn.Module):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        prompt_len=0,
        cache=torch.zeros(1, 80, 0, 2),
        **kwargs,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False, prompt_len=0, source_mel_offset=0):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16 inference, so set dtype=spks.dtype
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming=streaming,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False, prompt_len=0, source_mel_offset=0):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(
                x,
                mask,
                mu,
                t,
                spks,
                cond,
                streaming=streaming,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
            )
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', (2, 80, x.size(2)))
                estimator.set_input_shape('mask', (2, 1, x.size(2)))
                estimator.set_input_shape('mu', (2, 80, x.size(2)))
                estimator.set_input_shape('t', (2,))
                estimator.set_input_shape('spks', (2, 80))
                estimator.set_input_shape('cond', (2, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)

        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        streaming=False,
        prompt_len=0,
        use_prompt_cache=False,
        prompt_cache_steps=None,
        source_cache_len=0,
        source_cache_end=0,
        source_mel_offset=0,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.make_inference_noise(
            length=mu.size(2),
            prompt_len=prompt_len,
            source_mel_offset=source_mel_offset,
            device=mu.device,
            dtype=mu.dtype,
            temperature=temperature,
        )
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        if prompt_cache_steps is not None:
            return self.solve_euler_prepared_prompt_cache(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                prompt_len=prompt_len,
                prompt_cache_steps=prompt_cache_steps,
                source_cache_len=source_cache_len,
                source_cache_end=source_cache_end,
                source_mel_offset=source_mel_offset,
                streaming=streaming,
            )
        if use_prompt_cache is True:
            return self.solve_euler_prompt_cache(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
                streaming=streaming,
            ), None
        return self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            streaming=streaming,
            prompt_len=prompt_len,
            source_mel_offset=source_mel_offset,
        ), None

    def make_inference_noise(self, length, prompt_len, source_mel_offset, device, dtype, temperature):
        prompt_len = min(max(int(prompt_len), 0), int(length))
        source_len = int(length) - prompt_len
        if source_mel_offset is None:
            source_mel_offset = 0
        prompt_noise = self.noise_slice(0, prompt_len, device=device, dtype=dtype)
        source_noise = self.noise_slice(int(source_mel_offset), source_len, device=device, dtype=dtype)
        return torch.cat([prompt_noise, source_noise], dim=2) * temperature

    def noise_slice(self, start, length, device, dtype):
        if length <= 0:
            return torch.zeros([1, 80, 0], device=device, dtype=dtype)
        start = max(int(start), 0)
        end = start + int(length)
        if end <= self.rand_noise.shape[2]:
            return self.rand_noise[:, :, start:end].to(device=device, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn([1, 80, end], generator=generator)
        return noise[:, :, start:end].to(device=device, dtype=dtype)

    def prepare_prompt_cache(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        prompt_len = mu.size(2)
        z = self.rand_noise[:, :, :prompt_len].to(mu.device).to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        prompt_x = z
        steps = []
        for step in range(1, len(t_span)):
            prompt_out, prompt_cache, prompt_inputs = self.forward_prompt_cache_only(
                prompt_x,
                mask,
                mu,
                t,
                spks,
                cond,
                streaming=streaming,
            )
            prompt_dphi, prompt_cfg_dphi = torch.split(prompt_out, [prompt_x.size(0), prompt_x.size(0)], dim=0)
            prompt_dphi = ((1.0 + self.inference_cfg_rate) * prompt_dphi - self.inference_cfg_rate * prompt_cfg_dphi)
            steps.append({
                "prompt_cache": prompt_cache,
                "prompt_inputs": prompt_inputs,
            })
            prompt_x = prompt_x + dt * prompt_dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return {
            "steps": steps,
            "final_prompt_x": prompt_x,
            "cache_len": prompt_len,
            "base_cache_len": prompt_len,
            "history_cache_len": 0,
            "base_cache": None,
        }

    def solve_euler_prompt_cache(self, x, t_span, mu, mask, spks, cond, prompt_len, source_mel_offset=0, streaming=False):
        if prompt_len <= 0:
            return self.solve_euler(
                x,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                streaming=streaming,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
            )
        if getattr(self.estimator, "static_chunk_size", 0) > 0 and prompt_len % self.estimator.static_chunk_size != 0:
            return self.solve_euler(
                x,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                streaming=streaming,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
            )

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        prompt_x = x[:, :, :prompt_len]
        source_x = x[:, :, prompt_len:]
        prompt_mu = mu[:, :, :prompt_len]
        source_mu = mu[:, :, prompt_len:]
        prompt_cond = cond[:, :, :prompt_len]
        source_cond = cond[:, :, prompt_len:]
        prompt_mask = mask[:, :, :prompt_len]

        for step in range(1, len(t_span)):
            prompt_dphi, source_dphi = self.forward_estimator_prompt_cache(
                prompt_x,
                source_x,
                prompt_mask,
                prompt_mu,
                source_mu,
                t,
                spks,
                prompt_cond,
                source_cond,
                source_mel_offset=source_mel_offset,
                streaming=streaming,
            )
            prompt_dphi, prompt_cfg_dphi = torch.split(prompt_dphi, [x.size(0), x.size(0)], dim=0)
            source_dphi, source_cfg_dphi = torch.split(source_dphi, [x.size(0), x.size(0)], dim=0)
            prompt_dphi = ((1.0 + self.inference_cfg_rate) * prompt_dphi - self.inference_cfg_rate * prompt_cfg_dphi)
            source_dphi = ((1.0 + self.inference_cfg_rate) * source_dphi - self.inference_cfg_rate * source_cfg_dphi)
            prompt_x = prompt_x + dt * prompt_dphi
            source_x = source_x + dt * source_dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return torch.cat([prompt_x, source_x], dim=2).float()

    def solve_euler_prepared_prompt_cache(
        self,
        x,
        t_span,
        mu,
        mask,
        spks,
        cond,
        prompt_len,
        prompt_cache_steps,
        source_cache_len=0,
        source_cache_end=0,
        source_mel_offset=0,
        streaming=False,
    ):
        if prompt_len <= 0:
            return self.solve_euler(
                x,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                streaming=streaming,
                prompt_len=prompt_len,
                source_mel_offset=source_mel_offset,
            )

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        source_x = x[:, :, prompt_len:]
        source_mu = mu[:, :, prompt_len:]
        source_cond = cond[:, :, prompt_len:]
        collect_source_cache = source_cache_len > 0

        source_step_caches = []
        for step, step_cache in enumerate(prompt_cache_steps["steps"], start=1):
            source_dphi, source_step_cache = self.forward_source_with_prepared_prompt_cache(
                source_x,
                source_mu,
                t,
                spks,
                source_cond,
                step_cache,
                streaming=streaming,
                return_source_cache=collect_source_cache,
                source_mel_offset=source_mel_offset,
            )
            if collect_source_cache:
                source_step_caches.append(source_step_cache)
            source_dphi, source_cfg_dphi = torch.split(source_dphi, [x.size(0), x.size(0)], dim=0)
            source_dphi = ((1.0 + self.inference_cfg_rate) * source_dphi - self.inference_cfg_rate * source_cfg_dphi)
            source_x = source_x + dt * source_dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        sample = torch.cat([prompt_cache_steps["final_prompt_x"], source_x], dim=2).float()
        updated_cache = None
        if collect_source_cache:
            updated_cache = self.build_bounded_source_cache(
                prompt_cache_steps=prompt_cache_steps,
                source_step_caches=source_step_caches,
                source_x=source_x,
                source_cache_len=source_cache_len,
                source_cache_end=source_cache_end,
            )
        return sample, updated_cache

    def build_bounded_source_cache(
        self,
        prompt_cache_steps,
        source_step_caches,
        source_x,
        source_cache_len,
        source_cache_end,
    ):
        base_cache = prompt_cache_steps.get("base_cache") or prompt_cache_steps
        base_len = base_cache["base_cache_len"]
        source_end = source_x.shape[2] if source_cache_end <= 0 else min(source_cache_end, source_x.shape[2])
        source_start = max(0, source_end - source_cache_len)
        history_len = source_end - source_start
        if history_len <= 0:
            return base_cache

        steps = []
        for base_step, source_step in zip(base_cache["steps"], source_step_caches):
            source_cache = source_step["source_cache"]
            source_inputs = source_step["source_inputs"]
            input_embed_cache = None
            if isinstance(self.estimator, torch.nn.Module):
                input_embed_cache = self.estimator.extend_input_embed_cache(
                    base_step["prompt_cache"].get("input_embed_cache"),
                    source_inputs["x_source_in"][:, :, source_start:source_end],
                    source_inputs["source_mu_in"][:, :, source_start:source_end],
                    source_inputs["source_cond_in"][:, :, source_start:source_end],
                    base_step["prompt_inputs"]["spks_in"],
                )
            prompt_cache = {
                "kv": base_step["prompt_cache"]["kv"],
                "history_kv": [
                    (
                        key[:, :, source_start:source_end].detach(),
                        value[:, :, source_start:source_end].detach(),
                    )
                    for key, value in source_cache["kv"]
                ],
                "prompt_len": base_len + history_len,
                "base_prompt_len": base_len,
            }
            if input_embed_cache is not None:
                prompt_cache["input_embed_cache"] = input_embed_cache
            steps.append({
                "prompt_cache": prompt_cache,
                "prompt_inputs": {
                    "x_prompt_in": base_step["prompt_inputs"]["x_prompt_in"],
                    "prompt_mu_in": base_step["prompt_inputs"]["prompt_mu_in"],
                    "prompt_cond_in": base_step["prompt_inputs"]["prompt_cond_in"],
                    "history_x_in": source_inputs["x_source_in"][:, :, source_start:source_end].detach(),
                    "history_mu_in": source_inputs["source_mu_in"][:, :, source_start:source_end].detach(),
                    "history_cond_in": source_inputs["source_cond_in"][:, :, source_start:source_end].detach(),
                    "t_in": base_step["prompt_inputs"]["t_in"],
                    "spks_in": base_step["prompt_inputs"]["spks_in"],
                },
            })

        return {
            "steps": steps,
            "final_prompt_x": torch.cat(
                [base_cache["final_prompt_x"], source_x[:, :, source_start:source_end].detach()],
                dim=2,
            ),
            "cache_len": base_len + history_len,
            "base_cache_len": base_len,
            "history_cache_len": history_len,
            "base_cache": base_cache,
        }

    def forward_prompt_cache_only(self, prompt_x, prompt_mask, prompt_mu, t, spks, prompt_cond, streaming=False):
        batch = prompt_x.size(0)
        prompt_len = prompt_x.size(2)
        x_prompt_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        prompt_mask_in = torch.ones([2 * batch, 1, prompt_len], device=prompt_x.device, dtype=torch.bool)
        prompt_mu_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        prompt_cond_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        t_in = torch.zeros([2 * batch], device=prompt_x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2 * batch, 80], device=prompt_x.device, dtype=spks.dtype)

        x_prompt_in[:] = prompt_x
        prompt_mu_in[:batch] = prompt_mu
        prompt_cond_in[:batch] = prompt_cond
        t_in[:] = t.unsqueeze(0)
        spks_in[:batch] = spks

        prompt_out, prompt_cache = self.estimator.forward_prompt_cache(
            x_prompt_in,
            prompt_mask_in,
            prompt_mu_in,
            t_in,
            spks_in,
            prompt_cond_in,
            streaming=streaming,
        )
        return prompt_out, prompt_cache, {
            "x_prompt_in": x_prompt_in,
            "prompt_mu_in": prompt_mu_in,
            "prompt_cond_in": prompt_cond_in,
            "t_in": t_in,
            "spks_in": spks_in,
        }

    def forward_source_with_prepared_prompt_cache(
        self,
        source_x,
        source_mu,
        t,
        spks,
        source_cond,
        step_cache,
        streaming=False,
        return_source_cache=False,
        source_mel_offset=0,
    ):
        prompt_inputs = step_cache["prompt_inputs"]
        prompt_cache = step_cache["prompt_cache"]
        batch = source_x.size(0)
        source_len = source_x.size(2)
        prompt_len = prompt_cache["prompt_len"]
        x_source_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        source_mu_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        source_cond_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        full_mask_in = torch.ones([2 * batch, 1, prompt_len + source_len], device=source_x.device, dtype=torch.bool)

        x_source_in[:] = source_x
        source_mu_in[:batch] = source_mu
        source_cond_in[:batch] = source_cond
        x_prompt_in = prompt_inputs["x_prompt_in"]
        prompt_mu_in = prompt_inputs["prompt_mu_in"]
        prompt_cond_in = prompt_inputs["prompt_cond_in"]
        if "history_x_in" in prompt_inputs and "input_embed_cache" not in prompt_cache:
            x_prompt_in = torch.cat([x_prompt_in, prompt_inputs["history_x_in"]], dim=2)
            prompt_mu_in = torch.cat([prompt_mu_in, prompt_inputs["history_mu_in"]], dim=2)
            prompt_cond_in = torch.cat([prompt_cond_in, prompt_inputs["history_cond_in"]], dim=2)

        source_out, source_cache = self.estimator.forward_source_with_prompt_cache(
            x_source_in,
            full_mask_in,
            source_mu_in,
            prompt_inputs["t_in"],
            prompt_inputs["spks_in"],
            source_cond_in,
            prompt_x=x_prompt_in,
            prompt_mu=prompt_mu_in,
            prompt_cond=prompt_cond_in,
            prompt_cache=prompt_cache,
            streaming=streaming,
            return_source_cache=return_source_cache,
            source_mel_offset=source_mel_offset,
        )
        if return_source_cache:
            return source_out, {
                "source_cache": source_cache,
                "source_inputs": {
                    "x_source_in": x_source_in,
                    "source_mu_in": source_mu_in,
                    "source_cond_in": source_cond_in,
                },
            }
        return source_out, None

    def forward_estimator_prompt_cache(
        self,
        prompt_x,
        source_x,
        prompt_mask,
        prompt_mu,
        source_mu,
        t,
        spks,
        prompt_cond,
        source_cond,
        source_mel_offset=0,
        streaming=False,
    ):
        if not isinstance(self.estimator, torch.nn.Module):
            full_x = torch.cat([prompt_x, source_x], dim=2)
            full_mu = torch.cat([prompt_mu, source_mu], dim=2)
            full_cond = torch.cat([prompt_cond, source_cond], dim=2)
            full_mask = torch.ones([full_x.shape[0], 1, full_x.shape[2]], device=full_x.device, dtype=torch.bool)
            return self.forward_estimator(full_x, full_mask, full_mu, t, spks, full_cond, streaming=streaming)

        batch = prompt_x.size(0)
        prompt_len = prompt_x.size(2)
        source_len = source_x.size(2)
        x_prompt_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        x_source_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        prompt_mask_in = torch.ones([2 * batch, 1, prompt_len], device=prompt_x.device, dtype=torch.bool)
        full_mask_in = torch.ones([2 * batch, 1, prompt_len + source_len], device=prompt_x.device, dtype=torch.bool)
        prompt_mu_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        source_mu_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        prompt_cond_in = torch.zeros([2 * batch, 80, prompt_len], device=prompt_x.device, dtype=spks.dtype)
        source_cond_in = torch.zeros([2 * batch, 80, source_len], device=source_x.device, dtype=spks.dtype)
        t_in = torch.zeros([2 * batch], device=prompt_x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2 * batch, 80], device=prompt_x.device, dtype=spks.dtype)

        x_prompt_in[:] = prompt_x
        x_source_in[:] = source_x
        prompt_mu_in[:batch] = prompt_mu
        source_mu_in[:batch] = source_mu
        prompt_cond_in[:batch] = prompt_cond
        source_cond_in[:batch] = source_cond
        t_in[:] = t.unsqueeze(0)
        spks_in[:batch] = spks

        prompt_out, prompt_cache = self.estimator.forward_prompt_cache(
            x_prompt_in,
            prompt_mask_in,
            prompt_mu_in,
            t_in,
            spks_in,
            prompt_cond_in,
            streaming=streaming,
        )
        source_out, _ = self.estimator.forward_source_with_prompt_cache(
            x_source_in,
            full_mask_in,
            source_mu_in,
            t_in,
            spks_in,
            source_cond_in,
            prompt_x=x_prompt_in,
            prompt_mu=prompt_mu_in,
            prompt_cond=prompt_cond_in,
            prompt_cache=prompt_cache,
            streaming=streaming,
            source_mel_offset=source_mel_offset,
        )
        return prompt_out, source_out
