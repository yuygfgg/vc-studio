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
        grouped_prompt_uncached=None,
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
        if grouped_prompt_uncached is not None:
            grouped_prompt_uncached = dict(grouped_prompt_uncached)
            grouped_prompt_uncached["temperature"] = temperature
            return self.solve_euler_grouped_prompt_uncached(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                prompt_len=prompt_len,
                grouped_prompt=grouped_prompt_uncached,
                source_mel_offset=source_mel_offset,
                streaming=streaming,
            ), None
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

    def prepare_prompt_cache(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        streaming=False,
        cache_storage_dtype=None,
        cache_target_device=None,
        keep_prompt_inputs=True,
    ):
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
            prompt_cache = self.optimize_prompt_cache_step_for_storage(
                prompt_cache,
                storage_dtype=cache_storage_dtype,
                target_device=cache_target_device,
            )
            steps.append({
                "prompt_cache": prompt_cache,
                "prompt_inputs": prompt_inputs if keep_prompt_inputs else None,
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

    def optimize_prompt_cache_step_for_storage(self, prompt_cache, storage_dtype=None, target_device=None):
        if storage_dtype is None and target_device is None:
            return prompt_cache
        prompt_cache = dict(prompt_cache)
        if "kv" in prompt_cache:
            prompt_cache["kv"] = [
                (
                    self.optimize_cache_tensor(key, storage_dtype, target_device),
                    self.optimize_cache_tensor(value, storage_dtype, target_device),
                )
                for key, value in prompt_cache["kv"]
            ]
        input_embed_cache = prompt_cache.get("input_embed_cache")
        if isinstance(input_embed_cache, dict):
            prompt_cache["input_embed_cache"] = self.optimize_cache_tree(
                input_embed_cache,
                storage_dtype,
                target_device,
            )
        return prompt_cache

    def optimize_cache_tree(self, value, storage_dtype=None, target_device=None):
        if isinstance(value, torch.Tensor):
            return self.optimize_cache_tensor(value, storage_dtype, target_device)
        if isinstance(value, dict):
            return {key: self.optimize_cache_tree(item, storage_dtype, target_device) for key, item in value.items()}
        if isinstance(value, list):
            return [self.optimize_cache_tree(item, storage_dtype, target_device) for item in value]
        if isinstance(value, tuple):
            return tuple(self.optimize_cache_tree(item, storage_dtype, target_device) for item in value)
        return value

    def optimize_cache_tensor(self, tensor, storage_dtype=None, target_device=None):
        kwargs = {}
        if tensor.is_floating_point() and storage_dtype is not None:
            kwargs["dtype"] = storage_dtype
        if target_device is not None:
            kwargs["device"] = target_device
        if not kwargs:
            return tensor
        return tensor.to(**kwargs)

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
        cache_window = self.source_cache_window(
            prompt_cache_steps,
            source_x,
            source_cache_len,
            source_cache_end,
        )

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
                source_step_cache = self.prepare_source_step_cache_for_history_storage(
                    source_step_cache,
                    step_cache["prompt_cache"],
                    cache_window["current_start"],
                    cache_window["current_end"],
                )
                source_step_caches.append(source_step_cache)
            source_dphi, source_cfg_dphi = torch.split(source_dphi, [x.size(0), x.size(0)], dim=0)
            source_dphi = ((1.0 + self.inference_cfg_rate) * source_dphi - self.inference_cfg_rate * source_cfg_dphi)
            source_x = source_x + dt * source_dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        final_prompt_x = prompt_cache_steps["final_prompt_x"].to(device=source_x.device, dtype=source_x.dtype)
        sample = torch.cat([final_prompt_x, source_x], dim=2).float()
        updated_cache = None
        if collect_source_cache:
            updated_cache = self.build_bounded_source_cache(
                prompt_cache_steps=prompt_cache_steps,
                source_step_caches=source_step_caches,
                source_x=source_x,
                source_cache_len=source_cache_len,
                source_cache_end=cache_window["combined_end"],
            )
        return sample, updated_cache

    def forward_prepared_prompt_cache_source(
        self,
        source_mu,
        source_cond,
        n_timesteps,
        temperature=1.0,
        spks=None,
        prompt_cache_steps=None,
        source_cache_len=0,
        source_cache_end=0,
        source_mel_offset=0,
        streaming=False,
    ):
        source_x = self.noise_slice(
            int(source_mel_offset or 0),
            source_mu.size(2),
            device=source_mu.device,
            dtype=source_mu.dtype,
        ) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=source_mu.device, dtype=source_mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler_prepared_prompt_cache_source(
            source_x,
            t_span=t_span,
            source_mu=source_mu,
            spks=spks,
            source_cond=source_cond,
            prompt_cache_steps=prompt_cache_steps,
            source_cache_len=source_cache_len,
            source_cache_end=source_cache_end,
            source_mel_offset=source_mel_offset,
            streaming=streaming,
        )

    def solve_euler_prepared_prompt_cache_source(
        self,
        source_x,
        t_span,
        source_mu,
        spks,
        source_cond,
        prompt_cache_steps,
        source_cache_len=0,
        source_cache_end=0,
        source_mel_offset=0,
        streaming=False,
    ):
        if prompt_cache_steps is None:
            raise ValueError("prompt_cache_steps is required for source-only prompt-cache inference")

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        collect_source_cache = source_cache_len > 0
        cache_window = self.source_cache_window(
            prompt_cache_steps,
            source_x,
            source_cache_len,
            source_cache_end,
        )

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
                source_step_cache = self.prepare_source_step_cache_for_history_storage(
                    source_step_cache,
                    step_cache["prompt_cache"],
                    cache_window["current_start"],
                    cache_window["current_end"],
                )
                source_step_caches.append(source_step_cache)
            source_dphi, source_cfg_dphi = torch.split(source_dphi, [source_x.size(0), source_x.size(0)], dim=0)
            source_dphi = ((1.0 + self.inference_cfg_rate) * source_dphi - self.inference_cfg_rate * source_cfg_dphi)
            source_x = source_x + dt * source_dphi
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        final_prompt_x = prompt_cache_steps["final_prompt_x"].to(device=source_x.device, dtype=source_x.dtype)
        sample = torch.cat([final_prompt_x, source_x], dim=2).float()
        updated_cache = None
        if collect_source_cache:
            updated_cache = self.build_bounded_source_cache(
                prompt_cache_steps=prompt_cache_steps,
                source_step_caches=source_step_caches,
                source_x=source_x,
                source_cache_len=source_cache_len,
                source_cache_end=cache_window["combined_end"],
            )
        return sample, updated_cache

    def solve_euler_grouped_prompt_uncached(
        self,
        x,
        t_span,
        mu,
        mask,
        spks,
        cond,
        prompt_len,
        grouped_prompt,
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

        branch_mus = grouped_prompt["branch_mus"]
        branch_conds = grouped_prompt["branch_conds"]
        branch_spks = grouped_prompt["branch_spks"]
        branch_weights = grouped_prompt["branch_weights"]
        dominant_position = int(grouped_prompt.get("dominant_branch_position", 0))
        attention_temperature = float(grouped_prompt.get("attention_temperature", 1.0))
        temperature = float(grouped_prompt.get("temperature", 1.0))
        if not isinstance(self.estimator, torch.nn.Module) or not hasattr(self.estimator, "forward_grouped_prompt_source_uncached"):
            raise RuntimeError("uncached grouped prompt attention requires the PyTorch DiT estimator")

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        source_x = x[:, :, prompt_len:]
        source_mu = mu[:, :, prompt_len:]
        source_cond = cond[:, :, prompt_len:]
        branch_prompt_x = [
            self.noise_slice(0, branch_mu.shape[2], device=x.device, dtype=x.dtype) * temperature
            for branch_mu in branch_mus
        ]

        for step in range(1, len(t_span)):
            prompt_outs, source_dphi = self.estimator.forward_grouped_prompt_source_uncached(
                branch_prompt_x=[prompt_x.to(device=x.device, dtype=spks.dtype) for prompt_x in branch_prompt_x],
                branch_prompt_mu=[branch_mu.to(device=x.device, dtype=spks.dtype) for branch_mu in branch_mus],
                branch_prompt_cond=[branch_cond.to(device=x.device, dtype=spks.dtype) for branch_cond in branch_conds],
                branch_spks=[branch_spk.to(device=x.device, dtype=spks.dtype) for branch_spk in branch_spks],
                source_x=source_x,
                source_mu=source_mu,
                t=t,
                source_spks=spks,
                source_cond=source_cond,
                branch_weights=branch_weights,
                dominant_branch_position=dominant_position,
                streaming=streaming,
                source_mel_offset=source_mel_offset,
                attention_temperature=attention_temperature,
            )
            next_branch_prompt_x = []
            for prompt_x, prompt_out in zip(branch_prompt_x, prompt_outs):
                prompt_dphi, prompt_cfg_dphi = torch.split(prompt_out, [prompt_x.size(0), prompt_x.size(0)], dim=0)
                prompt_dphi = ((1.0 + self.inference_cfg_rate) * prompt_dphi - self.inference_cfg_rate * prompt_cfg_dphi)
                next_branch_prompt_x.append(prompt_x + dt * prompt_dphi)
            source_dphi, source_cfg_dphi = torch.split(source_dphi, [x.size(0), x.size(0)], dim=0)
            source_dphi = ((1.0 + self.inference_cfg_rate) * source_dphi - self.inference_cfg_rate * source_cfg_dphi)
            source_x = source_x + dt * source_dphi
            branch_prompt_x = next_branch_prompt_x
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        final_prompt_x = branch_prompt_x[dominant_position]
        return torch.cat([final_prompt_x, source_x], dim=2).float()

    def source_cache_window(self, prompt_cache_steps, source_x, source_cache_len, source_cache_end):
        existing_history_len = int(prompt_cache_steps.get("history_cache_len", 0)) if isinstance(prompt_cache_steps, dict) else 0
        current_len = int(source_x.shape[2])
        combined_len = existing_history_len + current_len
        if source_cache_len <= 0 or combined_len <= 0:
            combined_end = 0
        elif source_cache_end <= 0:
            combined_end = combined_len
        else:
            combined_end = min(max(0, int(source_cache_end)), combined_len)
        combined_start = max(0, combined_end - max(0, int(source_cache_len)))
        history_start = min(combined_start, existing_history_len)
        history_end = min(combined_end, existing_history_len)
        current_start = max(0, min(current_len, combined_start - existing_history_len))
        current_end = max(current_start, min(current_len, combined_end - existing_history_len))
        return {
            "existing_history_len": existing_history_len,
            "combined_start": combined_start,
            "combined_end": combined_end,
            "history_start": history_start,
            "history_end": history_end,
            "current_start": current_start,
            "current_end": current_end,
        }

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
        window = self.source_cache_window(prompt_cache_steps, source_x, source_cache_len, source_cache_end)
        existing_history_len = window["existing_history_len"]
        history_start = window["history_start"]
        history_end = window["history_end"]
        current_start = window["current_start"]
        current_end = window["current_end"]
        history_len = (history_end - history_start) + (current_end - current_start)
        if history_len <= 0:
            return base_cache

        active_steps = prompt_cache_steps.get("steps", []) if isinstance(prompt_cache_steps, dict) else []
        steps = []
        for step_index, (base_step, source_step) in enumerate(zip(base_cache["steps"], source_step_caches)):
            base_prompt_cache = base_step["prompt_cache"]
            active_step = active_steps[step_index] if step_index < len(active_steps) else base_step
            active_prompt_cache = active_step["prompt_cache"]
            active_prompt_inputs = active_step["prompt_inputs"]
            source_cache = source_step["source_cache"]
            source_inputs = source_step["source_inputs"]
            source_step_start = source_step.get("source_cache_start")
            source_step_end = source_step.get("source_cache_end")
            combined_inputs = self.concat_history_source_inputs(
                active_prompt_inputs,
                source_inputs,
                history_start,
                history_end,
                current_start,
                current_end,
                source_step_start,
            )
            input_embed_cache = None
            if isinstance(self.estimator, torch.nn.Module):
                input_embed_cache = self.estimator.extend_input_embed_cache(
                    base_prompt_cache.get("input_embed_cache"),
                    combined_inputs["x_source_in"],
                    combined_inputs["source_mu_in"],
                    combined_inputs["source_cond_in"],
                    base_step["prompt_inputs"]["spks_in"],
                )
            if base_prompt_cache.get("grouped_branch_attention"):
                if base_prompt_cache.get("grouped_attention_mode") == "sequential":
                    base_kv = base_prompt_cache["sequential_branch_caches"][0]["kv"]
                else:
                    base_kv = base_prompt_cache["grouped_kv"]
                history_kv = self.concat_history_source_kv(
                    active_prompt_cache.get("history_kv"),
                    source_cache["kv"],
                    base_kv,
                    history_start,
                    history_end,
                    current_start,
                    current_end,
                    source_step_start,
                    source_step_end,
                )
                prompt_cache = {
                    "grouped_branch_attention": True,
                    "grouped_attention_mode": base_prompt_cache.get("grouped_attention_mode", "vectorized"),
                    "branch_weights": base_prompt_cache["branch_weights"],
                    "branch_indices": base_prompt_cache.get("branch_indices"),
                    "branch_cache_lens": base_prompt_cache.get("branch_cache_lens"),
                    "attention_temperature": base_prompt_cache.get("attention_temperature", 1.0),
                    "history_kv": history_kv,
                    "prompt_len": base_len + history_len,
                    "base_prompt_len": base_len,
                }
                if base_prompt_cache.get("grouped_attention_mode") == "sequential":
                    prompt_cache["sequential_branch_caches"] = base_prompt_cache["sequential_branch_caches"]
                else:
                    prompt_cache["grouped_kv"] = base_prompt_cache["grouped_kv"]
                    prompt_cache["grouped_prompt_mask"] = base_prompt_cache["grouped_prompt_mask"]
            else:
                history_kv = self.concat_history_source_kv(
                    active_prompt_cache.get("history_kv"),
                    source_cache["kv"],
                    base_prompt_cache["kv"],
                    history_start,
                    history_end,
                    current_start,
                    current_end,
                    source_step_start,
                    source_step_end,
                )
                prompt_cache = {
                    "kv": base_prompt_cache["kv"],
                    "history_kv": history_kv,
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
                    "history_x_in": combined_inputs["x_source_in"].detach(),
                    "history_mu_in": combined_inputs["source_mu_in"].detach(),
                    "history_cond_in": combined_inputs["source_cond_in"].detach(),
                    "t_in": base_step["prompt_inputs"]["t_in"],
                    "spks_in": base_step["prompt_inputs"]["spks_in"],
                },
            })

        source_segments = []
        if existing_history_len > 0 and history_end > history_start:
            cached_source_x = prompt_cache_steps["final_prompt_x"][:, :, base_len:base_len + existing_history_len]
            source_segments.append(cached_source_x[:, :, history_start:history_end].detach())
        if current_end > current_start:
            source_segments.append(source_x[:, :, current_start:current_end].detach())
        cached_source_x = torch.cat(source_segments, dim=2)
        return {
            "steps": steps,
            "final_prompt_x": torch.cat([base_cache["final_prompt_x"], cached_source_x], dim=2),
            "cache_len": base_len + history_len,
            "base_cache_len": base_len,
            "history_cache_len": history_len,
            "base_cache": base_cache,
        }

    def concat_history_source_inputs(
        self,
        active_prompt_inputs,
        source_inputs,
        history_start,
        history_end,
        current_start,
        current_end,
        source_step_start,
    ):
        combined = {}
        key_pairs = (
            ("x_source_in", "history_x_in"),
            ("source_mu_in", "history_mu_in"),
            ("source_cond_in", "history_cond_in"),
        )
        for source_key, history_key in key_pairs:
            segments = []
            history_value = active_prompt_inputs.get(history_key)
            if history_value is not None and history_end > history_start:
                segments.append(history_value[:, :, history_start:history_end].detach())
            if current_end > current_start:
                offset = int(source_step_start or 0)
                segments.append(source_inputs[source_key][:, :, current_start - offset:current_end - offset].detach())
            combined[source_key] = torch.cat(segments, dim=2) if segments else source_inputs[source_key][:, :, :0]
        return combined

    def concat_history_source_kv(
        self,
        existing_history_kv,
        current_source_kv,
        base_kv,
        history_start,
        history_end,
        current_start,
        current_end,
        source_step_start,
        source_step_end,
    ):
        history_kv = []
        if existing_history_kv is not None and history_end > history_start:
            history_kv = self._slice_history_kv_like_base_cache(
                existing_history_kv,
                base_kv,
                history_start,
                history_end,
            )
        current_kv = []
        if current_end > current_start:
            if source_step_start == current_start and source_step_end == current_end:
                current_kv = current_source_kv
            else:
                offset = int(source_step_start or 0)
                current_kv = self._slice_history_kv_like_base_cache(
                    current_source_kv,
                    base_kv,
                    current_start - offset,
                    current_end - offset,
                )
        if not history_kv:
            return current_kv
        if not current_kv:
            return history_kv
        merged = []
        for (history_key, history_value), (current_key, current_value) in zip(history_kv, current_kv):
            merged.append(
                (
                    torch.cat([history_key, current_key.to(device=history_key.device, dtype=history_key.dtype)], dim=2),
                    torch.cat([history_value, current_value.to(device=history_value.device, dtype=history_value.dtype)], dim=2),
                )
            )
        return merged

    def prepare_source_step_cache_for_history_storage(self, source_step_cache, prompt_cache, source_start, source_end):
        if source_step_cache is None:
            return None
        source_start = max(0, int(source_start))
        source_end = max(source_start, int(source_end))
        base_kv = self.prompt_cache_base_kv(prompt_cache)
        source_cache = source_step_cache["source_cache"]
        if base_kv is not None:
            source_cache = {
                "kv": self._slice_history_kv_like_base_cache(
                    source_cache["kv"],
                    base_kv,
                    source_start,
                    source_end,
                ),
                "source_len": source_end - source_start,
            }
        source_inputs = {
            key: value[:, :, source_start:source_end].detach()
            for key, value in source_step_cache["source_inputs"].items()
        }
        return {
            "source_cache": source_cache,
            "source_inputs": source_inputs,
            "source_cache_start": source_start,
            "source_cache_end": source_end,
        }

    def prompt_cache_base_kv(self, prompt_cache):
        if prompt_cache.get("grouped_branch_attention"):
            if prompt_cache.get("grouped_attention_mode") == "sequential":
                branch_caches = prompt_cache.get("sequential_branch_caches") or []
                if not branch_caches:
                    return None
                return branch_caches[0].get("kv")
            return prompt_cache.get("grouped_kv")
        return prompt_cache.get("kv")

    def _slice_history_kv_like_base_cache(self, source_kv, base_kv, source_start, source_end):
        history_kv = []
        for (key, value), (base_key, base_value) in zip(source_kv, base_kv):
            history_kv.append(
                (
                    key[:, :, source_start:source_end].detach().to(device=base_key.device, dtype=base_key.dtype),
                    value[:, :, source_start:source_end].detach().to(device=base_value.device, dtype=base_value.dtype),
                )
            )
        return history_kv

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
