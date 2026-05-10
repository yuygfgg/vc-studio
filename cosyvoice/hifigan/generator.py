# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
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

"""HIFI-GAN"""

from typing import Dict, List
import numpy as np
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
from cosyvoice.transformer.convolution import CausalConv1d, CausalConv1dDownSample, CausalConv1dUpsample
from cosyvoice.transformer.activation import Snake
from cosyvoice.utils.common import init_weights


"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
 ,https://github.com/kan-bayashi/ParallelWaveGAN and
 https://github.com/NVIDIA/BigVGAN

"""


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""
    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        causal_type='left'
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        causal_type='left'
                    )
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs1))
        ])
        self.activations2 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs2))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def forward_stateful(
        self,
        x: torch.Tensor,
        state: Dict | None = None,
        commit_len: int | None = None,
    ) -> tuple[torch.Tensor, Dict]:
        state = state or {}
        if commit_len is None:
            commit_len = x.shape[2]
        commit_len = max(0, min(int(commit_len), x.shape[2]))
        convs1_cache = list(state.get("convs1", []))
        convs2_cache = list(state.get("convs2", []))
        new_convs1 = []
        new_convs2 = []
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            cache1 = convs1_cache[idx] if idx < len(convs1_cache) else torch.zeros(0, 0, 0)
            xt, new_cache1 = self.convs1[idx].forward_with_cache(xt, cache1, commit_len=commit_len)
            xt = self.activations2[idx](xt)
            cache2 = convs2_cache[idx] if idx < len(convs2_cache) else torch.zeros(0, 0, 0)
            xt, new_cache2 = self.convs2[idx].forward_with_cache(xt, cache2, commit_len=commit_len)
            x = xt + x
            new_convs1.append(new_cache1.detach())
            new_convs2.append(new_cache2.detach())
        return x, {"convs1": new_convs1, "convs2": new_convs2}

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class HarmonicSourceGenerator(torch.nn.Module):
    """Generate causal harmonic excitation from predicted f0."""

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.rand_ini = torch.rand(1, self.dim)
        self.rand_ini[:, 0] = 0
        self.sine_waves = torch.rand(1, 300 * samp_rate, self.dim)

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values, source_phase: torch.Tensor | None = None, return_phase_at_sample: int | None = None):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        if source_phase is None and return_phase_at_sample is not None:
            if self.training is False:
                rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
            else:
                rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                         scale_factor=1 / self.upsample_scale,
                                                         mode="linear").transpose(1, 2)
            phase_cycles = torch.cumsum(rad_values, dim=1) * self.upsample_scale
            phase = torch.nn.functional.interpolate((phase_cycles * 2 * np.pi).transpose(1, 2),
                                                    scale_factor=self.upsample_scale,
                                                    mode="nearest").transpose(1, 2)
            sines = torch.sin(phase)
            if int(return_phase_at_sample) <= 0:
                phase_state = phase_cycles[:, 0, :].detach() % 1
            else:
                frame = max(0, min(int(return_phase_at_sample) // self.upsample_scale - 1, phase_cycles.shape[1] - 1))
                phase_state = phase_cycles[:, frame, :].detach() % 1
            return sines, phase_state

        if source_phase is not None:
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                         scale_factor=1 / self.upsample_scale,
                                                         mode="linear").transpose(1, 2)
            phase_cycles = torch.cumsum(rad_values, dim=1) * self.upsample_scale
            if source_phase is None:
                source_phase = self.rand_ini.to(f0_values.device) if self.training is False else torch.zeros(
                    1, f0_values.shape[2], device=f0_values.device, dtype=f0_values.dtype)
            source_phase = source_phase.to(device=f0_values.device, dtype=f0_values.dtype)
            if source_phase.shape[0] == 1 and f0_values.shape[0] > 1:
                source_phase = source_phase.expand(f0_values.shape[0], -1)
            source_phase = source_phase.contiguous().view(f0_values.shape[0], 1, f0_values.shape[2])
            phase_cycles = phase_cycles + source_phase
            phase = torch.nn.functional.interpolate((phase_cycles * 2 * np.pi).transpose(1, 2),
                                                    scale_factor=self.upsample_scale,
                                                    mode="nearest").transpose(1, 2)
            sines = torch.sin(phase)
            phase_state = None
            if return_phase_at_sample is not None:
                if int(return_phase_at_sample) <= 0:
                    phase_state = source_phase[:, 0, :].detach() % 1
                else:
                    frame = max(0, min(int(return_phase_at_sample) // self.upsample_scale - 1, phase_cycles.shape[1] - 1))
                    phase_state = phase_cycles[:, frame, :].detach() % 1
            return sines, phase_state

        # initial phase noise (no noise for fundamental component)
        if self.training is False:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        else:
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantaneous phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                     scale_factor=1 / self.upsample_scale,
                                                     mode="linear").transpose(1, 2)

        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                                                scale_factor=self.upsample_scale, mode="nearest").transpose(1, 2)
        sines = torch.sin(phase)
        return sines, None

    def forward(
        self,
        f0,
        source_phase: torch.Tensor | None = None,
        source_sample_offset: int = 0,
        return_phase_at_sample: int | None = None,
    ):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves, phase_state = self._f02sine(
            fn,
            source_phase=source_phase,
            return_phase_at_sample=return_phase_at_sample,
        )
        sine_waves = sine_waves * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if self.training is False:
            noise_base = self._slice_noise(self.sine_waves, source_sample_offset, sine_waves.shape[1]).to(sine_waves.device)
            noise = noise_amp * noise_base
        else:
            noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        if return_phase_at_sample is not None:
            return sine_waves, uv, noise, phase_state
        return sine_waves, uv, noise

    def _slice_noise(self, noise: torch.Tensor, start: int, length: int) -> torch.Tensor:
        start = max(int(start), 0)
        end = start + int(length)
        if end <= noise.shape[1]:
            return noise[:, start:end]
        repeats = int(np.ceil(end / noise.shape[1]))
        return noise.repeat(1, repeats, 1)[:, start:end]


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = HarmonicSourceGenerator(
            sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        self.uv = torch.rand(1, 300 * sampling_rate, 1)

    def forward(self, x, source_phase: torch.Tensor | None = None, source_sample_offset: int = 0, return_phase_at_sample: int | None = None):
        """
        source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_out = self.l_sin_gen(
                x,
                source_phase=source_phase,
                source_sample_offset=source_sample_offset,
                return_phase_at_sample=return_phase_at_sample,
            )
            if return_phase_at_sample is not None:
                sine_wavs, uv, _, phase_state = sine_out
            else:
                sine_wavs, uv, _ = sine_out
                phase_state = None
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = self._slice_noise(self.uv, source_sample_offset, uv.shape[1]).to(uv.device) * self.sine_amp / 3
        if return_phase_at_sample is not None:
            return sine_merge, noise, uv, phase_state
        return sine_merge, noise, uv

    def _slice_noise(self, noise: torch.Tensor, start: int, length: int) -> torch.Tensor:
        start = max(int(start), 0)
        end = start + int(length)
        if end <= noise.shape[1]:
            return noise[:, start:end]
        repeats = int(np.ceil(end / noise.shape[1]))
        return noise.repeat(1, repeats, 1)[:, start:end]


class CausalHiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 24000,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = [8, 5, 3],
            upsample_kernel_sizes: List[int] = [16, 11, 7],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            conv_pre_look_right: int = 4,
            f0_predictor: torch.nn.Module = None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold)
        self.upsample_rates = upsample_rates
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type='right')
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    CausalConv1dUpsample(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u,
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    CausalConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1, causal_type='left')
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u)
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type='left'))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0), finalize: bool = True) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, :-self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right:])
            s_stft_real = s_stft_real[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[:, :, :-int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        if finalize is False:
            x = x[:, :-int(np.prod(self.upsample_rates) * self.istft_params['hop_len'])]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def _source_stft_stateful(
        self,
        s: torch.Tensor,
        current_samples: int,
        state: Dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = state or {}
        left_samples = self.istft_params["n_fft"] // 2
        hop = self.istft_params["hop_len"]
        previous_tail = state.get("source_tail")
        if previous_tail is not None:
            previous_tail = previous_tail.to(device=s.device, dtype=s.dtype)
            stft_input = torch.cat([previous_tail, s], dim=2)
            start_frame = previous_tail.shape[2] // hop
        else:
            stft_input = s
            start_frame = 0
        real, imag = self._stft(stft_input.squeeze(1))
        frames = min(real.shape[2] - start_frame, s.shape[2] // hop + 1)
        real = real[:, :, start_frame:start_frame + frames]
        imag = imag[:, :, start_frame:start_frame + frames]

        committed_source = s[:, :, :current_samples]
        if committed_source.shape[2] >= left_samples:
            new_tail = committed_source[:, :, -left_samples:].detach()
        elif previous_tail is None:
            new_tail = F.pad(committed_source, (left_samples - committed_source.shape[2], 0)).detach()
        else:
            new_tail = torch.cat([previous_tail, committed_source], dim=2)[:, :, -left_samples:].detach()
        return torch.cat([real, imag], dim=1), new_tail

    def decode_stateful(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        current_frames: int,
        state: Dict | None = None,
    ) -> tuple[torch.Tensor, Dict]:
        state = state or {}
        current_frames = int(current_frames)
        emit_frames = x.shape[2]
        current_samples = current_frames * int(np.prod(self.upsample_rates) * self.istft_params["hop_len"])
        emit_samples = emit_frames * int(np.prod(self.upsample_rates) * self.istft_params["hop_len"])
        x_current = x
        conv_pre_cache = x[:, :, emit_frames:emit_frames + self.conv_pre_look_right]
        if conv_pre_cache.shape[2] < self.conv_pre_look_right:
            conv_pre_cache = F.pad(conv_pre_cache, (0, self.conv_pre_look_right - conv_pre_cache.shape[2]), value=0.0)
        x = self.conv_pre(x_current, conv_pre_cache)
        s_stft, new_source_tail = self._source_stft_stateful(s, current_samples, state)

        started = bool(state.get("started", False))
        new_state = {
            "started": True,
            "source_tail": new_source_tail,
            "ups": [],
            "source_downs": [],
            "source_resblocks": [],
            "resblocks": [],
        }
        up_caches = list(state.get("ups", []))
        source_down_caches = list(state.get("source_downs", []))
        source_resblock_states = list(state.get("source_resblocks", []))
        resblock_states = list(state.get("resblocks", []))

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            up_cache = up_caches[i] if i < len(up_caches) else torch.zeros(0, 0, 0)
            up_commit_len = current_frames * int(np.prod(self.upsample_rates[:i]))
            x, new_up_cache = self.ups[i].forward_with_cache(x, up_cache, commit_len=up_commit_len)
            new_state["ups"].append(new_up_cache)

            if i == self.num_upsamples - 1:
                if started:
                    source_stft_i = s_stft[:, :, 1:]
                else:
                    x = self.reflection_pad(x)
                    source_stft_i = s_stft
            else:
                source_stft_i = s_stft

            source_down_cache = source_down_caches[i] if i < len(source_down_caches) else torch.zeros(0, 0, 0)
            if i == self.num_upsamples - 1:
                source_commit_len = current_samples // self.istft_params["hop_len"]
                if not started:
                    source_commit_len += 1
            else:
                source_commit_len = current_frames * int(np.prod(self.upsample_rates[:i + 1]))
            if i == self.num_upsamples - 1:
                source_down_commit_len = source_commit_len
            else:
                source_down_commit_len = current_samples // self.istft_params["hop_len"]
            if hasattr(self.source_downs[i], "forward_with_cache"):
                si, new_source_down_cache = self.source_downs[i].forward_with_cache(
                    source_stft_i,
                    source_down_cache,
                    commit_len=source_down_commit_len,
                )
            else:
                si = self.source_downs[i](source_stft_i)
                new_source_down_cache = torch.zeros(0, 0, 0, dtype=si.dtype, device=si.device)
            new_state["source_downs"].append(new_source_down_cache)

            source_resblock_state = source_resblock_states[i] if i < len(source_resblock_states) else None
            si, new_source_resblock_state = self.source_resblocks[i].forward_stateful(
                si,
                source_resblock_state,
                commit_len=source_commit_len,
            )
            new_state["source_resblocks"].append(new_source_resblock_state)
            if si.shape[2] != x.shape[2]:
                frames = min(si.shape[2], x.shape[2])
                si = si[:, :, :frames]
                x = x[:, :, :frames]
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                block_index = i * self.num_kernels + j
                block_state = resblock_states[block_index] if block_index < len(resblock_states) else None
                block_out, new_block_state = self.resblocks[block_index].forward_stateful(
                    x,
                    block_state,
                    commit_len=source_commit_len,
                )
                new_state["resblocks"].append(new_block_state)
                if xs is None:
                    xs = block_out
                else:
                    xs += block_out
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        conv_post_cache = state.get("conv_post", torch.zeros(0, 0, 0))
        conv_post_commit_len = current_samples // self.istft_params["hop_len"]
        if not started:
            conv_post_commit_len += 1
        x, new_conv_post_cache = self.conv_post.forward_with_cache(
            x,
            conv_post_cache,
            commit_len=conv_post_commit_len,
        )
        new_state["conv_post"] = new_conv_post_cache

        boundary = state.get("boundary_spec")
        boundary_frame = current_samples // self.istft_params["hop_len"]
        if boundary is not None:
            boundary = boundary.to(device=x.device, dtype=x.dtype)
            x_for_istft = torch.cat([boundary, x[:, :, :boundary_frame + 1]], dim=2)
            audio_offset = self.istft_params["hop_len"]
            new_boundary = x[:, :, max(0, boundary_frame - 2):boundary_frame]
        else:
            x_for_istft = x[:, :, :boundary_frame + 2]
            audio_offset = 0
            new_boundary = x[:, :, max(0, boundary_frame - 1):boundary_frame + 1]
        if new_boundary.shape[2] < 2 and x.shape[2] > 0:
            new_boundary = x[:, :, :min(2, x.shape[2])]
        new_state["boundary_spec"] = new_boundary.detach()

        magnitude = torch.exp(x_for_istft[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x_for_istft[:, self.istft_params["n_fft"] // 2 + 1:, :])
        audio = self._istft(magnitude, phase)
        audio = audio[:, audio_offset:audio_offset + current_samples]
        if audio.shape[1] < current_samples:
            audio = F.pad(audio, (0, current_samples - audio.shape[1]), value=0.0)
        audio = torch.clamp(audio, -self.audio_limit, self.audio_limit)
        return audio, new_state

    @torch.inference_mode()
    def inference(
        self,
        speech_feat: torch.Tensor,
        finalize: bool = True,
        source_sample_offset: int = 0,
        source_phase: torch.Tensor | None = None,
        return_source_state_at_sample: int | None = None,
    ) -> torch.Tensor:
        # mel->f0 NOTE f0_predictor precision is crucial for causal inference, move self.f0_predictor to cpu if necessary
        if speech_feat.device.type == "mps" or getattr(speech_feat, "is_mps", False):
            self.f0_predictor.cpu()
            self.f0_predictor.double()
            f0_input = speech_feat.cpu().double()
        else:
            self.f0_predictor.to(device=speech_feat.device)
            self.f0_predictor.double()
            f0_input = speech_feat.double()
        f0 = self.f0_predictor(f0_input, finalize=finalize).to(device=speech_feat.device, dtype=speech_feat.dtype)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        source_out = self.m_source(
            s,
            source_phase=source_phase,
            source_sample_offset=source_sample_offset,
            return_phase_at_sample=return_source_state_at_sample,
        )
        if return_source_state_at_sample is not None:
            s, _, _, source_state = source_out
        else:
            s, _, _ = source_out
            source_state = None
        s = s.transpose(1, 2)
        if finalize is True:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(x=speech_feat[:, :, :-self.f0_predictor.condnet[0].causal_padding], s=s, finalize=finalize)
        if return_source_state_at_sample is not None:
            return generated_speech, {"source": s, "source_phase": source_state}
        return generated_speech, s

    @torch.inference_mode()
    def inference_stateful(
        self,
        speech_feat: torch.Tensor,
        current_frames: int,
        state: Dict | None = None,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, Dict]:
        state = state or {}
        current_frames = int(current_frames)
        sample_per_frame = int(np.prod(self.upsample_rates) * self.istft_params["hop_len"])

        if speech_feat.device.type == "mps" or getattr(speech_feat, "is_mps", False):
            self.f0_predictor.cpu()
            self.f0_predictor.double()
            f0_input = speech_feat.cpu().double()
        else:
            self.f0_predictor.to(device=speech_feat.device)
            self.f0_predictor.double()
            f0_input = speech_feat.double()

        source_extra_frames = 0 if finalize else 1
        f0_right = self.f0_predictor.condnet[0].causal_padding
        max_emit_frames = f0_input.shape[2] if finalize else max(0, f0_input.shape[2] - f0_right)
        emit_frames = min(max_emit_frames, current_frames + source_extra_frames)
        f0, f0_state = self.f0_predictor.forward_stateful(
            f0_input,
            state=state.get("f0"),
            emit_frames=emit_frames,
            commit_frames=current_frames,
            finalize=finalize,
        )
        f0 = f0.to(device=speech_feat.device, dtype=speech_feat.dtype)

        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        source_sample_offset = int(state.get("source_sample_offset", 0))
        current_samples = current_frames * sample_per_frame
        source_out = self.m_source(
            s,
            source_phase=state.get("source_phase"),
            source_sample_offset=source_sample_offset,
            return_phase_at_sample=current_samples,
        )
        s, _, _, source_phase = source_out
        s = s.transpose(1, 2)

        audio, decode_state = self.decode_stateful(
            x=speech_feat,
            s=s,
            current_frames=current_frames,
            state=state.get("decode"),
        )
        return audio, {
            "f0": f0_state,
            "decode": decode_state,
            "source_phase": source_phase.detach() if source_phase is not None else None,
            "source_sample_offset": source_sample_offset + current_samples,
        }


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})
    model = configs['hift']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len, chunk_size, context_size = 300, 30, 8
    mel = torch.rand(1, 80, max_len).to(device)
    pred_gt, _ = model.inference(mel)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(mel[:, :, : i + chunk_size + context_size], finalize=finalize)
        pred_chunk = pred_chunk[:, i * 480:]
        print((pred_gt[:, i * 480:i * 480 + pred_chunk.shape[1]] - pred_chunk).abs().max().item())
