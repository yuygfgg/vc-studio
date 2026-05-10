from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/tmp/cosyvoice_hf_cache")

ROOT_DIR = Path(__file__).resolve().parents[2]
MATCHA_DIR = ROOT_DIR / "third_party" / "Matcha-TTS"
if MATCHA_DIR.exists():
    sys.path.insert(0, str(MATCHA_DIR))

import torch
from omegaconf import DictConfig

from cosyvoice.flow.DiT.dit import DiT
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.flow.flow_matching import CausalConditionalCFM
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
from cosyvoice.vc.audio import AudioFeatureExtractor
from cosyvoice.vc.device import select_device


class VCOnlyModel:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        ort_provider: str = "auto",
        coreml_cache_dir: str | Path | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = select_device(device)
        self.coreml_cache_dir = Path(coreml_cache_dir) if coreml_cache_dir is not None else self.model_dir / ".ort_coreml_cache"
        self.sample_rate = 24000
        self.token_mel_ratio = 2

        self._check_required_files()
        self.features = AudioFeatureExtractor(
            self.model_dir,
            ort_provider=ort_provider,
            speech_tokenizer_name="speech_tokenizer_v3.onnx",
            coreml_cache_dir=self.coreml_cache_dir,
        )
        self.flow = build_cosyvoice3_flow().to(self.device).eval()
        self.hift = build_cosyvoice3_hift().eval()
        self._load_weights()

    def _load_weights(self) -> None:
        self._check_required_files()
        self.flow.load_state_dict(torch.load(self.model_dir / "flow.pt", map_location="cpu", weights_only=True), strict=True)
        hift_state = torch.load(self.model_dir / "hift.pt", map_location="cpu", weights_only=True)
        hift_state = {key.replace("generator.", ""): value for key, value in hift_state.items()}
        self.hift.load_state_dict(hift_state, strict=True)
        self.flow.to(self.device).eval()
        self._place_hift()

    def _place_hift(self) -> None:
        self.hift.to(self.device).eval()
        if self.device.type == "mps":
            self.hift.f0_predictor.cpu()
            self.hift.f0_predictor.double()

    def _check_required_files(self) -> None:
        required = self._required_files()
        missing = [str(self.model_dir / name) for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError("missing model files:\n" + "\n".join(missing))

    def _required_files(self) -> list[str]:
        return ["flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx"]


def align_prompt_token_feat(token: torch.Tensor, feat: torch.Tensor, token_mel_ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_frames = min(token.shape[1], feat.shape[1] // token_mel_ratio)
    if token_frames <= 0:
        raise ValueError("prompt wav is too short after token/feature extraction")
    return token[:, :token_frames], feat[:, : token_frames * token_mel_ratio]


def build_cosyvoice3_flow() -> CausalMaskedDiffWithDiT:
    estimator = DiT(
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        static_chunk_size=50,
        num_decoding_left_chunks=-1,
    )
    decoder = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=flow_matching_config(),
        estimator=estimator,
    )
    return CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=6561,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3),
        decoder=decoder,
    )


def build_cosyvoice3_hift() -> CausalHiFTGenerator:
    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    )


def flow_matching_config() -> DictConfig:
    return DictConfig(
        {
            "sigma_min": 1e-6,
            "solver": "euler",
            "t_scheduler": "cosine",
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": "l1",
        }
    )
