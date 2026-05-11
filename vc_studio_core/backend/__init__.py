from __future__ import annotations

from .environment import configure_environment

configure_environment()

from cosyvoice.vc.model import VCOnlyModel, align_prompt_token_feat
from cosyvoice.vc.voice_package import (
    APP_VERSION,
    FORMAT_NAME,
    FORMAT_VERSION,
    MEL_BINS,
    MODEL_FAMILY,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
    TOKENIZER_SAFE_SECONDS,
    TOKENIZER_SAMPLE_RATE,
    TOKEN_MEL_RATIO,
    TOKEN_RATE,
    VoicePromptBranch,
    VoicePromptInputs,
    l2_normalize_array,
    load_voice_package,
    model_compatibility_fields,
    new_package_id,
    save_voice_package,
    sha256_file,
    sharpen_weights,
    utc_now_iso,
)
from cosyvoice.vc.soft_prompt import SoftPromptTrainingConfig, distill_soft_prompt_v1

from .callbacks import _noop_log, _noop_metrics, _noop_status
from .context import prepare_stream_context, stream_context_log_lines
from .device import sync_device
from .lavasr import (
    DEFAULT_LAVASR_DEVICE,
    DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE,
    LAVASR_INPUT_SAMPLE_RATE,
    LAVASR_OUTPUT_SAMPLE_RATE,
    LavaSRBandwidthExtender,
    add_lavasr_row_stats,
    add_realtime_resample_row_stats,
    create_lavasr_extender,
    fit_audio_length,
    normalize_audio_tensor,
    prepare_lavasr_input,
    realtime_output_sample_rate,
    resample_realtime_output,
    resolve_lavasr_model_path,
    validate_lavasr_lowpass_hz,
)
from .player import RealtimeAudioPlayer
from .prompt_cache import (
    _append_branch_prompt_cache_to_sequential_steps,
    _attach_dominant_grouped_prompt_inputs,
    _create_sequential_grouped_prompt_cache_steps,
    choose_full_prompt_cache_frames,
    choose_prompt_cache_budget_frames,
    estimate_prompt_cache_bytes,
    format_bytes,
    grouped_prompt_cache_enabled,
    grouped_prompt_enabled,
    normalize_prompt_runtime_policy,
    optimize_kv_list,
    optimize_prompt_cache_storage,
    optimize_tensor,
    optimize_tensor_tree,
    prepare_grouped_prompt_cache,
    prepare_grouped_prompt_runtime_inputs,
    prepare_prompt_inputs,
    prepare_prompt_inputs_from_package,
    prepare_prompt_inputs_from_wav,
    prompt_cache_dtype_bytes,
    prompt_cache_max_seconds,
    prompt_cache_memory_limit_bytes,
    prompt_cache_offload_kv_to_cpu,
    prompt_cache_storage_dtype,
    select_runtime_prompt_inputs,
    select_soft_prompt_runtime_inputs,
    trim_cache_mel_frames_to_static,
    trim_prompt_to_static_cache,
)
from .service import VCStudioBackend
from .streaming import (
    apply_vad_speech_gate,
    blend_mel_chunks,
    blend_speech_chunks,
    concat_chunks,
    declick_speech_boundary,
    hift_required_right_context_mel,
    infer_flow_window,
    keep_mel_tail,
    make_row,
    print_chunk,
    run_window_stream,
    write_rows,
)
from .summary import offline_summary_lines
from .timing import (
    align_audio_blend_samples,
    align_audio_declick_samples,
    align_delayed_commit_tokens,
    align_history_tokens,
    align_overlap_tokens,
    is_static_cache_aligned,
)
from .tokenizers import AsyncSourceTokenizer, MicrophoneSourceTokenizer
from .types import BackendConfig, PreparedStreamContext, StreamSettings
from .vad import SileroVADGate, build_token_speech_mask, create_vad_gate
from .voice_package_builder import (
    _fuse_speaker_embeddings,
    _prepare_voice_reference,
    _resolve_raw_fusion_weights,
    _voice_prompt_inputs_from_package_parts,
    create_voice_package,
)

__all__ = [name for name in globals() if not name.startswith("__")]
