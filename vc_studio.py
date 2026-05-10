#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vc_studio_ui import launch_gui


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice VC Studio GUI")
    parser.add_argument("--model-dir", default="", help="Prefill model directory")
    parser.add_argument("--source", default="", help="Prefill offline source wav")
    parser.add_argument("--voice-package", default="", help="Prefill .cvvoice package for inference")
    parser.add_argument("--prompt", default="", help="Prefill legacy target speaker reference wav")
    parser.add_argument("--output", default="out/vc_streaming.wav", help="Prefill offline output wav path")
    parser.add_argument("--csv", default="", help="Prefill optional offline CSV path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Prefill torch device")
    parser.add_argument(
        "--ort-provider",
        default="auto",
        choices=["auto", "cpu", "cuda", "coreml"],
        help="Prefill ONNX Runtime provider",
    )
    parser.add_argument("--coreml-cache-dir", default="", help="Prefill CoreML cache directory")
    parser.add_argument("--chunk-sec", type=float, default=2.0, help="Prefill source chunk size")
    parser.add_argument("--tokenizer-chunk-sec", type=float, default=0.0, help="Prefill tokenizer chunk size; 0 follows chunk size")
    parser.add_argument("--tokenizer-left-context-sec", type=float, default=0.5, help="Prefill tokenizer left context")
    parser.add_argument("--tokenizer-right-context-sec", type=float, default=0.2, help="Prefill tokenizer right context")
    parser.add_argument("--history-sec", type=float, default=3.0, help="Prefill flow left history")
    parser.add_argument("--mel-overlap-sec", type=float, default=0.25, help="Prefill mel overlap")
    parser.add_argument("--delayed-commit-sec", type=float, default=0.5, help="Prefill delayed commit")
    parser.add_argument("--audio-declick-ms", type=float, default=0.0, help="Prefill waveform de-click")
    parser.add_argument("--audio-blend-ms", type=float, default=0.0, help="Prefill waveform crossfade")
    parser.add_argument("--enable-vad", action="store_true", help="Prefill Silero VAD noise gate on")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Prefill Silero VAD speech threshold")
    parser.add_argument("--vad-min-speech-ms", type=float, default=100.0, help="Prefill Silero VAD minimum speech duration")
    parser.add_argument("--vad-min-silence-ms", type=float, default=100.0, help="Prefill Silero VAD minimum silence duration")
    parser.add_argument("--vad-speech-pad-ms", type=float, default=30.0, help="Prefill Silero VAD speech padding")
    parser.add_argument("--flow-context", default="streaming", choices=["streaming", "window-full"], help="Prefill flow context")
    parser.add_argument("--hift-mode", default="stateful", choices=["window", "stateful"], help="Prefill HiFT mode")
    parser.add_argument("--disable-prompt-kv-cache", action="store_true", help="Prefill prompt cache off")
    parser.add_argument("--disable-history-kv-cache", action="store_true", help="Prefill history cache off")
    parser.add_argument("--prompt-cache-max-mb", type=float, default=1024.0, help="Prefill prompt KV cache memory budget; 0 disables the limit")
    parser.add_argument("--prompt-cache-max-seconds", type=float, default=0.0, help="Prefill maximum prompt seconds to cache; 0 follows memory budget")
    parser.add_argument(
        "--prompt-cache-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Prefill prompt KV cache storage dtype",
    )
    parser.add_argument(
        "--prompt-cache-storage",
        default="device",
        choices=["device", "cpu_offload"],
        help="Prefill prompt KV cache storage location",
    )
    parser.add_argument("--enable-grouped-prompt", action="store_true", help="Prefill multi-branch grouped prompt fusion on")
    parser.add_argument(
        "--enable-grouped-prompt-cache",
        action="store_true",
        help="Prefill multi-branch grouped prompt KV cache on",
    )
    args = parser.parse_args()
    launch_gui(args)


if __name__ == "__main__":
    main()
