#!/usr/bin/env bash
# Every GGUF model on this machine, for jitLLM (MODEL_CHAT=... / MODEL_BENCH=... bash demoJitllm.sh).
find "$HOME" -maxdepth 4 -name '*.gguf' -size +100M 2>/dev/null \
    | grep -v '/llama.cpp-ref/models/ggml-vocab' | sort \
    | while read -r f; do printf '%8s  %s\n' "$(du -h "$f" | cut -f1)" "$f"; done
