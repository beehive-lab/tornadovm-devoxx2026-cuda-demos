#!/usr/bin/env bash
# jitLLM, live dashboard version (~30 s): one GPU, two jobs, each component lit while it works.
#   1 jitLLM engine (Java, on the GPU) runs Qwen3-4B, which writes   2 a GPU kernel in Java   3 javac compiles it
#   4 TornadoVM JIT-compiles it to CUDA   5 it runs on the same GPU: a zoom into the Mandelbrot set, drawn live.
# The GPU panel polls nvidia-smi: utilization, memory, and which process holds the GPU, in its component's color.
#   bash fancyJitllmLive.sh          [enter] at the end to leave the dashboard (NO_PAUSE=1 to exit straight away)
# Needs a terminal of at least 120 x 40.
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec python3 "$DEMO_ROOT/live/live.py" "$(mktemp -d)"
