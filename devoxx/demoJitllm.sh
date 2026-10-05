#!/usr/bin/env bash
# jitLLM (live demo 1, slides 20-22): LLM inference in pure Java on the GPU, then head to head with llama.cpp.
# About 45 seconds.
#   bash demoJitllm.sh            generate, then the benchmark
#   bash demoJitllm.sh chat       generation only
#   bash demoJitllm.sh bench      jitLLM vs llama.cpp only
#   MODEL_CHAT=~/Llama-3.2-3B-Instruct-Q8_0.gguf bash demoJitllm.sh chat     any model from models.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_jitllm
cd "$JITLLM_DIR"
mode="${1:-all}"

if [ "$mode" = all ] || [ "$mode" = chat ]; then
    step "Generate with $(basename "$MODEL_CHAT") on the GPU, pure Java"
    run "./jitllm --gpu --cuda-graphs --model '$MODEL_CHAT' --max-new-tokens 120 --prompt 'In two sentences: why run LLM inference in pure Java on a GPU?' 2>&1 | grep -v WARNING"
    [ "$mode" = chat ] || pause
fi

if [ "$mode" = all ] || [ "$mode" = bench ]; then
    step "jitLLM, prefill on NVIDIA libraries: $(basename "$MODEL_BENCH"), 512-token prompt, batch 512   (slide 22)"
    note "--with-native-libraries: cuBLAS GEMMs for the prompt batch, Java kernels for everything else"
    run "./jitllm bench --gpu --model '$MODEL_BENCH' --pp 512 --tg 0 --with-prefill-decode --batch-prefill-size 512 --with-native-libraries 2>&1 | grep -E '^\[bench\] [A-Z]'"
    note "decode: one token at a time, Java kernels, replayed as a CUDA graph"
    run "./jitllm bench --gpu --cuda-graphs --model '$MODEL_BENCH' --pp 0 --tg 128 2>&1 | grep -E '^\[bench\] [A-Z]'"

    step "llama.cpp, same GPU, same GGUF, same batch size"
    run "'$LLAMA_BENCH' -m '$MODEL_BENCH' -p 512 -n 128 -b 512 -ngl 99 2>/dev/null | grep -E 'model|---|pp512|tg128'"
fi
