#!/usr/bin/env bash
# jitLLM, the fancy version of demoJitllm.sh (live demo 1, slides 20-22), in three acts. About 1.5 minutes.
#   1 chat    live generation in pure Java on the GPU
#   2 code    an LLM (Qwen3-4B, run by jitLLM on the GPU) writes a GPU kernel in Java; TornadoVM compiles it to CUDA and
#             runs it; the same method runs on the CPU as the reference; the result is drawn in the terminal
#   3 bench   jitLLM vs llama.cpp as bar charts
#   bash fancyJitllm.sh [chat|code|bench]     NO_PAUSE=1 to skip the [enter] between acts
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_jitllm
cd "$JITLLM_DIR"
mode="${1:-all}"

$FANCY banner "jitLLM · LLM inference in pure Java on the GPU" "TornadoVM-compiled Java kernels · NVIDIA libraries where they win"

if [ "$mode" = all ] || [ "$mode" = chat ]; then
    $FANCY act 1 "Generate" "$(basename "$MODEL_CHAT") · decode replayed as a CUDA graph · the tokens appear as they are generated"
    printf '    \033[1;36m❯\033[0m \033[1mIn two sentences: why run LLM inference in pure Java on a GPU?\033[0m\n\n    '
    # tokens stream to the terminal as they come; the JVM's warnings go to a log
    ./jitllm --gpu --cuda-graphs --model "$MODEL_CHAT" --max-new-tokens 120 \
        --prompt 'In two sentences: why run LLM inference in pure Java on a GPU?' 2>"$FANCY_LOGS/chat.err" \
        | tee "$FANCY_LOGS/chat.log" | sed -u 's/^/    /'
    # jitLLM prints its performance line on stderr
    $FANCY llm-chat "$FANCY_LOGS/chat.err"
    [ "$mode" = chat ] || fancy_pause
fi

if [ "$mode" = all ] || [ "$mode" = code ]; then
    $FANCY act 2 "The model writes a GPU kernel" "$(basename "$MODEL_CODE") runs on the GPU in jitLLM and writes Java; TornadoVM turns that Java into CUDA"
    printf '    \033[1;36m❯\033[0m \033[1m%s\033[0m\n\n' "$(fold -s -w 104 "$DEMO_ROOT/codegen/prompt.txt" | sed '2,$s/^/      /')"
    # greedy decoding (temperature 0): the same prompt gives the same kernel, so rehearsal predicts the stage
    ./jitllm run --gpu --cuda-graphs --model "$MODEL_CODE" -c 2048 --max-new-tokens 500 --temperature 0 \
        -sp "$(cat "$DEMO_ROOT/codegen/system.txt")" --prompt "$(cat "$DEMO_ROOT/codegen/prompt.txt") /no_think" \
        2>"$FANCY_LOGS/code.err" | tee "$FANCY_LOGS/code.out" | $FANCY stream
    $FANCY llm-chat "$FANCY_LOGS/code.err" "jitLLM writes the kernel"
    awk '/```java/{f=1;next} /```/{f=0} f' "$FANCY_LOGS/code.out" > "$FANCY_LOGS/kernel.java"
    fancy_pause
    printf '    \033[1mthe kernel, as written by the model\033[0m\n'
    $FANCY code "$FANCY_LOGS/kernel.java"
    # the harness: the kernel inside a class that runs it on the GPU, then on the CPU, and compares
    mkdir -p "$FANCY_LOGS/kernel"
    insert_kernel() {
        python3 - "$1" "$FANCY_LOGS/kernel/Harness.java" "$DEMO_ROOT/codegen/Harness.java" <<'PY'
import sys
kernel, out, harness = sys.argv[1], sys.argv[2], sys.argv[3]
body = open(kernel).read().rstrip().replace("\n", "\n    ")
open(out, "w").write(open(harness).read().replace("/*KERNEL*/", body))
PY
    }
    run_kernel() (
        use_tornadovm_7
        cd "$FANCY_LOGS/kernel"
        javac -cp "$TORNADO_CP" Harness.java && tornado --printKernel --classpath . Harness
    )
    insert_kernel "$FANCY_LOGS/kernel.java"
    if ! spin "javac, TornadoVM JIT to CUDA, run on GPU and CPU" "$FANCY_LOGS/kernel-run.log" run_kernel; then
        printf '\n    \033[1;31mthe live kernel did not compile:\033[0m\n'
        grep -m 6 -A 2 "error:" "$FANCY_LOGS/kernel-run.log" | sed 's/^/      /'
        printf '\n    \033[2mfalling back to the kernel the same model wrote for the same prompt in rehearsal\033[0m\n'
        insert_kernel "$DEMO_ROOT/codegen/mandelbrot.reference.java"
        spin "javac, TornadoVM JIT to CUDA, run on GPU and CPU" "$FANCY_LOGS/kernel-run.log" run_kernel || true
    fi
    echo
    $FANCY cuda "$FANCY_LOGS/kernel-run.log" 14
    fancy_pause
    $FANCY fractal "$FANCY_LOGS/kernel-run.log"
    [ "$mode" = code ] || fancy_pause
fi

if [ "$mode" = all ] || [ "$mode" = bench ]; then
    $FANCY act 3 "jitLLM vs llama.cpp" "$(basename "$MODEL_BENCH") · same GPU, same GGUF, same batch size   (slide 22)"
    spin "jitLLM prefill: cuBLAS GEMMs + Java kernels" "$FANCY_LOGS/pp.log" \
        ./jitllm bench --gpu --model "$MODEL_BENCH" --pp 512 --tg 0 --with-prefill-decode --batch-prefill-size 512 --with-native-libraries
    spin "jitLLM decode: Java kernels as a CUDA graph" "$FANCY_LOGS/tg.log" \
        ./jitllm bench --gpu --cuda-graphs --model "$MODEL_BENCH" --pp 0 --tg 128
    spin "llama.cpp: llama-bench pp512 + tg128" "$FANCY_LOGS/llama.log" \
        "$LLAMA_BENCH" -m "$MODEL_BENCH" -p 512 -n 128 -b 512 -ngl 99
    echo
    $FANCY llm-bench "$FANCY_LOGS/pp.log" "$FANCY_LOGS/tg.log" "$FANCY_LOGS/llama.log"
fi

$FANCY scoreboard "jitLLM on the RTX 4090"
