#!/usr/bin/env bash
# CUDA Tile from Java (slides 17-19): threads, tiles and cuBLAS in one CUDA graph, then the tile GEMM ladder.
# About 10 seconds.     bash demoTile.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_tornadovm_7
# a tile kernel that fails to compile would otherwise fall back to the CPU and still print "correct"
BAILOUT="-Dtornado.recover.bailout=False"

step "1/2  Threads, tiles, cuBLAS: one CUDA graph   (slide 18)"
note "scale (@Parallel) -> gemm (TileContext, CUDA Tile) -> cublasSgemv -> biasRelu (JIT), captured once and replayed"
compile_demo 20-cutile-hybrid
cd "$BUILD/20-cutile-hybrid"
run "tornado --printBytecodes --jvm='$BAILOUT' --classpath . TileHybridPipeline 256 2 graph 2>&1 | grep -E 'EXECUTION_GRAPH|LAUNCH|withCUDAGraph' | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/ on  \[NVIDIA.*//'"
pause

step "2/2  Ten lines of tiles vs a hand-tuned kernel: FP16 GEMM, n = 2048   (slide 19)"
note "simple and hand-optimised KernelContext, four tile shapes, a launch hint, and cuBLAS; all checked for correctness"
note "wall clock includes host dispatch; the slide's TFLOP/s are kernel time from nsys (see the demos repo, demo 25)"
compile_demo 25-tile-ladder
cd "$BUILD/25-tile-ladder"
run "tornado --jvm='$BAILOUT' --classpath . TileLadder 2048 10 2>&1 | sed -n '/=== Summary/,\$p'"
