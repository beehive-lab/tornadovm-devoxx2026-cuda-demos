#!/usr/bin/env bash
# Hybrid API (slides 14-16): a Java kernel, cuBLAS and a Java kernel in one task graph, then CUDA-graph replay.
# About 5 seconds.     bash demoHybrid.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_tornadovm_7

step "1/2  JIT -> cuBLAS sgemv -> JIT, one task graph, shared device buffers   (slide 15)"
note "scale (Java kernel) -> CuBlas::cublasSgemv (NVIDIA) -> bias (Java kernel); no host round trip in between"
compile_demo 04-cublas-hybrid
cd "$BUILD/04-cublas-hybrid"
# one line per task: which code ran (Java kernel or cuBLAS), on which backend, and its kernel time
run "tornado --enableProfiler console --classpath . CuBlasSgemvHybrid 8 8 3 2>&1 | awk -f '$DEMO_ROOT/tasks.awk'"
pause

step "2/2  The same kind of graph, captured once as a CUDA graph and replayed   (slide 16)"
note "plan.withCUDAGraph(): first run captures, every later run is one cuGraphLaunch"
compile_demo 07-cuda-graph-benefit
cd "$BUILD/07-cuda-graph-benefit"
run "tornado --classpath . CudaGraphBenefit 4096 6 50 both 2>&1 | grep -E 'steady-state|speedup|All executions'"
