#!/usr/bin/env bash
# JVector on the GPU, the showcase (demo 27 of the demos repo): five acts on real OpenAI ada-002 embeddings, with live
# progress bars, bar charts and a scoreboard. [enter] between acts (NO_PAUSE=1 to run them back to back).
#   bash demoShowcase.sh                 all five acts (~1.5 min of run time + your talking)
#   bash demoShowcase.sh race            act 1 only: live CPU vs GPU build race at 100k (~20 s), the fallback
#   bash demoShowcase.sh scale search    any acts: race scale search compact pq
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_cuvs_sdk
step "JVector on the GPU: ${*:-race scale search compact pq}"
note "real OpenAI ada-002 embeddings; GPU = NVIDIA cuVS candidates + a Java tensor-core kernel (TornadoVM)"
note "acts 2 and 4 compare with JVector's CPU times measured on this machine (4.5 and 5 minutes live)"
show "bash $DEMOS_REPO/demos/27-jvector-gpu-showcase/run.sh $SHOWCASE_DATA $*"
# the showcase pauses between acts itself; keep its colors, drop the JVM's start-up noise
bash "$DEMOS_REPO/demos/27-jvector-gpu-showcase/run.sh" "$SHOWCASE_DATA" "$@" 2> >(grep -vE 'WARNING|^Oct |^INFO|SLF4J|^Note:' >&2)
