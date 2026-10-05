#!/usr/bin/env bash
# JVector + cuVS (live demo 3, slides 29-31): the same JVector index built on the GPU and on the CPU.
#   bash demoJVector.sh           real OpenAI ada-002 embeddings, 99k x 1536 (~20 s)
#   bash demoJVector.sh big       500k x 1024 synthetic embeddings (~1 min, bigger speed-up)
#   bash demoJVector.sh quick     100k x 768 synthetic (~13 s, the fallback)
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_cuvs_sdk
demo="$DEMOS_REPO/demos/26-jvector-gpu-index"

case "${1:-ada}" in
    ada)   step "JVector index of OpenAI ada-002 embeddings, built on the GPU and on the CPU"
           args="--fvecs $ADA_100K" ;;
    big)   step "JVector index of 500,000 x 1024 embeddings, built on the GPU and on the CPU"
           args="" ;;
    quick) step "JVector index of 100,000 x 768 embeddings, built on the GPU and on the CPU"
           args="100000 768" ;;
    *)     sed -n '3,6p' "$0"; exit 1 ;;
esac
note "GPU: cuVS NN-Descent candidates (TornadoVM cuVS provider) + JVector's diversity rule in a tensor-core Java kernel"
note "CPU: JVector's own builder on all cores. Then the same search on both graphs, against exact neighbors."
run "bash $demo/run.sh tornado $args 2>&1 | grep -vE 'WARNING|^Oct |^INFO|SLF4J|^Note:'"
