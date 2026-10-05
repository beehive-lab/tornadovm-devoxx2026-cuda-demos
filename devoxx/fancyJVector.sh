#!/usr/bin/env bash
# JVector + cuVS, the fancy version of demoJVector.sh (live demo 3, slides 29-31): demo 26 drawn as a GPU time
# breakdown and a bar chart. For the five-act version (race, 1M, search, compaction, PQ) use demoShowcase.sh.
#   bash fancyJVector.sh [ada|big|quick]     ada: real OpenAI ada-002, 99k x 1536 (~20 s)
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_cuvs_sdk
demo="$DEMOS_REPO/demos/26-jvector-gpu-index"
case "${1:-ada}" in
    ada)   what="99k OpenAI ada-002 embeddings x 1536"; args="--fvecs $ADA_100K" ;;
    big)   what="500k x 1024 synthetic embeddings"; args="" ;;
    quick) what="100k x 768 synthetic embeddings"; args="100000 768" ;;
    *)     sed -n '4,5p' "$0"; exit 1 ;;
esac

$FANCY banner "JVector on the GPU · TornadoVM + NVIDIA cuVS" "the vector index inside Apache Cassandra, built by a Java tensor-core kernel"
$FANCY act 1 "GPU vs JVector's own CPU build" "$what · the same graph settings · then the same search on both graphs"
# shellcheck disable=SC2086
spin "GPU build, then JVector on 32 CPU threads" "$FANCY_LOGS/jvector.log" bash "$demo/run.sh" tornado $args
$FANCY jvector "$FANCY_LOGS/jvector.log"
$FANCY scoreboard "JVector on the RTX 4090"
