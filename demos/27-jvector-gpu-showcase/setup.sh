#!/usr/bin/env bash
# One-time setup for demo 27: JVector with the GPU module, from the branch that has the GPU build, compaction and
# product quantization, into lib/ (reuses demo 26's setup), plus NVIDIA cuVS.
#
#   source scripts/setup-env.sh          # with a TornadoVM SDK that has tornado-cuvs
#   bash demos/27-jvector-gpu-showcase/setup.sh
#
# JVECTOR_REPO / JVECTOR_BRANCH select the JVector source (a local clone works: JVECTOR_REPO=file:///path/to/jvector).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
JVECTOR_LIB="$here/lib" JVECTOR_BRANCH="${JVECTOR_BRANCH:-feat/gpu-build-pq-compaction}" \
    bash "$here/../26-jvector-gpu-index/setup.sh"
echo "== then, once: bash demos/27-jvector-gpu-showcase/prepare.sh <dataDir>"
