#!/usr/bin/env bash
# One-time setup (needs the network): clone and build the latest jitLLM, set up the JVector demo.
# Re-run to update jitLLM to the latest main.
set -euo pipefail
source "$(dirname "$0")/env.sh"

step "jitLLM: latest main"
if [ -d "$JITLLM_DIR/.git" ]; then git -C "$JITLLM_DIR" pull -q --ff-only; else git clone -q https://github.com/beehive-lab/jitllm.git "$JITLLM_DIR"; fi
git -C "$JITLLM_DIR" log --oneline -1
export JAVA_HOME="$HOME/.sdkman/candidates/java/21.0.2-open" PATH="$HOME/.sdkman/candidates/java/21.0.2-open/bin:$PATH"
(cd "$JITLLM_DIR" && { scripts/tornadovm-dev.sh status >/dev/null 2>&1 || scripts/tornadovm-dev.sh setup --backend cuda --jdk 21; } \
    && scripts/tornadovm-dev.sh build clean package -DskipTests -q)

step "JVector demo (demos repo, demo 26): JVector + jvector-gpu jars, cuVS"
use_cuvs_sdk
bash "$DEMOS_REPO/demos/26-jvector-gpu-index/setup.sh"

step "JVector showcase (demos repo, demo 27): jars from the local branch with compaction + PQ, data"
JVECTOR_REPO="file://$JVECTOR_LOCAL" bash "$DEMOS_REPO/demos/27-jvector-gpu-showcase/setup.sh"
# the data dir: JVector's public ada-002 files, plus JVector's CPU-built 1M graph and 4 segments from the benchmarks here
# (prepare.sh would rebuild those two in ~10 min if they were missing)
mkdir -p "$SHOWCASE_DATA/segments"
for f in ada_002_100k_base_99287.fvecs ada_002_100k_query_10000.fvecs ada_002_100k_gt_ip_100.ivecs \
         ada_002_1M_base_982790.fvecs ada_002_1M_query_10000.fvecs ada_002_1M_gt_ip_100.ivecs; do
    ln -sf "$HOME/jvector-work/datasets/public/$f" "$SHOWCASE_DATA/$f"
done
ln -sf "$HOME/jvector-work/reeval-1M/cpu-graph.bin" "$SHOWCASE_DATA/cpu-graph-1M.bin"
for p in 0 1 2 3; do ln -sf "$HOME/jvector-work/compaction-1M/segment-$p-of-4.idx" "$SHOWCASE_DATA/segments/segment-$p-of-4.idx"; done
bash "$DEMOS_REPO/demos/27-jvector-gpu-showcase/prepare.sh" "$SHOWCASE_DATA"
step "done: bash check.sh"
