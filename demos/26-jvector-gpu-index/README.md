# 26 — A JVector vector index, built on the GPU

[JVector](https://github.com/datastax/jvector) is a pure-Java vector index that "merges the DiskANN and HNSW
family trees" (its README). This demo builds the **same JVector index twice**: once on the GPU (TornadoVM + NVIDIA cuVS)
and once with JVector's own multi-threaded CPU builder. It then searches both with the same code and compares
build time and recall. About one minute end to end.

```
  JVector index of 500,000 x 1024 vectors, synthetic clustered embeddings (generated in 1.1 s)

  [GPU] building with TornadoVM + cuVS ...
GPU build of 500000 x 1024 (COSINE): rows 0.58 s, candidates 2.23 s (1 batches), prune 2.64 s, install 0.18 s, upper layers 0.49 s, total 6.12 s
  [GPU] built in   6.23 s
  [CPU] building with JVector on 32 threads ...
  [CPU] built in  47.22 s

  | graph build       |    47.22 s |     6.23 s |
  | recall@10         |     0.9700 |     0.9835 |
  speed-up: 7.6x
JVectorGpuIndex: PASSED -- the GPU graph's recall@10 matches or beats JVector's
```

## What does this demonstrate?

**A real Java library, accelerated without leaving Java.** JVector builds its graph (DiskANN/Vamana-style, with
an HNSW-like hierarchy) one insertion at a time on the CPU. The GPU build plugs in through a small SPI in
JVector, `GraphBuildAccelerator`, and produces an ordinary JVector graph. Search, the on-disk format and the rest
of JVector are untouched.

**A JIT kernel and an NVIDIA library in one pipeline:**

1. **Candidates: NVIDIA cuVS NN-Descent**, called from Java through TornadoVM's cuVS module
   (beehive-lab/TornadoVM#1155), finds about 96 nearest neighbors per vector. There's no JNI; the binding uses
   the FFM API and DLPack.
2. **Diversity pruning: a TornadoVM-compiled Java kernel.** For each node it computes the 96 × 96 Gram matrix of
   the candidates with **tensor cores** (`ctx.mma`, FP16 in, FP32 accumulate, as in [demo 08](../08-tensor-core-mma/)),
   then applies JVector's alpha rule with one work-group per node.
3. Reverse edges are merged, the lists that grew are pruned again, and the upper layers are built the same way.

**Speed and quality together.** The GPU graph's recall matches or beats JVector's own on every dataset measured;
JVector's CPU build itself varies by about ±0.01 between runs. On real embeddings:

| dataset | JVector (CPU) | GPU | speed-up | recall@10 CPU / GPU |
|---|---|---|---|---|
| ada002, 99k × 1536 (`--fvecs`, below) | 15.5 s | 2.5 s | 6.2x | 0.9885 / 0.9950 |
| ada002, 1M × 1536 | 268 s | 14 s | 19x | 0.969 / 0.973 |
| Cohere Wikipedia, 10M × 1024 | 2271 s | 207 s | 11x | 0.875 / 0.897 |

The 1M and 10M rows are from the JVector work itself (`jvector-gpu/README.md` on the JVector branch), not from
this demo; they take minutes, not seconds.

## What TornadoVM feature/API does it use?

* `KernelContext` kernels with local memory, barriers and `ctx.mma` tensor-core fragments
  (`PruneKernels.gramMmaPacked`, `PruneKernels.prune` in `jvector-gpu`).
* A persistent `TornadoExecutionPlan` reused for each batch of 4096 nodes, with the vectors resident on the device.
* `tornado-cuvs` (`CuVS.allNeighborsOnHost`): cuVS on the same device. It isn't in TornadoVM 7.0.0; it comes from
  PR #1155.

## Prerequisites

* Everything in the top-level README, plus a TornadoVM SDK that includes `tornado-cuvs` (PR #1155). Until a
  release has it, build that branch, and point `TORNADOVM_HOME` at its `dist/` SDK.
* Maven and git, for the one-time setup, which builds JVector from the PR branch
  (`mikepapadim/jvector:feat/graph-build-accelerator`).
* Python 3 with pip: `setup.sh` downloads NVIDIA cuVS (2.1 GB of wheels, once) into `~/.jvector-gpu/cuvs`.

This demo is **not** in `scripts/run-all-demos.sh`. It needs the JVector jars and an SDK the default
`sdkman-7.0.0` profile doesn't have, so like demos 09 and 10 it has its own scripts.

## How do I run it?

```bash
export TORNADOVM_HOME=<TornadoVM SDK with tornado-cuvs>   # then: export PATH=$TORNADOVM_HOME/bin:$PATH
bash demos/26-jvector-gpu-index/setup.sh                  # once, before the talk (needs the network)

bash demos/26-jvector-gpu-index/run.sh                    # tornado launcher, 500k x 1024 (~1 min)
bash demos/26-jvector-gpu-index/run.sh java               # java @argfile path
bash demos/26-jvector-gpu-index/run.sh tornado 250000 768 # smaller: ~25 s
```

On real embeddings (OpenAI ada-002, 1536 dims, about 20 s):

```bash
curl -LO https://jvector-datasets-public.s3.us-east-1.amazonaws.com/datasets-clean/ada_002_100k_base_99287.fvecs
bash demos/26-jvector-gpu-index/run.sh tornado --fvecs ada_002_100k_base_99287.fvecs
```

## How do I validate the result?

The demo computes the exact 10 nearest neighbors of 200 held-out queries by brute force, and searches both
graphs with the same JVector search (top 10, rerank 50). It prints `PASSED` when the GPU graph's recall@10 is at
least the CPU graph's minus 0.01, and `FAILED` otherwise. A fast but wrong graph can't pass. If the GPU path is
unavailable (no `tornado-cuvs`, or cuVS not on `LD_LIBRARY_PATH`), it prints `FAILED -- GPU build unavailable`
instead of silently using the CPU.

## What GPU/CUDA requirements exist?

* An NVIDIA GPU with tensor cores, Ampere (sm_80) or newer (`mma.sync m16n8k16` FP16).
* Device memory for the FP16 vectors plus about 0.5 GB: 1 GB at the defaults. TornadoVM's default 4 GB budget is
  enough.
* cuVS 26.08 and the CUDA 12 math libraries it uses (cuBLAS, cuSOLVER, cuSPARSE, cuRAND, NVRTC) come from the
  wheels `setup.sh` downloads. TornadoVM's own CUDA requirements are as for every other demo.

## What should the presenter point out?

* **The GPU build is done in about 6 seconds; then the CPU builds the same graph for about 47 seconds while you
  talk.** Then point to the recall row: the GPU graph is not a rough approximation of JVector's, it matches it
  (on 500k synthetic vectors the CPU's recall itself ranged 0.964–0.986 across runs; the GPU's 0.983–0.984).
* The trace line breaks the GPU time down: `candidates` is cuVS, `prune` is the Java tensor-core kernel. Both are
  called from the same Java program.
* The speed-up grows with size: 6x at 100k, 7–8x at 500k, 19x at 1M × 1536.
* JVector needed about 190 lines in its core (the SPI and the external-build hooks), plus tests. The GPU code
  is a separate module that JVector never sees unless it's selected.

If it misbehaves on stage: `run.sh tornado 100000 768` takes about 13 s, and the logs in
`results/raw/46-jvector-gpu-index/` show a full run.

## Provenance

Captured on the sm_89 host (RTX 4090, driver 610.57.04, JDK 25.0.2, i9-13900K with 32 threads), against a source
build of TornadoVM with PR #1155 (`7.0.2-jdk22plus-dev`) and the JVector branch above. Evidence:
`results/raw/46-jvector-gpu-index/`.
