# 27 — JVector on the GPU: the showcase

A staged, colorful terminal show of the JVector GPU work, on **real OpenAI ada-002 embeddings**. It has five acts,
each with live progress bars and bar charts, and each ends in a quality check. Run all five back to back (about
1.5 minutes), or pick acts. [Demo 26](../26-jvector-gpu-index/) is the minimal version: one build, one comparison.

| act | what runs live | result on the RTX 4090 host |
|---|---|---|
| 1 `race` | 100k × 1536: JVector's CPU build (32 threads, live progress bar) vs. the GPU build | **15.6 s → 2.1 s, 7.6×**, recall@10 0.987 / 0.988 |
| 2 `scale` | 1M × 1536 GPU build with its phase breakdown, against JVector's measured 267 s | **267.4 s → 9.2 s, 29×** |
| 3 `search` | the same JVector search on the GPU-built and JVector's CPU-built 1M graph | GPU graph: higher recall *and* lower p50 at every beam |
| 4 `compact` | merge 4 JVector segments into one 1M index on the GPU, against JVector's measured compactor | **294.6 s → 14.7 s, 20×**, recall@10 0.971 |
| 5 `pq` | product quantization of 1M vectors, CPU and GPU both live | **13.8 s → 3.6 s, 3.8×**, JVector's codes up to float ties |

```
  ━━ Act 2 · Scale ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    982,790 ada-002 embeddings on the GPU, live; JVector's CPU build measured on this machine

    GPU  TornadoVM + cuVS    ██████████████████████████████████████████████    9.2 s done

    where the GPU build's time goes
    ████████████████████████████████████████████████████████████
    ■ rows 0.3 s   ■ candidates 5.4 s   ■ prune 2.9 s   ■ install 0.5 s   ■ upper layers 0.2 s

    graph build, 1M
    JVector CPU (measured)         ██████████████████████████████████████████████   267.4 s
    GPU (live)                     ██                                                 9.2 s

    ▶ 29.0x faster

  ╔══════════════════════════════════════════════════════════════════════╗
  ║  Scoreboard                   JVector CPU -> GPU                      ║
  ╟──────────────────────────────────────────────────────────────────────╢
  ║  Build 100k x 1536              15.6 s ->   2.1 s    7.6x             ║
  ║  Build 1M x 1536               267.4 s ->   9.2 s   29.0x             ║
  ║  Search at equal recall       p50 up to 19% lower, GPU graph          ║
  ║  Compaction 4 -> 1 (1M)        294.6 s ->  14.7 s   20.0x             ║
  ║  PQ train + encode (1M)         13.8 s ->   3.6 s    3.8x             ║
  ╚══════════════════════════════════════════════════════════════════════╝
JVectorShowcase: PASSED -- faster on the GPU, and every quality check holds
```

## What does this demonstrate?

* **A real Java library, accelerated without leaving Java.** JVector, the vector index inside Apache Cassandra,
  gets its graph built through a small opt-in SPI (`GraphBuildAccelerator`). The result is an ordinary JVector
  graph: search, the on-disk format and the writers are unchanged.
* **Three uses of the same GPU machinery:** graph builds, compaction (a fresh GPU build of the merged segments
  instead of stitching graphs together), and product quantization.
* **Better, not just faster.** The GPU prunes every node's list from near-exact candidates, while JVector inserts
  nodes one at a time into a growing graph. The GPU graph has higher recall with fewer visited nodes, so at equal
  recall it searches faster.

## What TornadoVM feature/API does it use?

* `KernelContext` kernels with `ctx.mma` tensor-core fragments (FP16 in, FP32 accumulate) and local memory: the
  pruning kernel and the PQ k-means and encode kernels in `jvector-gpu`.
* `TornadoExecutionPlan.withPreCompilation()`. The pruning kernels are compiled on the CPU while the GPU runs
  cuVS.
* `tornado-cuvs` (`CuVS.allNeighborsOnHost`): NVIDIA cuVS NN-Descent from Java, through FFM, with no JNI. It is
  merged into TornadoVM `develop` (beehive-lab/TornadoVM#1155) but not yet in a release.

## Prerequisites

* A TornadoVM SDK with `tornado-cuvs` (a `develop` build after #1155), JDK 22+, and an NVIDIA GPU with tensor
  cores (Ampere or newer). About 40 GB of RAM: the 1M vectors are held on the heap.
* The JVector branch with the GPU module, compaction and PQ (`feat/gpu-build-pq-compaction`). Set `JVECTOR_REPO`
  to the repository that has it; a local clone works (`file:///path/to/jvector`).
* JVector's public ada-002 datasets in one directory:

  ```bash
  base=https://jvector-datasets-public.s3.us-east-1.amazonaws.com/datasets-clean
  for f in ada_002_100k_base_99287.fvecs ada_002_100k_query_10000.fvecs ada_002_100k_gt_ip_100.ivecs \
           ada_002_1M_base_982790.fvecs ada_002_1M_query_10000.fvecs ada_002_1M_gt_ip_100.ivecs; do
      curl -L -o "$DATA/$f" "$base/$f"
  done
  ```

## How do I run it?

```bash
export TORNADOVM_HOME=<SDK with tornado-cuvs>; export PATH=$TORNADOVM_HOME/bin:$PATH
bash demos/27-jvector-gpu-showcase/setup.sh            # once: JVector jars into lib/, cuVS
bash demos/27-jvector-gpu-showcase/prepare.sh $DATA    # once, ~10 min: JVector's CPU graph and 4 segments

bash demos/27-jvector-gpu-showcase/run.sh $DATA                 # all five acts, [enter] between acts
NO_PAUSE=1 bash demos/27-jvector-gpu-showcase/run.sh $DATA      # back to back (~1.5 min)
bash demos/27-jvector-gpu-showcase/run.sh $DATA race scale      # just some acts
bash demos/27-jvector-gpu-showcase/run.sh java $DATA            # java @argfile instead of the tornado launcher
```

## How do I validate the result?

Every act ends with a check, listed under the scoreboard. The run prints `PASSED` only if all of them hold:

* **race:** the GPU graph's recall@10 is at least JVector's minus 0.01 (1000 queries, exact ground truth).
* **search:** at beams 20/40/80 the GPU graph's recall@10 is at least JVector's minus 0.005.
* **compact:** the merged index has recall@10 ≥ 0.95.
* **pq:** encoding with JVector's own codebooks on the GPU gives JVector's codes. The only exception is a vector
  where two centroids are equally close up to float rounding (relative difference ≤ 1e-6). About 15 of the 982,790
  vectors hit such a tie.

## What should the presenter point out?

* **Act 1:** the yellow CPU bar fills for about 15 s while you talk, then the GPU finishes in about 2 s. Point at
  the recall line: same quality.
* **Act 2:** the phase bar shows where the time goes. *Candidates* is NVIDIA cuVS; *prune* is a Java kernel on
  tensor cores. Both are called from one Java program. The CPU bar is measured, not live: it is 4.5 minutes.
* **Act 3:** the same JVector search code on both graphs. The GPU graph wins on recall *and* latency at every
  beam.
* **Act 4:** 4 segments, as a database writes them, merged in about 15 s instead of 5 minutes.
* **Act 5:** both sides live. 3.8× faster, and the GPU returns JVector's own codes.

If something misbehaves on stage, `run.sh $DATA race` takes about 20 s and needs no prepared files. The logs in
`results/raw/47-jvector-gpu-showcase/` show full runs.

## What GPU/CUDA requirements exist?

* An NVIDIA GPU with tensor cores (sm_80+), and about 4 GB of device memory for the 1M FP16 vectors and
  workspaces. `run.sh` sets `-Dtornado.device.memory=8GB`.
* cuVS 26.08 and the CUDA 12 libraries it needs come from the wheels `setup.sh` downloads.

## Provenance

Captured on the sm_89 host: RTX 4090, driver 610.57.04, JDK 25.0.2, i9-13900K (32 threads). TornadoVM is a source
build with #1155 (`7.0.2-jdk22plus-dev`); JVector is `feat/gpu-build-pq-compaction` at `e7f71db`. The two
reference CPU times (267.4 s build, 294.6 s compaction) were measured on the same machine with the same JVector
settings. See `results/raw/47-jvector-gpu-showcase/MANIFEST.md`.
