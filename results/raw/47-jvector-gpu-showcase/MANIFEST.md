# Batch 47: demo 27, the JVector GPU showcase

Captured 2026-10-04 on the sm_89 host (`environment.txt`): RTX 4090, driver 610.57.04, JDK 25.0.2, i9-13900K.

* **TornadoVM:** a source build with `tornado-cuvs` (beehive-lab/TornadoVM#1155, now merged into `develop`), SDK
  `7.0.2-jdk22plus-dev`. This is **not** the repo's default `sdkman-7.0.0` profile.
* **JVector:** `mikepapadim/jvector:feat/gpu-build-pq-compaction` at `e7f71db`, built by `setup.sh` from a local
  clone (`JVECTOR_REPO=file://...`).
* **Data:** JVector's public ada-002 files. The prepared artifacts came from the same JVector settings on the same
  machine: `cpu-graph-1M.bin`, built by `GraphIndexBuilder` in 267.4 s, and 4 segments written with
  `OnDiskGraphIndex.write`.

| log | run | race 100k | GPU 1M build | compaction | PQ | verdict | wall |
|---|---|---|---|---|---|---|---|
| `run-tornado.log` | tornado, all acts | 15.8 → 1.9 s (8.2×) | 9.5 s | 14.8 s | 13.5 → 3.6 s | PASSED | 84.4 s |
| `run-java-argfile.log` | java @argfile, all acts | 15.8 → 2.0 s (7.9×) | 9.6 s | 15.5 s | 13.8 → 3.7 s | PASSED | 85.8 s |
| `run-tornado-race-only.log` | tornado, act 1 | 15.8 → 2.0 s (8.0×) | | | | PASSED | 19.3 s |

* **Reference bars, not run live:** acts 2 and 4 compare against JVector's CPU times measured on this machine,
  267.4 s for the 1M build and 294.6 s for the compaction. They are constants in `JVectorShowcase.java`, with
  their source logs named there.
* **Search act:** at the same beam the two graphs are within a few percent. The GPU graph has higher recall and
  equal or lower p50 at every beam.
* **PQ act:** GPU encoding with JVector's codebooks reproduces JVector's codes except for 15–22 of 982,790 vectors.
  Those have two equally close centroids (relative distance difference ≤ 1.5e-7); the check verifies that.

`setup.log` is a setup run with warm Maven and cuVS caches.
