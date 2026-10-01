# Batch 46 — demo 26, JVector index built on the GPU

Captured 2026-10-01 on the sm_89 host (`environment.txt`): RTX 4090, driver 610.57.04, JDK 25.0.2.

* TornadoVM: source build of `mikepapadim:feat/cuvs-provider` (beehive-lab/TornadoVM#1155) at `17424e265`,
  SDK `7.0.2-jdk22plus-dev`. **Not** the repo's default `sdkman-7.0.0` profile, which has no `tornado-cuvs`.
* JVector: `mikepapadim/jvector:feat/graph-build-accelerator` at `7597583`, built by `setup.sh`
  (jvector-base, jvector-twenty, jvector-gpu).
* cuVS 26.08 from NVIDIA's wheels (`setup.sh` → `jvector-gpu/cuvs-env.sh`).

| log | run | GPU build | CPU build | recall@10 CPU / GPU | verdict | wall |
|---|---|---|---|---|---|---|
| `run-tornado.log` | tornado, 500k × 1024 synthetic | 6.52 s | 48.17 s | 0.9855 / 0.9830 | PASSED | 59.5 s |
| `run-java-argfile.log` | java @argfile, 500k × 1024 synthetic | 6.44 s | 47.56 s | 0.9645 / 0.9825 | PASSED | 59.0 s |
| `run-tornado-ada002-100k.log` | tornado, ada002 99,087 × 1536 | 2.56 s | 15.83 s | 0.9905 / 0.9940 | PASSED | 20.7 s |
| `run-tornado-100k-768.log` | tornado, 100k × 768 synthetic | 2.24 s | 8.43 s | 1.0000 / 1.0000 | PASSED | 12.6 s |

`setup.log`: setup with warm Maven and cuVS caches (4.2 s). A first setup downloads JVector's Maven dependencies
and 2.1 GB of cuVS wheels.
