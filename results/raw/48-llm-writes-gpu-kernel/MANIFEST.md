# Batch 48: demo 28, an LLM in Java writes a GPU kernel in Java

Captured 2026-10-04 on the sm_89 host (`environment.txt`): RTX 4090, driver 610.57.04.
* **The kernel** runs on the TornadoVM 7.0.0 release, JDK 25.0.2.
* **Generation** uses jitLLM `main` `d875f083`, with its own TornadoVM 7.0.2-dev build and JDK 21, and Qwen3-4B F16.

| log | run | generation | GPU | CPU (same method, 1 thread) | pixels identical | verdict | wall |
|---|---|---|---|---|---|---|---|
| `run-tornado-live.log` | tornado, model writes the kernel live | 67.04 tok/s, 559 tokens (prompt + answer) in 8.34 s | 3.1 ms | 1,182.2 ms | 99.79% | PASSED | 15.4 s |
| `run-java-argfile-reference.log` | java @argfile, `--reference` | none | 3.1 ms | 1,166.1 ms | 99.79% | PASSED | 2.3 s |
| `run-tornado-reference.log` | tornado, `--reference` | none | 3.1 ms | 1,167.2 ms | 99.79% | PASSED | 2.4 s |

* **The live kernel equals `reference.kernel`.** `generation.out` (the model's raw output) and `generation.err`
  (jitLLM's metrics) are from the live run. The kernel extracted from `generation.out` is byte-identical to
  `reference.kernel` (checked with `diff`), as it was in five earlier rehearsal runs: greedy decoding at
  temperature 0.
* **What the 0.21% non-identical pixels are:** float rounding at the set's edge. The GPU fuses multiply-adds, so
  near the escape threshold a pixel's iteration count can differ by one.
* **Rejected alternatives, from the same day's rehearsal:**
  * Llama-3.2-3B Q8_0's kernel did not compile.
  * Asking for a whole program failed with both models: one did not compile, the other gave wrong results.
