# Batch 49: demo 28's live dashboard (`run.sh dashboard`)

Captured 2026-10-05 on the sm_89 host (RTX 4090, driver 610.57.04).
* **The kernel** runs on the TornadoVM 7.0.0 release (SDKMAN `7.0.0-jdk22plus-cuda`), JDK 25.0.2.
* **Generation** uses jitLLM `main` `d875f083` (its own TornadoVM 7.0.2-dev, JDK 21), Qwen3-4B F16.

| item | value |
|---|---|
| jitLLM | `achieved tok/s: 66.90. Tokens: 644, seconds: 9.63` (prompt + answer) |
| live kernel | byte-identical to `dashboard/reference.kernel` (`generation.out`, checked with `diff`) |
| javac | clean (`javac.log` empty); `Harness.java` is the kernel inside the template |
| check (GPU vs the same method on the CPU) | 785,373 of 786,432 pixels identical (99.87%) |
| zoom | 100 frames at 7680 × 4320, median 16.72 ms per GPU execution (incl. the copy back) |
| GPU processes seen by the dashboard (nvidia-smi) | jitLLM engine 8,632 MiB, then the kernel 516 MiB, same device |
| verdict | `LiveDashboard: PASSED`, 35.5 s wall with `NO_PAUSE=1` |

Files:
* `dashboard-screen.log.gz`: the terminal output, every dashboard frame (ANSI).
* `run.log.no-rows`: the TornadoVM output with the generated CUDA, minus the image rows.
* `generation.out` and `generation.err`.
* `Harness.java` and `javac.log`.

The screenshots in `demos/28-llm-writes-gpu-kernel/dashboard/screenshots/` are frames of this run, rendered from
the ANSI with a small PIL script.
