import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Runs a Mandelbrot kernel written by an LLM (inserted below by fancyJitllm.sh) on the GPU with TornadoVM, runs the
 * same Java method on the CPU as the reference, compares them, and prints a downsampled image for the terminal.
 * Output lines: GPU_MS, CPU_MS, MATCH, then ROW lines of iteration counts (fancy.py draws them).
 */
public class Harness {

    // ---- the generated kernel ----
    /*KERNEL*/
    // ------------------------------

    public static void main(String[] args) throws Exception {
        int width = 4096, height = 3072, maxIterations = 256;
        IntArray gpu = new IntArray(width * height);
        TaskGraph tg = new TaskGraph("fractal")
                .task("mandelbrot", Harness::mandelbrot, width, height, maxIterations, gpu)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, gpu);
        long best = Long.MAX_VALUE;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            for (int r = 0; r < 5; r++) { // the first run includes the JIT compilation to CUDA
                long t = System.nanoTime();
                plan.execute();
                best = Math.min(best, System.nanoTime() - t);
            }
        }
        IntArray cpu = new IntArray(width * height);
        long t = System.nanoTime();
        mandelbrot(width, height, maxIterations, cpu); // the very same method, as plain Java on one CPU thread
        long cpuNanos = System.nanoTime() - t;

        long same = 0;
        for (int i = 0; i < width * height; i++) {
            same += gpu.get(i) == cpu.get(i) ? 1 : 0;
        }
        System.out.printf("GPU_MS %.2f%nCPU_MS %.1f%nMATCH %d %d%n", best / 1e6, cpuNanos / 1e6, same, (long) width * height);
        int cols = 110, rows = 38;
        for (int r = 0; r < rows; r++) {
            StringBuilder row = new StringBuilder("ROW");
            for (int c = 0; c < cols; c++) {
                row.append(' ').append(gpu.get((r * height / rows) * width + c * width / cols));
            }
            System.out.println(row);
        }
    }
}
