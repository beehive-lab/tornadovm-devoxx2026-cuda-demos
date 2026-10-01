import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.gpu.TornadoGraphBuildAccelerator;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Demo 26: the same JVector vector index built twice, once on the GPU (TornadoVM + NVIDIA cuVS, through JVector's
 * GraphBuildAccelerator SPI) and once with JVector's own multi-threaded CPU builder, then searched with the same
 * code to compare recall.
 *
 * The GPU build: NVIDIA cuVS NN-Descent finds candidate neighbors (called through TornadoVM's cuVS module), a
 * TornadoVM-compiled Java kernel prunes them with JVector's diversity rule on tensor cores (ctx.mma), and the
 * result is an ordinary JVector graph.
 *
 *   JVectorGpuIndex [n=500000] [dim=1024]            synthetic clustered embeddings
 *   JVectorGpuIndex --fvecs base.fvecs [n]            the first n rows of a real dataset (cosine)
 */
@SuppressWarnings("deprecation")
public class JVectorGpuIndex {

    static final int QUERIES = 200;
    static final int M = 32, BEAM = 100;
    static final float OVERFLOW = 1.2f, ALPHA = 1.2f;

    public static void main(String[] args) throws IOException {
        var sim = VectorSimilarityFunction.COSINE;
        List<VectorFloat<?>> data;
        String source;
        if (args.length > 0 && args[0].equals("--fvecs")) {
            int n = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE - QUERIES;
            data = fvecs(Path.of(args[1]), n + QUERIES);
            source = args[1];
        } else {
            int n = args.length > 0 ? Integer.parseInt(args[0]) : 500_000;
            int dim = args.length > 1 ? Integer.parseInt(args[1]) : 1024;
            long t = System.nanoTime();
            data = clustered(n + QUERIES, dim, new Random(42));
            source = String.format("synthetic clustered embeddings (generated in %.1f s)", (System.nanoTime() - t) / 1e9);
        }
        List<VectorFloat<?>> base = data.subList(0, data.size() - QUERIES);
        List<VectorFloat<?>> queries = data.subList(data.size() - QUERIES, data.size());
        int dim = base.get(0).length();
        RandomAccessVectorValues vectors = new ListRandomAccessVectorValues(base, dim);
        System.out.printf("%n  JVector index of %,d x %d vectors, %s%n", base.size(), dim, source);
        System.out.printf("  graph: M=%d, beam %d, alpha %.1f, hierarchy; %s%n%n", M, BEAM, ALPHA,
                VectorizationProvider.getInstance().getClass().getSimpleName());

        // 1. GPU: TornadoVM + cuVS, plugged in through JVector's GraphBuildAccelerator SPI
        var accelerator = new TornadoGraphBuildAccelerator();
        var gpuBuilder = new GraphIndexBuilder(vectors, sim, M, BEAM, OVERFLOW, ALPHA, true);
        if (!accelerator.supports(gpuBuilder, vectors, sim)) {
            System.out.println("JVectorGpuIndex: FAILED -- GPU build unavailable (TornadoVM with tornado-cuvs, cuVS on LD_LIBRARY_PATH?)");
            return;
        }
        System.out.println("  [GPU] building with TornadoVM + cuVS ...");
        long t0 = System.nanoTime();
        accelerator.build(gpuBuilder, vectors, sim);
        gpuBuilder.cleanup();
        double gpuSeconds = (System.nanoTime() - t0) / 1e9;
        System.out.printf("  [GPU] built in %6.2f s%n", gpuSeconds);

        // 2. CPU: JVector's own builder, all cores
        System.out.printf("  [CPU] building with JVector on %d threads ...%n", Runtime.getRuntime().availableProcessors());
        var cpuBuilder = new GraphIndexBuilder(vectors, sim, M, BEAM, OVERFLOW, ALPHA, true);
        long t1 = System.nanoTime();
        cpuBuilder.build(vectors);
        double cpuSeconds = (System.nanoTime() - t1) / 1e9;
        System.out.printf("  [CPU] built in %6.2f s%n%n", cpuSeconds);

        // 3. the same search on both graphs, against exact neighbors
        List<int[]> truth = exactTop10(base, queries, sim);
        double gpuRecall = recall(gpuBuilder.getGraph(), vectors, queries, truth, sim);
        double cpuRecall = recall(cpuBuilder.getGraph(), vectors, queries, truth, sim);

        System.out.println("  +-------------------+------------+------------+");
        System.out.println("  |                   |  JVector   |   GPU      |");
        System.out.println("  |                   |  (CPU)     | (TornadoVM)|");
        System.out.println("  +-------------------+------------+------------+");
        System.out.printf("  | graph build       | %8.2f s | %8.2f s |%n", cpuSeconds, gpuSeconds);
        System.out.printf("  | recall@10         | %10.4f | %10.4f |%n", cpuRecall, gpuRecall);
        System.out.println("  +-------------------+------------+------------+");
        System.out.printf("  speed-up: %.1fx%n%n", cpuSeconds / gpuSeconds);
        boolean ok = gpuRecall >= cpuRecall - 0.01;
        System.out.println(ok ? "JVectorGpuIndex: PASSED -- the GPU graph's recall@10 matches or beats JVector's"
                : "JVectorGpuIndex: FAILED -- the GPU graph's recall@10 is below JVector's");
    }

    static double recall(ImmutableGraphIndex graph, RandomAccessVectorValues vectors, List<VectorFloat<?>> queries, List<int[]> truth,
                         VectorSimilarityFunction sim) {
        double hits = 0;
        for (int q = 0; q < queries.size(); q++) {
            var result = GraphSearcher.search(queries.get(q), 10, 50, vectors, sim, graph, Bits.ALL);
            var expected = new HashSet<Integer>();
            for (int id : truth.get(q)) {
                expected.add(id);
            }
            for (var ns : result.getNodes()) {
                if (expected.contains(ns.node)) {
                    hits++;
                }
            }
        }
        return hits / (10.0 * queries.size());
    }

    /** Exact top-10 of every query by brute force, in parallel. */
    static List<int[]> exactTop10(List<VectorFloat<?>> base, List<VectorFloat<?>> queries, VectorSimilarityFunction sim) {
        return IntStream.range(0, queries.size()).parallel().mapToObj(q -> {
            int[] ids = new int[10];
            float[] scores = new float[10];
            Arrays.fill(scores, -Float.MAX_VALUE);
            for (int i = 0; i < base.size(); i++) {
                float s = sim.compare(queries.get(q), base.get(i));
                if (s > scores[9]) {
                    int p = 9;
                    while (p > 0 && scores[p - 1] < s) {
                        scores[p] = scores[p - 1];
                        ids[p] = ids[p - 1];
                        p--;
                    }
                    scores[p] = s;
                    ids[p] = i;
                }
            }
            return ids;
        }).toList();
    }

    /** Unit vectors around 1000 random centers, like embeddings of 1000 topics. */
    static List<VectorFloat<?>> clustered(int n, int dim, Random random) {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        int clusters = 1000;
        float[][] centers = new float[clusters][dim];
        for (float[] c : centers) {
            for (int d = 0; d < dim; d++) {
                c[d] = (float) random.nextGaussian();
            }
        }
        long seed = random.nextLong();
        VectorFloat<?>[] rows = new VectorFloat<?>[n];
        IntStream.range(0, n).parallel().forEach(i -> {
            Random r = new Random(seed + i);
            float[] c = centers[r.nextInt(clusters)];
            float[] v = new float[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                v[d] = c[d] + 0.7f * (float) r.nextGaussian();
                norm += v[d] * v[d];
            }
            for (int d = 0; d < dim; d++) {
                v[d] /= (float) Math.sqrt(norm);
            }
            rows[i] = vts.createFloatVector(v);
        });
        return new ArrayList<>(Arrays.asList(rows));
    }

    static List<VectorFloat<?>> fvecs(Path path, int limit) throws IOException {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        List<VectorFloat<?>> rows = new ArrayList<>();
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            while (rows.size() < limit && in.available() > 0) {
                int dim = Integer.reverseBytes(in.readInt());
                float[] v = new float[dim];
                ByteBuffer.wrap(in.readNBytes(dim * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(v);
                rows.add(vts.createFloatVector(v));
            }
        }
        return rows;
    }
}
