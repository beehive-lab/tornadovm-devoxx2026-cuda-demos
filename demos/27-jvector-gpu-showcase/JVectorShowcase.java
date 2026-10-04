import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.DoubleSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.gpu.GpuCompaction;
import io.github.jbellis.jvector.gpu.GpuProductQuantization;
import io.github.jbellis.jvector.gpu.TornadoGraphBuildAccelerator;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Demo 27: the JVector GPU work as a stage show, in five acts on real OpenAI ada-002 embeddings.
 *
 *   1 race     100k vectors: JVector's CPU build and the GPU build, both live
 *   2 scale    1M vectors: the GPU build live, against JVector's measured CPU time
 *   3 search   the same queries on the GPU-built and the CPU-built 1M graphs: recall and latency
 *   4 compact  merging 4 on-disk JVector segments: GPU rebuild live, against JVector's measured compactor
 *   5 pq       product quantization of 1M vectors: CPU and GPU, both live, with identical codes
 *
 *   JVectorShowcase <dataDir> [acts...]       (default: all acts; e.g. "race scale")
 *   JVectorShowcase <dataDir> prepare         once: JVector's CPU artifacts for acts 3 and 4 (about 10 minutes)
 *
 * dataDir holds JVector's public ada-002 files and two prepared artifacts (see README.md and prepare.sh).
 * Every act ends with a check; the run prints PASSED only if all of them hold.
 */
@SuppressWarnings("deprecation")
public class JVectorShowcase {

    static final VectorSimilarityFunction SIM = VectorSimilarityFunction.COSINE;
    static final int M = 32, BEAM = 100;
    static final float OVERFLOW = 1.2f, ALPHA = 1.2f;
    static final int QUERIES = 1000;

    /**
     * JVector's CPU times at 1M, measured on the stage machine (RTX 4090 host, i9-13900K, 32 threads) and shown as
     * reference bars: running them live would take 4.5 and 5 minutes. Source: jvector-tornadovm-extensions,
     * results/reeval-1M/build-cpu.log and results/improve4/compaction.log.
     */
    static final double CPU_BUILD_1M_SECONDS = 267.4, CPU_COMPACTION_1M_SECONDS = 294.6;

    static final List<String> ACTS = List.of("race", "scale", "search", "compact", "pq");
    static final List<String> verdicts = new ArrayList<>();
    static final Map<String, String> scoreboard = new LinkedHashMap<>();

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("usage: JVectorShowcase <dataDir> [" + String.join("|", ACTS) + "]...");
            return;
        }
        Path data = Path.of(args[0]);
        if (args.length > 1 && args[1].equals("prepare")) {
            prepare(data);
            return;
        }
        List<String> acts = args.length > 1 ? Arrays.asList(args).subList(1, args.length) : ACTS;
        Tui.banner();
        if (!new TornadoGraphBuildAccelerator().supports(new GraphIndexBuilder(
                new ListRandomAccessVectorValues(List.of(vector(1, 0), vector(0, 1)), 2), SIM, M, BEAM, OVERFLOW, ALPHA, true),
                new ListRandomAccessVectorValues(List.of(vector(1, 0), vector(0, 1)), 2), SIM)) {
            System.out.println(Tui.red("  GPU build unavailable: run under the tornado launcher, with cuVS on LD_LIBRARY_PATH"));
            System.out.println("JVectorShowcase: FAILED -- GPU build unavailable");
            return;
        }
        Dataset oneMillion = null;
        for (String act : acts) {
            switch (act) {
                case "race" -> race(data);
                case "scale", "search", "compact", "pq" -> {
                    if (oneMillion == null) {
                        oneMillion = Dataset.load(data, "ada_002_1M_base_982790.fvecs", "ada_002_1M_query_10000.fvecs", "ada_002_1M_gt_ip_100.ivecs");
                    }
                    switch (act) {
                        case "scale" -> {
                            scale(oneMillion);
                            if (acts.contains("search")) {
                                cpuGraphLoading = loadCpuGraph(oneMillion, data); // during the pause, while the presenter talks
                            }
                        }
                        case "search" -> search(oneMillion, data);
                        case "compact" -> compact(oneMillion, data);
                        default -> pq(oneMillion);
                    }
                }
                default -> throw new IllegalArgumentException("unknown act " + act + "; acts: " + ACTS);
            }
            Tui.pause();
        }
        finale();
    }

    // ---------------------------------------------------------------------------------------------------------
    // Act 1: the race

    static GraphHolder scaleGraph; // the 1M GPU graph, built in act 2 and searched in act 3

    static void race(Path data) throws IOException {
        Tui.act(1, "The race", "100k OpenAI ada-002 embeddings x 1536 dims, built by JVector on 32 CPU threads and on the GPU");
        var ds = Dataset.load(data, "ada_002_100k_base_99287.fvecs", "ada_002_100k_query_10000.fvecs", "ada_002_100k_gt_ip_100.ivecs");

        var cpuBuilder = builder(ds.vectors());
        double cpu = Tui.live("CPU  JVector", Tui.YELLOW, () -> cpuBuilder.getGraph().size(0) / (double) ds.size(),
                () -> cpuBuilder.build(ds.vectors()));
        var gpu = gpuBuild(ds, "GPU  TornadoVM + cuVS");
        double cpuRecall = recallAt(cpuBuilder.getGraph(), ds, 50);
        double gpuRecall = recallAt(gpu.graph(), ds, 50);

        Tui.bars("graph build", new String[] { "JVector CPU", "GPU" }, new double[] { cpu, gpu.seconds() }, "s", new String[] { Tui.YELLOW, Tui.GREEN });
        Tui.speedup(cpu / gpu.seconds());
        System.out.printf("    recall@10   JVector CPU %s   GPU %s%n", Tui.bold(fmt(cpuRecall)), Tui.bold(fmt(gpuRecall)));
        check("race: GPU recall@10 >= CPU - 0.01", gpuRecall >= cpuRecall - 0.01);
        scoreboard.put("Build 100k x 1536", String.format(Locale.ROOT, "%6.1f s -> %5.1f s  %5.1fx", cpu, gpu.seconds(), cpu / gpu.seconds()));
    }

    // ---------------------------------------------------------------------------------------------------------
    // Act 2: scale

    static void scale(Dataset ds) {
        Tui.act(2, "Scale", String.format(Locale.ROOT, "%,d ada-002 embeddings on the GPU, live; JVector's CPU build measured on this machine", ds.size()));
        var gpu = gpuBuild(ds, "GPU  TornadoVM + cuVS");
        scaleGraph = gpu;
        Tui.phases(gpu.phases());
        Tui.bars("graph build, 1M", new String[] { "JVector CPU (measured)", "GPU (live)" },
                new double[] { CPU_BUILD_1M_SECONDS, gpu.seconds() }, "s", new String[] { Tui.DIM_YELLOW, Tui.GREEN });
        Tui.speedup(CPU_BUILD_1M_SECONDS / gpu.seconds());
        System.out.println(Tui.dim("    candidates = NVIDIA cuVS NN-Descent | prune = Java tensor-core kernel, JIT-compiled by TornadoVM"));
        scoreboard.put("Build 1M x 1536", String.format(Locale.ROOT, "%6.1f s -> %5.1f s  %5.1fx", CPU_BUILD_1M_SECONDS, gpu.seconds(),
                CPU_BUILD_1M_SECONDS / gpu.seconds()));
    }

    // ---------------------------------------------------------------------------------------------------------
    // Act 3: search quality

    static void search(Dataset ds, Path data) throws IOException {
        Tui.act(3, "Search", "the same JVector search on the GPU-built graph and on JVector's own CPU-built graph (1M)");
        if (scaleGraph == null) {
            scaleGraph = gpuBuild(ds, "GPU  building the 1M graph first");
        }
        if (cpuGraphLoading == null) {
            cpuGraphLoading = loadCpuGraph(ds, data);
        }
        ImmutableGraphIndex cpuGraph = cpuGraphLoading.join();
        System.out.println(Tui.dim("    JVector's CPU-built 1M graph: cpu-graph-1M.bin, built by JVector's GraphIndexBuilder with the same settings"));
        System.out.printf("%n    %-6s %s %s%n", "beam", Tui.yellow(String.format("%-24s", "JVector CPU graph")), Tui.green(String.format("%-24s", "GPU graph")));
        System.out.printf("    %-6s %-24s %-24s%n", "", "recall@10   p50 latency", "recall@10   p50 latency");
        boolean better = true;
        for (int beam : new int[] { 20, 40, 80 }) {
            var cpu = timedSearch(cpuGraph, ds, beam);
            var gpu = timedSearch(scaleGraph.graph(), ds, beam);
            better &= gpu.recall() >= cpu.recall() - 0.005;
            System.out.printf("    %-6d %s   %6.0f us        %s   %6.0f us   %s%n", beam, fmt(cpu.recall()), cpu.p50Micros(),
                    Tui.bold(fmt(gpu.recall())), gpu.p50Micros(), gpu.recall() >= cpu.recall() && gpu.p50Micros() <= cpu.p50Micros() * 1.03
                            ? Tui.green("GPU graph wins") : "");
        }
        System.out.println(Tui.dim("    measured at equal recall over 10,000 queries: p50 -6% to -19%, p99 -4% to -18% (FINDINGS.md)"));
        check("search: GPU graph recall@10 >= CPU graph's at every beam", better);
        scoreboard.put("Search at equal recall", "p50 up to 19% lower, GPU graph");
    }

    static java.util.concurrent.CompletableFuture<ImmutableGraphIndex> cpuGraphLoading;

    /** JVector's CPU-built 1M graph, saved by prepare, loaded in the background. */
    static java.util.concurrent.CompletableFuture<ImmutableGraphIndex> loadCpuGraph(Dataset ds, Path data) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try (var reader = new SimpleReader(data.resolve("cpu-graph-1M.bin"))) {
                var bsp = BuildScoreProvider.randomAccessScoreProvider(ds.vectors(), SIM);
                var loaded = OnHeapGraphIndex.load(reader, ds.dim(), OVERFLOW, new VamanaDiversityProvider(bsp, ALPHA));
                loaded.setAllMutationsCompleted(); // a finished graph: searched through the same (frozen) view as the GPU one
                return loaded;
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        });
    }

    // ---------------------------------------------------------------------------------------------------------
    // Act 4: compaction

    static void compact(Dataset ds, Path data) throws IOException {
        Tui.act(4, "Compaction", "merge 4 on-disk JVector segments (built by JVector on the CPU) into one 1M index");
        List<ReaderSupplier> suppliers = new ArrayList<>();
        List<OnDiskGraphIndex> sources = new ArrayList<>();
        List<FixedBitSet> live = new ArrayList<>();
        List<OrdinalMapper> mappers = new ArrayList<>();
        Path out = Files.createTempFile("showcase-compacted", ".idx");
        try {
            int offset = 0;
            for (int p = 0; p < 4; p++) {
                var rs = ReaderSupplierFactory.open(data.resolve("segments").resolve("segment-" + p + "-of-4.idx"));
                suppliers.add(rs);
                var graph = OnDiskGraphIndex.load(rs);
                sources.add(graph);
                int size = graph.size(0);
                var bits = new FixedBitSet(size);
                bits.set(0, size);
                live.add(bits);
                mappers.add(new OrdinalMapper.OffsetMapper(offset, size));
                offset += size;
            }
            double gpu = Tui.live("GPU  GpuCompaction", Tui.GREEN, null,
                    () -> new GpuCompaction(sources, live, mappers, SIM).compact(out));
            double recall;
            try (var rs = ReaderSupplierFactory.open(out); var merged = OnDiskGraphIndex.load(rs)) {
                recall = recallAt(merged, ds, 50);
            }
            Tui.bars("compaction, 1M", new String[] { "JVector compactor (measured)", "GPU rebuild (live)" },
                    new double[] { CPU_COMPACTION_1M_SECONDS, gpu }, "s", new String[] { Tui.DIM_YELLOW, Tui.GREEN });
            Tui.speedup(CPU_COMPACTION_1M_SECONDS / gpu);
            System.out.printf("    merged index recall@10 %s: a freshly built graph, not a stitched one%n", Tui.bold(fmt(recall)));
            System.out.println(Tui.dim("    measured against the compactor's output over 10,000 queries: p50 -6% to -15% at equal recall (FINDINGS.md)"));
            check("compact: merged index recall@10 >= 0.95", recall >= 0.95);
            scoreboard.put("Compaction 4 -> 1 (1M)", String.format(Locale.ROOT, "%6.1f s -> %5.1f s  %5.1fx", CPU_COMPACTION_1M_SECONDS, gpu,
                    CPU_COMPACTION_1M_SECONDS / gpu));
        } finally {
            for (var g : sources) {
                g.close();
            }
            for (var rs : suppliers) {
                rs.close();
            }
            Files.deleteIfExists(out);
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Act 5: product quantization

    static void pq(Dataset ds) {
        int subspaces = ds.dim() / 8;
        Tui.act(5, "Product quantization", String.format(Locale.ROOT, "compress %,d vectors to %d bytes each (M=%d, 256 centroids), CPU and GPU live",
                ds.size(), subspaces, subspaces));
        var cpu = new Object() { ProductQuantization pq; io.github.jbellis.jvector.quantization.PQVectors codes; };
        double cpuSeconds = Tui.live("CPU  JVector", Tui.YELLOW, null, () -> {
            cpu.pq = ProductQuantization.compute(ds.vectors(), subspaces, 256, false);
            cpu.codes = (io.github.jbellis.jvector.quantization.PQVectors) cpu.pq.encodeAll(ds.vectors(), PhysicalCoreExecutor.pool());
        });
        double gpuSeconds = Tui.live("GPU  TornadoVM", Tui.GREEN, null, () -> {
            var pq = GpuProductQuantization.compute(ds.vectors(), subspaces, 256, false);
            GpuProductQuantization.encodeAll(pq, ds.vectors());
        });
        var sameBooks = GpuProductQuantization.encodeAll(cpu.pq, ds.vectors());
        long identical = IntStream.range(0, ds.size()).parallel().filter(i -> cpu.codes.get(i).equals(sameBooks.get(i))).count();
        long ties = IntStream.range(0, ds.size()).parallel().filter(i -> !cpu.codes.get(i).equals(sameBooks.get(i)))
                .filter(i -> equallyClose(cpu.pq, ds.vectors().getVector(i), cpu.codes.get(i), sameBooks.get(i))).count();
        Tui.bars("train + encode", new String[] { "JVector CPU", "GPU" }, new double[] { cpuSeconds, gpuSeconds }, "s",
                new String[] { Tui.YELLOW, Tui.GREEN });
        Tui.speedup(cpuSeconds / gpuSeconds);
        System.out.printf("    GPU encoding with JVector's codebooks: %s of %,d vectors encoded identically;%n", Tui.bold(String.format(Locale.ROOT, "%,d", identical)), ds.size());
        System.out.printf("    the other %,d pick an equally close centroid (a float rounding tie)%n", ds.size() - identical);
        check("pq: JVector's codes, up to exact ties", identical + ties == ds.size());
        scoreboard.put("PQ train + encode (1M)", String.format(Locale.ROOT, "%6.1f s -> %5.1f s  %5.1fx", cpuSeconds, gpuSeconds, cpuSeconds / gpuSeconds));
    }

    /**
     * Whether two encodings of {@code v} are equally good: in every subspace where they differ, the two centroids are at
     * the same distance from {@code v} up to float rounding.
     */
    static boolean equallyClose(ProductQuantization pq, VectorFloat<?> v, io.github.jbellis.jvector.vector.types.ByteSequence<?> a,
                                io.github.jbellis.jvector.vector.types.ByteSequence<?> b) {
        int dim = v.length();
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?> da = vts.createFloatVector(dim), db = vts.createFloatVector(dim);
        pq.decode(a, da);
        pq.decode(b, db);
        int sub = dim / a.length();
        for (int m = 0; m < a.length(); m++) {
            if (a.get(m) == b.get(m)) {
                continue;
            }
            double distA = 0, distB = 0;
            for (int d = m * sub; d < (m + 1) * sub; d++) {
                distA += Math.pow(v.get(d) - da.get(d), 2);
                distB += Math.pow(v.get(d) - db.get(d), 2);
            }
            if (Math.abs(distA - distB) > 1e-6 * Math.max(distA, distB)) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------------------------------------------
    // Preparation (once, before the talk): what JVector itself builds on the CPU, saved for acts 3 and 4

    static void prepare(Path data) throws IOException {
        var ds = Dataset.load(data, "ada_002_1M_base_982790.fvecs", "ada_002_1M_query_10000.fvecs", "ada_002_1M_gt_ip_100.ivecs");
        Path graphPath = data.resolve("cpu-graph-1M.bin");
        if (!Files.exists(graphPath)) {
            var cpu = builder(ds.vectors());
            Tui.live("CPU  JVector 1M graph", Tui.YELLOW, () -> cpu.getGraph().size(0) / (double) ds.size(), () -> cpu.build(ds.vectors()));
            try (var out = new java.io.DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(graphPath), 1 << 24))) {
                ((OnHeapGraphIndex) cpu.getGraph()).save(out);
            }
        }
        Files.createDirectories(data.resolve("segments"));
        for (int p = 0; p < 4; p++) {
            Path segment = data.resolve("segments").resolve("segment-" + p + "-of-4.idx");
            if (Files.exists(segment)) {
                continue;
            }
            int from = (int) ((long) ds.size() * p / 4), to = (int) ((long) ds.size() * (p + 1) / 4);
            var rows = new ListRandomAccessVectorValues(IntStream.range(from, to).mapToObj(ds.vectors()::getVector).toList(), ds.dim());
            var cpu = builder(rows);
            Tui.live("CPU  JVector segment " + p, Tui.YELLOW, () -> cpu.getGraph().size(0) / (double) rows.size(), () -> {
                cpu.build(rows);
                OnDiskGraphIndex.write(cpu.getGraph(), rows, segment);
            });
        }
        System.out.println("JVectorShowcase: prepared " + data);
    }

    // ---------------------------------------------------------------------------------------------------------

    static void finale() {
        boolean ok = verdicts.stream().noneMatch(v -> v.startsWith("FAIL"));
        Tui.scoreboard(scoreboard);
        verdicts.forEach(v -> System.out.println("    " + (v.startsWith("PASS") ? Tui.green("✔ ") : Tui.red("✘ ")) + v.substring(5)));
        System.out.println();
        System.out.println(ok ? "JVectorShowcase: PASSED -- faster on the GPU, and every quality check holds"
                : "JVectorShowcase: FAILED -- a quality check did not hold (see above)");
    }

    static void check(String what, boolean ok) {
        verdicts.add((ok ? "PASS " : "FAIL ") + what);
    }

    record GraphHolder(ImmutableGraphIndex graph, double seconds, Map<String, Double> phases) {
    }

    /** The GPU build through JVector's accelerator SPI, with the phase times of its trace. */
    static GraphHolder gpuBuild(Dataset ds, String label) {
        var builder = builder(ds.vectors());
        var accelerator = new TornadoGraphBuildAccelerator();
        var trace = new ByteArrayOutputStream();
        PrintStream console = System.out;
        double seconds = Tui.live(label, Tui.GREEN, null, () -> {
            System.setOut(new PrintStream(trace, true)); // the accelerator's -Djvector.gpu.trace line
            try {
                accelerator.build(builder, ds.vectors(), SIM);
                builder.cleanup();
            } finally {
                System.setOut(console);
            }
        });
        return new GraphHolder(builder.getGraph(), seconds, phases(trace.toString()));
    }

    /** "rows 0.30 s, candidates 4.91 s, prune 3.29 s, ..." from the accelerator's trace. */
    static Map<String, Double> phases(String trace) {
        Map<String, Double> phases = new LinkedHashMap<>();
        int colon = trace.indexOf("): ");
        if (colon < 0) {
            return phases;
        }
        Matcher m = Pattern.compile("([a-z ]+) ([0-9.]+) s").matcher(trace.substring(colon + 3));
        while (m.find() && !m.group(1).trim().equals("total")) {
            phases.put(m.group(1).trim(), Double.parseDouble(m.group(2)));
        }
        return phases;
    }

    static GraphIndexBuilder builder(RandomAccessVectorValues vectors) {
        return new GraphIndexBuilder(vectors, SIM, M, BEAM, OVERFLOW, ALPHA, true);
    }

    static double recallAt(ImmutableGraphIndex graph, Dataset ds, int beam) {
        return IntStream.range(0, QUERIES).parallel().mapToDouble(q -> hits(GraphSearcher.search(ds.queries().get(q), 10, beam, ds.vectors(), SIM, graph,
                Bits.ALL).getNodes(), ds.truth().get(q))).sum() / (10.0 * QUERIES);
    }

    record Timed(double recall, double p50Micros) {
    }

    /** Single-threaded search of {@value #QUERIES} queries after a warm-up: recall@10 and the median latency. */
    static Timed timedSearch(ImmutableGraphIndex graph, Dataset ds, int beam) {
        var searcher = new GraphSearcher(graph);
        long[] nanos = new long[QUERIES];
        double hits = 0;
        for (int pass = 0; pass < 2; pass++) {
            hits = 0;
            for (int q = 0; q < QUERIES; q++) {
                var ssp = DefaultSearchScoreProvider.exact(ds.queries().get(q), SIM, ds.vectors());
                long t = System.nanoTime();
                var result = searcher.search(ssp, 10, beam, 0f, 0f, Bits.ALL);
                nanos[q] = System.nanoTime() - t;
                hits += hits(result.getNodes(), ds.truth().get(q));
            }
        }
        Arrays.sort(nanos);
        return new Timed(hits / (10.0 * QUERIES), nanos[QUERIES / 2] / 1e3);
    }

    static double hits(io.github.jbellis.jvector.graph.SearchResult.NodeScore[] nodes, int[] truth) {
        var expected = new HashSet<Integer>();
        for (int k = 0; k < 10; k++) {
            expected.add(truth[k]);
        }
        return Arrays.stream(nodes).filter(ns -> expected.contains(ns.node)).count();
    }

    static String fmt(double recall) {
        return String.format(Locale.ROOT, "%.4f", recall);
    }

    static VectorFloat<?> vector(float... values) {
        return VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(values);
    }

    // ---------------------------------------------------------------------------------------------------------
    // Data: JVector's public fvecs/ivecs files, read through a memory mapping in parallel

    record Dataset(RandomAccessVectorValues vectors, List<VectorFloat<?>> queries, List<int[]> truth) {
        int size() {
            return vectors.size();
        }

        int dim() {
            return vectors.dimension();
        }

        static Dataset load(Path dir, String base, String queries, String truth) throws IOException {
            long t = System.nanoTime();
            List<VectorFloat<?>> rows = fvecs(dir.resolve(base), Integer.MAX_VALUE);
            var ds = new Dataset(new ListRandomAccessVectorValues(rows, rows.get(0).length()), fvecs(dir.resolve(queries), QUERIES),
                    ivecs(dir.resolve(truth), QUERIES));
            System.out.println(Tui.dim(String.format(Locale.ROOT, "    loaded %,d x %d vectors (%s) in %.1f s", ds.size(), ds.dim(), base,
                    (System.nanoTime() - t) / 1e9)));
            return ds;
        }

        static List<VectorFloat<?>> fvecs(Path path, int limit) throws IOException {
            try (var channel = FileChannel.open(path, StandardOpenOption.READ); var arena = Arena.ofShared()) {
                MemorySegment file = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
                int dim = file.get(ValueLayout.JAVA_INT_UNALIGNED, 0);
                long rowBytes = 4L + 4L * dim;
                int n = (int) Math.min(limit, file.byteSize() / rowBytes);
                VectorFloat<?>[] rows = new VectorFloat<?>[n];
                IntStream.range(0, n).parallel().forEach(i -> {
                    float[] v = new float[dim];
                    MemorySegment.copy(file, ValueLayout.JAVA_FLOAT_UNALIGNED, i * rowBytes + 4, v, 0, dim);
                    rows[i] = vector(v);
                });
                return new ArrayList<>(Arrays.asList(rows));
            }
        }

        static List<int[]> ivecs(Path path, int limit) throws IOException {
            try (var channel = FileChannel.open(path, StandardOpenOption.READ); var arena = Arena.ofShared()) {
                MemorySegment file = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
                int k = file.get(ValueLayout.JAVA_INT_UNALIGNED, 0);
                long rowBytes = 4L + 4L * k;
                int n = (int) Math.min(limit, file.byteSize() / rowBytes);
                List<int[]> rows = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    int[] row = new int[k];
                    MemorySegment.copy(file, ValueLayout.JAVA_INT_UNALIGNED, i * rowBytes + 4, row, 0, k);
                    rows.add(row);
                }
                return rows;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // Terminal output: colors, live progress lines, bar charts. NO_COLOR=1 turns colors off; NO_PAUSE=1 skips pauses.

    static final class Tui {
        static final boolean COLOR = System.getenv("NO_COLOR") == null;
        static final String RESET = "\033[0m", BOLD = "\033[1m", DIM = "\033[2m", GREEN = "\033[1;32m", YELLOW = "\033[1;33m",
                DIM_YELLOW = "\033[0;33m", CYAN = "\033[1;36m", MAGENTA = "\033[1;35m", BLUE = "\033[1;34m", RED = "\033[1;31m";
        static final int WIDTH = 46;

        interface Work {
            void run() throws Exception;
        }

        static String paint(String color, String s) {
            return COLOR ? color + s + RESET : s;
        }

        static String bold(String s) {
            return paint(BOLD, s);
        }

        static String dim(String s) {
            return paint(DIM, s);
        }

        static String green(String s) {
            return paint(GREEN, s);
        }

        static String yellow(String s) {
            return paint(YELLOW, s);
        }

        static String red(String s) {
            return paint(RED, s);
        }

        static void banner() {
            System.out.println();
            System.out.println(paint(CYAN, "  ╔══════════════════════════════════════════════════════════════════════╗"));
            System.out.println(paint(CYAN, "  ║") + bold("   JVector on the GPU  ·  pure Java  ·  TornadoVM + NVIDIA cuVS          ") + paint(CYAN, "║"));
            System.out.println(paint(CYAN, "  ║") + dim("   the vector index inside Apache Cassandra, built on an RTX 4090       ") + paint(CYAN, "║"));
            System.out.println(paint(CYAN, "  ╚══════════════════════════════════════════════════════════════════════╝"));
        }

        static void act(int number, String title, String subtitle) {
            System.out.println();
            System.out.println(paint(MAGENTA, "  ━━ Act " + number + " · " + title + " ") + paint(MAGENTA, "━".repeat(Math.max(3, 62 - title.length()))));
            System.out.println("    " + subtitle);
            System.out.println();
        }

        /**
         * Runs {@code work} while one status line animates: a progress bar when {@code progress} is given, else a
         * spinner. Returns the wall-clock seconds.
         */
        static double live(String label, String color, DoubleSupplier progress, Work work) {
            PrintStream out = System.out;
            AtomicBoolean done = new AtomicBoolean();
            long start = System.nanoTime();
            Thread ticker = new Thread(() -> {
                String spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
                for (int tick = 0; !done.get(); tick++) {
                    double seconds = (System.nanoTime() - start) / 1e9;
                    String bar = progress != null ? bar(Math.min(1, progress.getAsDouble()), WIDTH, color)
                            : paint(color, String.valueOf(spin.charAt(tick % spin.length()))) + " " + dim("working");
                    out.printf("\r    %-24s %s %6.1f s ", label, bar, seconds);
                    out.flush();
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        return;
                    }
                }
            });
            ticker.setDaemon(true);
            ticker.start();
            try {
                work.run();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            } finally {
                done.set(true);
                try {
                    ticker.join();
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            out.printf("\r    %-24s %s %6.1f s %s%n", label, bar(1, WIDTH, color), seconds, green("done"));
            return seconds;
        }

        static String bar(double fraction, int width, String color) {
            int full = (int) Math.round(fraction * width);
            return paint(color, "█".repeat(full)) + dim("░".repeat(width - full));
        }

        /** Horizontal bars on one scale, longest = full width. */
        static void bars(String title, String[] labels, double[] values, String unit, String[] colors) {
            double max = Arrays.stream(values).max().orElse(1);
            System.out.println();
            System.out.println("    " + bold(title));
            for (int i = 0; i < labels.length; i++) {
                int width = Math.max(1, (int) Math.round(values[i] / max * WIDTH));
                System.out.printf("    %-30s %s %7.1f %s%n", labels[i], paint(colors[i], "█".repeat(width)) + " ".repeat(WIDTH - width), values[i], unit);
            }
        }

        static void speedup(double x) {
            System.out.printf("%n    %s%n%n", paint(GREEN, String.format(Locale.ROOT, "▶ %.1fx faster", x)));
        }

        /** The GPU build's phases as one stacked bar, with a legend. */
        static void phases(Map<String, Double> phases) {
            if (phases.isEmpty()) {
                return;
            }
            String[] colors = { BLUE, CYAN, MAGENTA, YELLOW, GREEN };
            double total = phases.values().stream().mapToDouble(Double::doubleValue).sum();
            StringBuilder bar = new StringBuilder();
            StringBuilder legend = new StringBuilder();
            int i = 0;
            for (var e : phases.entrySet()) {
                String color = colors[i++ % colors.length];
                bar.append(paint(color, "█".repeat(Math.max(1, (int) Math.round(e.getValue() / total * 60)))));
                legend.append(paint(color, "■ ")).append(String.format(Locale.ROOT, "%s %.1f s   ", e.getKey(), e.getValue()));
            }
            System.out.println();
            System.out.println("    " + bold("where the GPU build's time goes"));
            System.out.println("    " + bar);
            System.out.println("    " + legend);
        }

        static void scoreboard(Map<String, String> rows) {
            System.out.println();
            System.out.println(paint(CYAN, "  ╔══════════════════════════════════════════════════════════════════════╗"));
            System.out.println(paint(CYAN, "  ║") + bold(String.format("  %-28s %-40s", "Scoreboard", "JVector CPU -> GPU")) + paint(CYAN, "║"));
            System.out.println(paint(CYAN, "  ╟──────────────────────────────────────────────────────────────────────╢"));
            rows.forEach((k, v) -> System.out.println(paint(CYAN, "  ║") + String.format("  %-28s %-40s", k, v) + paint(CYAN, "║")));
            System.out.println(paint(CYAN, "  ╚══════════════════════════════════════════════════════════════════════╝"));
            System.out.println();
        }

        static void pause() {
            if (System.getenv("NO_PAUSE") == null && System.console() != null) {
                System.out.print(dim("    [enter] next act "));
                System.console().readLine();
            }
        }
    }
}
