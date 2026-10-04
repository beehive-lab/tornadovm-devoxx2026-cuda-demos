#!/usr/bin/env bash
# Demo 27: the JVector GPU showcase, five acts on real OpenAI ada-002 embeddings (about 2-3 minutes in all).
#
#   bash demos/27-jvector-gpu-showcase/run.sh [tornado|java] <dataDir> [race] [scale] [search] [compact] [pq]
#
# tornado (default) runs under the tornado launcher, java with `java @$TORNADOVM_HOME/tornado-argfile`.
# Needs: source scripts/setup-env.sh (a TornadoVM SDK with tornado-cuvs), then setup.sh and prepare.sh once.
# NO_PAUSE=1 runs the acts back to back (otherwise [enter] between acts); NO_COLOR=1 for plain output.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mode=tornado
case "${1:-}" in tornado|java) mode="$1"; shift ;; esac
[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 1; }
[ -n "${TORNADOVM_HOME:-}" ] || { echo "run.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
[ -d "$here/lib" ] || { echo "run.sh: run setup.sh first" >&2; exit 1; }
cuvs="${JVECTOR_GPU_CUVS:-$HOME/.jvector-gpu/cuvs}"
[ -s "$cuvs/ld_library_path" ] || { echo "run.sh: cuVS not installed in $cuvs (run setup.sh)" >&2; exit 1; }
export LD_LIBRARY_PATH="$(cat "$cuvs/ld_library_path")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

classes="$here/build/classes"
mkdir -p "$classes"
sdk=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')
"$JAVA_HOME/bin/javac" -proc:none -cp "$sdk$here/lib/*" -d "$classes" "$here/JVectorShowcase.java"

# device memory for 1M x 1536 in FP16 plus the pruner's workspace; the heap holds the 1M vectors twice at most
flags="-Xmx40g -Dtornado.device.memory=8GB --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Djvector.gpu.trace=true ${TORNADO_JVM_FLAGS:-}"
cp="$classes:$here/lib/*"
if [ "$mode" = java ]; then
    [ -f "$TORNADOVM_HOME/tornado-argfile" ] || tornado --generate-argfile >/dev/null
    exec "$JAVA_HOME/bin/java" "@$TORNADOVM_HOME/tornado-argfile" $flags -cp "$cp" JVectorShowcase "$@"
fi
# the launcher takes program arguments that start with -- as its own (TornadoVM #1150): pass them with --params
exec tornado --jvm="$flags" --classpath "$cp" JVectorShowcase --params "$*"
