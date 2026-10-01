#!/usr/bin/env bash
# Demo 26: build the same JVector index on the GPU and on the CPU (about 1-2 minutes at the defaults).
#
#   bash demos/26-jvector-gpu-index/run.sh [tornado|java|both] [n] [dim]
#   bash demos/26-jvector-gpu-index/run.sh tornado --fvecs base.fvecs [n]
#
# Needs: source scripts/setup-env.sh (a TornadoVM SDK with tornado-cuvs), then setup.sh once.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
mode="${1:-tornado}"; [ $# -gt 0 ] && shift
[ -n "${TORNADOVM_HOME:-}" ] || { echo "run.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
[ -d "$here/lib" ] || { echo "run.sh: run setup.sh first" >&2; exit 1; }
cuvs="${JVECTOR_GPU_CUVS:-$HOME/.jvector-gpu/cuvs}"
[ -s "$cuvs/ld_library_path" ] || { echo "run.sh: cuVS not installed in $cuvs (run setup.sh)" >&2; exit 1; }
export LD_LIBRARY_PATH="$(cat "$cuvs/ld_library_path")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

classes="$here/build/classes"
mkdir -p "$classes"
sdk=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')
"$JAVA_HOME/bin/javac" -proc:none -cp "$sdk$here/lib/*" -d "$classes" "$here/JVectorGpuIndex.java"

cp="$classes:$here/lib/*"
flags="-Xmx24g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Djvector.gpu.trace=true ${TORNADO_JVM_FLAGS:-}"
case "$mode" in
    tornado) # the launcher takes program arguments that start with -- as its own (TornadoVM #1150): pass them with --params
             if [ $# -gt 0 ]; then tornado --jvm="$flags" --classpath "$cp" JVectorGpuIndex --params "$*"
             else tornado --jvm="$flags" --classpath "$cp" JVectorGpuIndex; fi ;;
    java)    java "@$TORNADOVM_HOME/tornado-argfile" $flags -cp "$cp" JVectorGpuIndex "$@" ;;
    both)    "$0" tornado "$@" && "$0" java "$@" ;;
    *)       sed -n '2,7p' "$0"; exit 1 ;;
esac
