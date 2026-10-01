#!/usr/bin/env bash
# One-time setup for demo 26 (run before going on stage; needs the network, ~5-15 min):
#   1. builds JVector with the GPU module (jvector-gpu) into lib/;
#   2. installs NVIDIA cuVS (the C library the GPU build calls) into ~/.jvector-gpu/cuvs.
#
#   source scripts/setup-env.sh          # with a TornadoVM SDK that has tornado-cuvs
#   bash demos/26-jvector-gpu-index/setup.sh
#
# JVECTOR_REPO / JVECTOR_BRANCH select the JVector source (default: the PR branch).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="${JVECTOR_REPO:-https://github.com/mikepapadim/jvector.git}"
branch="${JVECTOR_BRANCH:-feat/graph-build-accelerator}"
src="$here/build/jvector"

[ -n "${TORNADOVM_HOME:-}" ] || { echo "setup.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
api=$(ls "$TORNADOVM_HOME"/share/java/tornado/tornado-api-*.jar | head -1)
cuvs=$(ls "$TORNADOVM_HOME"/share/java/tornado/tornado-cuvs-*.jar 2>/dev/null | head -1 || true)
[ -n "$cuvs" ] || { echo "setup.sh: this TornadoVM SDK has no tornado-cuvs (needs beehive-lab/TornadoVM#1155)" >&2; exit 1; }
version=$(basename "$api" .jar); version=${version#tornado-api-}
echo "== TornadoVM $version at $TORNADOVM_HOME"

# 1. JVector sources
if [ -d "$src/.git" ]; then
    git -C "$src" fetch -q --depth 1 origin "$branch" && git -C "$src" reset -q --hard FETCH_HEAD
else
    git clone -q --depth 1 --branch "$branch" "$repo" "$src"
fi
echo "== JVector $(git -C "$src" log --oneline -1)"

# 2. the SDK's TornadoVM jars, as Maven artifacts of the same version (a release has them on Maven Central)
for a in api cuvs; do
    mvn -q install:install-file -Dfile="$TORNADOVM_HOME/share/java/tornado/tornado-$a-$version.jar" \
        -DgroupId=io.github.beehive-lab -DartifactId=tornado-$a -Dversion="$version" -Dpackaging=jar -DgeneratePom=true
done

# 3. build jvector-base, jvector-twenty (the Panama SIMD provider: a fair CPU baseline) and jvector-gpu.
#    jvector-native is not needed, which avoids its Highway submodule and GCC >= 12 requirement.
(cd "$src" && ./mvnw -q -B -Pgpu -pl jvector-base,jvector-twenty,jvector-gpu -am package -DskipTests -Drat.skip \
    -Dtornadovm.version="$version")
rm -rf "$here/lib" && mkdir -p "$here/lib"
for m in jvector-base jvector-twenty jvector-gpu; do
    cp "$(ls "$src/$m"/target/"$m"-*.jar | grep -v -e sources -e javadoc | head -1)" "$here/lib/"
done
(cd "$src" && ./mvnw -q -B -Pgpu -pl jvector-base dependency:copy-dependencies -DincludeScope=runtime \
    -DoutputDirectory="$here/lib")
echo "== lib/: $(ls "$here/lib" | tr '\n' ' ')"

# 4. cuVS
source "$src/jvector-gpu/cuvs-env.sh" "${JVECTOR_GPU_CUVS:-$HOME/.jvector-gpu/cuvs}"
echo "== cuVS ready; now: bash demos/26-jvector-gpu-index/run.sh"
