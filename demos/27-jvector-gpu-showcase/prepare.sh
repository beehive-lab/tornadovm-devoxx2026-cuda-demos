#!/usr/bin/env bash
# One-time data preparation for demo 27 (about 10 minutes, before the talk).
#
#   bash demos/27-jvector-gpu-showcase/prepare.sh <dataDir>
#
# <dataDir> must hold JVector's public ada-002 files (download links in README.md):
#   ada_002_100k_base_99287.fvecs  ada_002_100k_query_10000.fvecs  ada_002_100k_gt_ip_100.ivecs
#   ada_002_1M_base_982790.fvecs   ada_002_1M_query_10000.fvecs    ada_002_1M_gt_ip_100.ivecs
# This adds what JVector itself builds on the CPU, which acts 3 and 4 compare against:
#   cpu-graph-1M.bin (JVector's 1M graph) and segments/segment-{0..3}-of-4.idx (4 on-disk segments).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] || { sed -n '4,10p' "$0"; exit 1; }
exec bash "$here/run.sh" tornado "$1" prepare
