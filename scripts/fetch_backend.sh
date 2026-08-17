#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {dsagen|assassyn|accel-sim|timeloop}" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
BACKEND_ROOT="${PROJECT_ROOT}/third_party"
BACKEND="$1"

case "${BACKEND}" in
  dsagen)
    REPOSITORY="https://github.com/PolyArch/dsa-framework.git"
    REVISION="273e141a519d12138ee0fbc9743059d13e9b5a64"
    DESTINATION="${BACKEND_ROOT}/dsa-framework"
    ;;
  assassyn)
    REPOSITORY="https://github.com/Synthesys-Lab/assassyn.git"
    REVISION="6a99ade0e9380c93d4817f7de51b7edd8a473dd2"
    DESTINATION="${BACKEND_ROOT}/assassyn"
    ;;
  accel-sim)
    REPOSITORY="https://github.com/accel-sim/accel-sim-framework.git"
    REVISION="c5296df152c99a28dd64e5d9560bd58a8fd2e774"
    DESTINATION="${BACKEND_ROOT}/accel-sim-framework"
    ;;
  timeloop)
    REPOSITORY="https://github.com/NVlabs/timeloop.git"
    REVISION="32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa"
    DESTINATION="${BACKEND_ROOT}/timeloop"
    ;;
  *)
    echo "Unknown backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

mkdir -p "${BACKEND_ROOT}"
if [ ! -d "${DESTINATION}/.git" ]; then
  git clone --filter=blob:none "${REPOSITORY}" "${DESTINATION}"
fi
git -C "${DESTINATION}" fetch --depth 1 origin "${REVISION}"
git -C "${DESTINATION}" checkout --detach "${REVISION}"

echo "Fetched ${BACKEND} at $(git -C "${DESTINATION}" rev-parse HEAD)"
if [ "${BACKEND}" = "dsagen" ]; then
  echo "DSAGEN submodules are intentionally not initialized; the official full stack is about 70 GB."
fi
if [ "${BACKEND}" = "assassyn" ]; then
  echo "Assassyn is inspect-only: submodules are intentionally not initialized and no top-level license was found at this pin."
fi
