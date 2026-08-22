#!/usr/bin/env bash
set -euo pipefail

ORFS_ROOT=/opt/OpenROAD-flow-scripts
OPENROAD_DEB_SHA256=40ed178396b0276a5d5dfbbe695c9de9aac9088157a6655be02b39a0cef07207
OPENROAD_DEB_URL=https://github.com/Precision-Innovations/OpenROAD/releases/download/2024-12-14/openroad_2.0-17598-ga008522d8_amd64-ubuntu-22.04.deb
OPENROAD_DEBIAN11_SHA256=a3918391a20ee817ed40f2f4d75d9c32950155e4602aafd2c03d63ab4f49279c
OPENROAD_DEBIAN11_URL=https://github.com/Precision-Innovations/OpenROAD/releases/download/2024-12-14/openroad_2.0-17598-ga008522d8_amd64-debian11.deb
ORFS_COMMIT=6101364b2d7909dd797e1e3e7f80695401cfa4e4
NANGATE_LIB_SHA256=8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1
NANGATE_TECH_LEF_SHA256=834a79295054cd4209178d1bade67c353863c47bb4b3c22ee38b862b7cec37f2
NANGATE_MACRO_LEF_SHA256=840b01e500826096d1edcc752350834da647fdbf360798f243f8122b52b357c3
NANGATE_LICENSE_SHA256=0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594
NANGATE_MOD_LEF_SHA256=a43aea339f12a57a63497783e508ba16f3da2dc056d3247dec7d99707c2dedef
NANGATE_RCX_SHA256=2f65fafbe2c704b378563c53a680b93cef080c2799997019d43df7d1e5a563e9
NANGATE_TAPCELL_SHA256=ed63997dc12c57c5542e4058338c63e63b13773ded5e0f4b261ac41f769299c0

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y yosys iverilog verilator curl git

if ! command -v openroad >/dev/null 2>&1; then
  glibc_version=$(ldd --version | head -n1 | awk '{print $NF}')
  if dpkg --compare-versions "$glibc_version" lt 2.35; then
    OPENROAD_DEB_URL=$OPENROAD_DEBIAN11_URL
    OPENROAD_DEB_SHA256=$OPENROAD_DEBIAN11_SHA256
  fi
  package_path=$(mktemp --suffix=.deb)
  curl -L --fail --silent --show-error "$OPENROAD_DEB_URL" -o "$package_path"
  echo "$OPENROAD_DEB_SHA256  $package_path" | sha256sum --check --status
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$package_path"
fi

fetch_pinned_platform_file() {
  local relative_path=$1
  local expected_sha256=$2
  local target="${ORFS_ROOT}/${relative_path}"
  if [[ -f "$target" ]] \
      && echo "$expected_sha256  $target" | sha256sum --check --status; then
    return 0
  fi
  local download
  download=$(mktemp)
  curl -L --fail --silent --show-error \
    "https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/${ORFS_COMMIT}/${relative_path}" \
    -o "$download"
  echo "$expected_sha256  $download" | sha256sum --check --status
  install -D -m 0644 "$download" "$target"
  rm -f "$download"
}

# Fetch only the pinned Nangate45 timing/physical/RCX inputs used by this flow.
# Direct downloads avoid cloning the large ORFS history and are fully hash checked.
fetch_pinned_platform_file \
  flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib \
  "$NANGATE_LIB_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef \
  "$NANGATE_TECH_LEF_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.lef \
  "$NANGATE_MACRO_LEF_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/LICENSE \
  "$NANGATE_LICENSE_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef \
  "$NANGATE_MOD_LEF_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/rcx_patterns.rules \
  "$NANGATE_RCX_SHA256"
fetch_pinned_platform_file \
  flow/platforms/nangate45/tapcell.tcl \
  "$NANGATE_TAPCELL_SHA256"
