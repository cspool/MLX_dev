#!/usr/bin/env bash
set -euo pipefail

ORFS_ROOT=/opt/OpenROAD-flow-scripts
OPENROAD_DEB_SHA256=40ed178396b0276a5d5dfbbe695c9de9aac9088157a6655be02b39a0cef07207
OPENROAD_DEB_URL=https://github.com/Precision-Innovations/OpenROAD/releases/download/2024-12-14/openroad_2.0-17598-ga008522d8_amd64-ubuntu-22.04.deb
ORFS_COMMIT=6101364b2d7909dd797e1e3e7f80695401cfa4e4

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y yosys iverilog verilator curl git

if ! command -v openroad >/dev/null 2>&1; then
  package_path=$(mktemp --suffix=.deb)
  curl -L --fail --silent --show-error "$OPENROAD_DEB_URL" -o "$package_path"
  echo "$OPENROAD_DEB_SHA256  $package_path" | sha256sum --check --status
  DEBIAN_FRONTEND=noninteractive apt-get install -y "$package_path"
fi

if [[ ! -d "$ORFS_ROOT/.git" ]]; then
  git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git "$ORFS_ROOT"
fi
git -C "$ORFS_ROOT" fetch origin "$ORFS_COMMIT"
git -C "$ORFS_ROOT" checkout --detach "$ORFS_COMMIT"
