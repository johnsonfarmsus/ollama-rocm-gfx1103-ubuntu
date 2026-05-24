#!/usr/bin/env bash
# Native ROCm acceleration for Ollama on the AMD Phoenix iGPU (gfx1103)
# Tested on Ubuntu 26.04 with ROCm 7.1 system libs.
#
# This script:
#   1. Installs build dependencies
#   2. Clones likelovewant/ollama-for-amd
#   3. Applies three patches to ml/device.go
#   4. Builds the C++/HIP backend (libggml-hip.so) targeting gfx1103
#   5. Builds the Go binary
#   6. Downloads Fedora 44's rocblas 7.1.1 RPM and extracts gfx1103 Tensile
#      kernels into the system rocBLAS library directory. Critical: the
#      Fedora version must match the system's rocBLAS runtime ABI. Earlier
#      versions of this script used Fedora 43's rocblas 6.4 — those kernels
#      run under 7.1 but trigger frequent "ROCm error: unspecified launch
#      failure" crashes because the kernels were built against an older
#      Tensile/rocBLAS ABI. Always use the kernel version that matches your
#      runtime version.
#   7. Installs the binary + libs, writes a systemd drop-in, restarts Ollama
#   8. Runs a smoke test
#
# Re-runnable: safe to re-run after `apt upgrade librocblas5` overwrites the
# kernel files, since we re-copy them every time.

set -euo pipefail

# ---------- configuration ----------
WORK_DIR="${WORK_DIR:-$HOME/ollama-rocm-gfx1103-build}"
FORK_URL="${FORK_URL:-https://github.com/likelovewant/ollama-for-amd.git}"
FEDORA_RPM_URL="${FEDORA_RPM_URL:-https://kojipkgs.fedoraproject.org/packages/rocblas/7.1.1/7.fc44/x86_64/rocblas-7.1.1-7.fc44.x86_64.rpm}"
GPU_TARGET="${GPU_TARGET:-gfx1103}"
ROCR_PIN_DEVICE="${ROCR_PIN_DEVICE:-}"  # leave empty if you have only one ROCm GPU
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"
OVERRIDE_TEMPLATE="$SCRIPT_DIR/override.conf"

# ---------- helpers ----------
log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

require_root() {
    [ "$EUID" -ne 0 ] || die "do not run this script as root; it will sudo what it needs"
}

# ---------- preflight ----------
require_root

if ! command -v sudo >/dev/null; then
    die "sudo is required"
fi

if ! [ -d "$PATCH_DIR" ] || ! ls "$PATCH_DIR"/*.patch >/dev/null 2>&1; then
    die "patch directory $PATCH_DIR is missing or empty; clone this repo and run setup.sh from inside it"
fi

if ! [ -r "$OVERRIDE_TEMPLATE" ]; then
    die "override.conf template not found at $OVERRIDE_TEMPLATE"
fi

# ---------- kernel version check ----------
# Older kernels (≤ 7.0.0-14 in Ubuntu's numbering, or upstream < 6.11) have
# a gfx11 MES REMOVE_QUEUE bug that causes ~14% of inference requests to
# crash with "ROCm error: unspecified launch failure" and a GPU reset. The
# fix landed in Ubuntu's HWE kernel 7.0.0-15. Bail out early if the running
# kernel is too old — saves the user the surprise of a working ROCm setup
# that's still constantly crashing.
KERNEL=$(uname -r)
KERNEL_MAJOR=$(echo "$KERNEL" | grep -oE "^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+" | head -1)
case "$KERNEL_MAJOR" in
    7.0.0-1[5-9]|7.0.0-[2-9][0-9]|7.[1-9].*|[89].*.*)
        # 7.0.0-15+ on the Ubuntu 26.04 HWE line, or anything newer
        :
        ;;
    7.0.0-1[0-4]|7.0.0-[0-9])
        warn "your running kernel is $KERNEL (older than 7.0.0-15)"
        warn "older kernels have a gfx11 MES bug that causes ~14% of inference"
        warn "requests to crash. To upgrade:"
        warn "  sudo apt install linux-image-generic-hwe-26.04"
        warn "  sudo reboot"
        die  "refusing to continue on a kernel known to be broken for this stack"
        ;;
    *)
        # unrecognized version pattern — warn but proceed
        warn "could not parse kernel version '$KERNEL'; assuming it's new enough"
        ;;
esac

# ---------- step 1: install build dependencies ----------
log "installing build dependencies (this may take a while on first run)"
sudo apt-get update
sudo apt-get install -y \
    golang-go cmake clang rocm-cmake ninja-build git curl \
    rocm-dev libamdhip64-dev librocblas-dev librocm-smi-dev libhipblas-dev \
    libarchive-tools  # for bsdtar; Ubuntu's rpm2cpio can't handle zstd cpio

if ! command -v rocminfo >/dev/null; then
    warn "rocminfo not on PATH; rocm-smi tools may be installed at /opt/rocm/bin or unavailable"
fi

# ---------- step 2: clone or update the fork ----------
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ -d ollama-for-amd/.git ]; then
    log "ollama-for-amd already cloned; resetting to upstream main"
    cd ollama-for-amd
    git fetch origin
    git reset --hard origin/main
    git clean -fd
else
    log "cloning $FORK_URL"
    git clone --depth 1 "$FORK_URL" ollama-for-amd
    cd ollama-for-amd
fi

# ---------- step 3: apply patches ----------
log "applying patches"
for p in "$PATCH_DIR"/*.patch; do
    echo "  applying $(basename "$p")"
    git apply --check "$p" || die "patch $p does not apply cleanly (upstream may have moved on)"
    git apply "$p"
done

# ---------- step 4: build C++/HIP backend ----------
log "configuring CMake (target: $GPU_TARGET)"
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS="$GPU_TARGET"

log "building C++/HIP backend (this is the slow part — 10-30 min depending on your CPU)"
cmake --build build -j"$(nproc)"

# ---------- step 5: build Go binary ----------
log "building Go binary"
go build -trimpath -o ollama .

# ---------- step 6: download Fedora RPM and extract gfx1103 kernels ----------
log "downloading Fedora 44 rocblas 7.1.1 RPM (276 MB)"
mkdir -p "$WORK_DIR/fedora-rocblas"
curl -fsSL -o "$WORK_DIR/rocblas.rpm" "$FEDORA_RPM_URL"

log "extracting gfx1103 Tensile kernels"
cd "$WORK_DIR/fedora-rocblas"
bsdtar -xf "$WORK_DIR/rocblas.rpm"

KERNEL_COUNT=$(find . -name "*${GPU_TARGET}*" | wc -l)
# Fedora 44's rocblas 7.1.1 ships 96 gfx1103 files. Demand at least 50 so
# we catch the case where the RPM URL above silently regresses to an older
# build that doesn't actually have working kernels for this arch.
if [ "$KERNEL_COUNT" -lt 50 ]; then
    die "Fedora RPM didn't contain enough $GPU_TARGET kernels (found $KERNEL_COUNT, expected ~96); maybe the RPM URL is stale or gfx1103 was dropped from newer ROCm"
fi

# Locate the system rocBLAS library directory
SYS_ROCBLAS_DIR="$(find /usr/lib -path '*rocblas*library' -type d 2>/dev/null | head -1)"
if [ -z "$SYS_ROCBLAS_DIR" ]; then
    die "could not find system rocBLAS library directory under /usr/lib"
fi
log "system rocBLAS library dir: $SYS_ROCBLAS_DIR"

log "installing $KERNEL_COUNT $GPU_TARGET kernel files into system rocBLAS"
sudo cp "$WORK_DIR/fedora-rocblas/usr/lib64/rocblas/library/"*"${GPU_TARGET}"* "$SYS_ROCBLAS_DIR/"
INSTALLED=$(find "$SYS_ROCBLAS_DIR" -name "*${GPU_TARGET}*" | wc -l)
log "$INSTALLED $GPU_TARGET files now present in $SYS_ROCBLAS_DIR"

# ---------- step 7: install Ollama binary + libs ----------
log "installing patched Ollama"
sudo systemctl stop ollama 2>/dev/null || true

if [ -e /usr/local/bin/ollama ] && ! [ -e /usr/local/bin/ollama.bak ]; then
    sudo cp /usr/local/bin/ollama /usr/local/bin/ollama.bak
fi
sudo install -m 0755 -o root -g root "$WORK_DIR/ollama-for-amd/ollama" /usr/local/bin/ollama

if [ -d /usr/local/lib/ollama ] && ! [ -d /usr/local/lib/ollama.bak ]; then
    sudo mv /usr/local/lib/ollama /usr/local/lib/ollama.bak
fi
sudo mkdir -p /usr/local/lib/ollama
sudo cp -a "$WORK_DIR/ollama-for-amd/build/lib/ollama/." /usr/local/lib/ollama/

log "writing systemd drop-in"
sudo mkdir -p /etc/systemd/system/ollama.service.d
if [ -n "$ROCR_PIN_DEVICE" ]; then
    # User wants a specific device pinned — substitute the index
    sed "s/^Environment=\"ROCR_VISIBLE_DEVICES=.*\"$/Environment=\"ROCR_VISIBLE_DEVICES=${ROCR_PIN_DEVICE}\"/" \
        "$OVERRIDE_TEMPLATE" | sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
else
    # Single-GPU system — drop the pin line
    grep -v "ROCR_VISIBLE_DEVICES" "$OVERRIDE_TEMPLATE" | \
        sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
fi

sudo systemctl daemon-reload

if systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    log "starting Ollama"
    sudo systemctl start ollama
    sleep 5
else
    warn "ollama.service unit not registered; assuming first-time install — start manually with: sudo systemctl enable --now ollama"
fi

# ---------- step 8: smoke test ----------
log "smoke test"
DETECTED=$(sudo journalctl -u ollama --since "30 seconds ago" --no-pager 2>/dev/null | \
    grep "inference compute" | grep -o "compute=gfx[0-9]*" | head -1 || true)

if [ "$DETECTED" = "compute=$GPU_TARGET" ]; then
    log "✅ SUCCESS — Ollama reports $DETECTED"
else
    warn "Ollama is not reporting native $GPU_TARGET in 'inference compute' logs. Got: ${DETECTED:-(nothing)}"
    warn "check 'sudo journalctl -u ollama --since \"1 minute ago\"' for the full picture"
fi

cat <<EOF

Setup complete. Verify by running:

    curl -s http://127.0.0.1:11434/api/chat -d '{
      "model": "gemma4:e2b",
      "stream": false,
      "messages": [{"role": "user", "content": "Hello"}]
    }' | python3 -m json.tool

(Pull a model first if you don't have one: 'ollama pull gemma4:e2b')

Build artifacts left at: $WORK_DIR
Re-run this script anytime apt upgrades librocblas5 — it will replace the
gfx1103 kernel files that the upgrade wipes out.
EOF
