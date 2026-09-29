#!/usr/bin/env bash
#
# Compile-check the HIP port on a machine with no AMD GPU and no ROCm.
#
#     bash rocm/compile_check_local.sh            # host + device
#     STAGE=host   bash rocm/compile_check_local.sh
#     STAGE=device bash rocm/compile_check_local.sh
#
# Two stages, both against the *real* ROCm headers for the version on the
# node (fetched once from GitHub into a cache directory):
#
#   host    Every OSKAR library source that the system compiler builds, with
#           the same definitions CMake's hip::host target supplies. This is
#           where the CUDA-to-HIP shim and the vector-type handling have to
#           be right.
#
#   device  Every .cu file, compiled by the local clang for the node's GPU
#           architecture through to AMDGPU assembly. This exercises parsing,
#           overload resolution, host/device attribute checks and the AMDGPU
#           backend. Nothing is linked, so the ROCm device libraries are not
#           needed.
#
# Limitations: it says nothing about runtime behaviour or numerical results,
# does not link, and uses upstream clang rather than the AMD clang shipped
# with ROCm. A pass means the code compiles for the target, not that it works.
#
# The ROCm headers must be the real ones. Stub headers omit the vector types
# that hip_runtime_api.h and hipfft.h declare, and so miss a whole class of
# conflict.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ROCM_VERSION="${ROCM_VERSION:-6.1.0}"
HIP_PATCH="${HIP_PATCH:-40091}"          # as reported by `hipconfig --version`
GPU_ARCH="${GPU_ARCH:-gfx90a}"
STAGE="${STAGE:-all}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CACHE="${OSKAR_ROCM_HEADERS:-$HOME/.cache/oskar-rocm-headers/rocm-$ROCM_VERSION}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# -----------------------------------------------------------------------------
# Real ROCm headers, assembled into the layout a ROCm install has.
# -----------------------------------------------------------------------------
fetch_headers() {
    local src="$CACHE/src"
    mkdir -p "$src"
    for repo in HIP clr hipFFT; do
        [ -d "$src/$repo" ] && continue
        echo "  fetching ROCm/$repo @ rocm-$ROCM_VERSION (headers only)"
        git -c advice.detachedHead=false clone -q --depth 1 --branch "rocm-$ROCM_VERSION" --filter=blob:none \
            --sparse "https://github.com/ROCm/$repo.git" "$src/$repo" || return 1
    done
    (cd "$src/HIP"    && git sparse-checkout set include)          || return 1
    (cd "$src/clr"    && git sparse-checkout set hipamd/include)   || return 1
    (cd "$src/hipFFT" && git sparse-checkout set library/include)  || return 1

    local inc="$CACHE/root/include"
    rm -rf "$CACHE/root"
    mkdir -p "$inc/hip/amd_detail" "$inc/hipfft" "$CACHE/root/bin"
    cp -r "$src/HIP/include/hip/"* "$inc/hip/"
    cp -r "$src/clr/hipamd/include/hip/amd_detail/"* "$inc/hip/amd_detail/"
    cp "$src/hipFFT/library/include/hipfft/"*.h "$inc/hipfft/"

    # Files a real install generates at build time.
    local maj="${ROCM_VERSION%%.*}" rest="${ROCM_VERSION#*.}"
    local min="${rest%%.*}"
    cat > "$inc/hip/hip_version.h" <<EOF
#ifndef HIP_VERSION_H
#define HIP_VERSION_H
#define HIP_VERSION_MAJOR $maj
#define HIP_VERSION_MINOR $min
#define HIP_VERSION_PATCH $HIP_PATCH
#define HIP_VERSION_GITHASH ""
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#endif
EOF
    sed -e 's/@hipfft_VERSION_MAJOR@/1/;s/@hipfft_VERSION_MINOR@/0/' \
        -e 's/@hipfft_VERSION_PATCH@/0/;s/@hipfft_VERSION_TWEAK@/0/' \
        "$src/hipFFT/library/include/hipfft/hipfft-version.h.in" \
        > "$inc/hipfft/hipfft-version.h"
    printf '#ifndef HIPFFT_EXPORT_H\n#define HIPFFT_EXPORT_H\n#define HIPFFT_EXPORT\n#endif\n' \
        > "$inc/hipfft/hipfft-export.h"
    printf 'HIP_VERSION_MAJOR=%s\nHIP_VERSION_MINOR=%s\nHIP_VERSION_PATCH=%s\n' \
        "$maj" "$min" "$HIP_PATCH" > "$CACHE/root/bin/.hipVersion"
    touch "$CACHE/.complete"
}

if [ ! -f "$CACHE/.complete" ]; then
    echo "ROCm $ROCM_VERSION headers not cached; fetching into $CACHE"
    if ! fetch_headers; then
        echo "Could not fetch the ROCm headers (network?). Nothing was checked."
        exit 2
    fi
fi
ROOT="$CACHE/root"
INC="$ROOT/include"

# Generated OSKAR headers (oskar_version.h and friends) live in a build tree.
BUILD_INC=""
for b in build build-hip build-cpu; do
    [ -d "$b/oskar" ] && { BUILD_INC="-I$b -I$b/oskar"; break; }
done
if [ -z "$BUILD_INC" ]; then
    echo "No configured build directory found; run cmake once so that the"
    echo "generated headers exist (any backend will do)."
    exit 2
fi

COMMON="-DOSKAR_HAVE_HIP -DOSKAR_HAVE_GPU -DOSKAR_NO_MS -DSOURCE_PATH_SIZE=40 \
    -I. -Ioskar -Iextern -Iextern/Random123 -Iextern/Random123/features \
    -Iextern/rapidxml-1.13 -Iextern/cfitsio $BUILD_INC"
export COMMON INC ROOT GPU_ARCH WORK

total_bad=0

# -----------------------------------------------------------------------------
# Host stage.
# -----------------------------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = host ]; then
    # Mirrors hip::host: the AMD platform macros and ROCm's include directory.
    # ms/ needs casacore and harp/ needs HARP, neither of which is on the node.
    find oskar \( -path '*/test' -o -path oskar/ms -o -path oskar/harp \) -prune \
        -o \( -name '*.c' -o -name '*.cpp' \) -print | sort > "$WORK/host.txt"
    n=$(wc -l < "$WORK/host.txt")
    echo
    echo "host: $n library sources, system compiler, real ROCm $ROCM_VERSION headers"
    : > "$WORK/host_failed.txt"
    xargs -P "$JOBS" -I{} sh -c '
        f="{}"; case "$f" in *.c) cc=gcc ;; *) cc=g++ ;; esac
        log="$WORK/$(echo "$f" | tr / _).log"
        $cc -fsyntax-only -fopenmp -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 \
            -I"$INC" -I"$INC/hipfft" $COMMON "$f" > "$log" 2>&1 \
            || echo "$f" >> "$WORK/host_failed.txt"
    ' < "$WORK/host.txt"
    bad=$(wc -l < "$WORK/host_failed.txt")
    total_bad=$((total_bad + bad))
    if [ "$bad" -eq 0 ]; then
        echo "  all $n compile"
    else
        echo "  $bad of $n FAILED:"
        sort "$WORK/host_failed.txt" | while read -r f; do
            echo "    $f"
            grep -m3 "error" "$WORK/$(echo "$f" | tr / _).log" | sed 's/^/        /'
        done
    fi
fi

# -----------------------------------------------------------------------------
# Device stage.
# -----------------------------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = device ]; then
    CLANG=""
    for c in clang++ clang++-20 clang++-19 clang++-18 clang++-17; do
        if command -v "$c" >/dev/null 2>&1 && \
                "$c" -print-targets 2>/dev/null | grep -q amdgcn; then
            CLANG="$c"; break
        fi
    done
    echo
    if [ -z "$CLANG" ]; then
        echo "device: skipped -- no local clang with the amdgcn target"
    else
        export CLANG
        find oskar -name '*.cu' | sort > "$WORK/dev.txt"
        n=$(wc -l < "$WORK/dev.txt")
        echo "device: $n .cu files, $($CLANG --version | head -1), --offload-arch=$GPU_ARCH"
        : > "$WORK/dev_failed.txt"
        xargs -P "$JOBS" -I{} sh -c '
            f="{}"; b="$(echo "$f" | tr / _)"
            $CLANG -x hip --offload-arch="$GPU_ARCH" -nogpulib \
                --rocm-path="$ROOT" --hip-path="$ROOT" -I"$INC/hipfft" \
                --cuda-device-only -S -O3 $COMMON "$f" -o "$WORK/$b.s" \
                > "$WORK/$b.dev.log" 2>&1 || echo "$f" >> "$WORK/dev_failed.txt"
        ' < "$WORK/dev.txt"
        bad=$(wc -l < "$WORK/dev_failed.txt")
        total_bad=$((total_bad + bad))
        if [ "$bad" -eq 0 ]; then
            k=$(cat "$WORK"/*.s 2>/dev/null | grep -c '^[[:space:]]*\.amdhsa_kernel[[:space:]]')
            echo "  all $n reach $GPU_ARCH assembly ($k kernels)"
        else
            echo "  $bad of $n FAILED:"
            sort "$WORK/dev_failed.txt" | while read -r f; do
                echo "    $f"
                grep "error" "$WORK/$(echo "$f" | tr / _).dev.log" | sort | uniq -c \
                    | sort -rn | head -4 | sed 's/^/        /'
            done
        fi
    fi
fi

echo
if [ "$total_bad" -eq 0 ]; then
    echo "PASS: everything compiles for the target. Runtime behaviour untested."
else
    echo "FAIL: $total_bad translation unit(s)."
fi
exit "$total_bad"
