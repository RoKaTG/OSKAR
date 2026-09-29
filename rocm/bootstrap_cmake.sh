#!/usr/bin/env bash
#
# The node has no cmake, and OSKAR needs >= 3.18 (HIP language support needs
# >= 3.21). This installs a self-contained official binary under
# ~/.local/oskar-cmake. No root, no package manager, nothing system-wide.
#
#     bash rocm/bootstrap_cmake.sh
#     export PATH="$HOME/.local/oskar-cmake/bin:$PATH"
#
# The tarball is downloaded from Kitware. On a node with no outbound network,
# fetch it elsewhere and drop it in rocm/ -- the script prefers a local copy
# over downloading, and says so if it can do neither.

set -euo pipefail

VERSION="${CMAKE_VERSION:-3.28.6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Installed outside the source tree on purpose, so that replacing or
# re-checking-out the source does not delete it.
DEST="${OSKAR_TOOLCHAIN_DIR:-$HOME/.local/oskar-cmake}"
TARBALL="cmake-${VERSION}-linux-x86_64.tar.gz"
URL="https://github.com/Kitware/CMake/releases/download/v${VERSION}/${TARBALL}"

echo "cmake version : $VERSION"
echo "install into  : $DEST"
echo "source        : $URL"
echo

# Already good enough? Then do nothing.
if command -v cmake >/dev/null 2>&1; then
    have="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
    if [ "$(printf '%s\n3.21.0\n' "$have" | sort -V | head -1)" = "3.21.0" ]; then
        echo "cmake $have is already installed and recent enough. Nothing to do."
        exit 0
    fi
    echo "Found cmake $have, which is too old for the HIP language. Continuing."
fi

mkdir -p "$DEST"
cd "$DEST"

if [ -f "$HERE/$TARBALL" ]; then
    echo "Using tarball already staged at $HERE/$TARBALL"
    cp "$HERE/$TARBALL" .
elif [ ! -f "$TARBALL" ]; then
    echo "Downloading..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TARBALL" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$TARBALL" "$URL"
    else
        echo "ERROR: neither curl nor wget is available." >&2
        echo "Fetch $URL elsewhere and place it in $HERE, then re-run." >&2
        exit 1
    fi
fi

tar xzf "$TARBALL" --strip-components=1
rm -f "$TARBALL"

echo
echo "Installed: $("$DEST/bin/cmake" --version | head -1)"
echo
echo "Add it to your path for this session:"
echo "    export PATH=\"$DEST/bin:\$PATH\""
