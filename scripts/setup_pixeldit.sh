#!/usr/bin/env bash
set -euo pipefail

PIXELDIT_URL="https://github.com/NVlabs/PixelDiT.git"
PIXELDIT_COMMIT="41f73006ae532b0b41fee72b181dc22891a5a01a"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${1:-$REPO_ROOT/third_party/PixelDiT}"

mkdir -p "$(dirname "$TARGET")"
if [[ -d "$TARGET/.git" ]]; then
    ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
    if [[ "$ACTUAL" != "$PIXELDIT_COMMIT" ]]; then
        echo "Existing PixelDiT checkout is $ACTUAL; expected $PIXELDIT_COMMIT." >&2
        echo "Move it aside or check out the pinned commit explicitly." >&2
        exit 1
    fi
    echo "PixelDiT is already pinned at $PIXELDIT_COMMIT"
    exit 0
fi
if [[ -e "$TARGET" ]]; then
    echo "Target exists but is not a Git checkout: $TARGET" >&2
    exit 1
fi

git clone --filter=blob:none "$PIXELDIT_URL" "$TARGET"
git -C "$TARGET" checkout --detach "$PIXELDIT_COMMIT"
echo "PixelDiT installed at $TARGET"
echo "Install adapter/reference dependencies with: pip install -r $REPO_ROOT/requirements-pixeldit.txt"
