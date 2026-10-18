#!/usr/bin/env bash
# Builds the WASM modules and writes them into web/src/generated/ as base64
# TypeScript modules, so the npm package needs no asset copying. The brain is
# built once per detector (Silero only, Auricle only) so a page loads just the
# model it asked for; the wasm-bindgen JS glue is shared because the exports
# are identical.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=wasm32-unknown-unknown
OUT=web/src/generated
mkdir -p "$OUT/brain"
OPT_FLAGS="-O3 --enable-simd --enable-bulk-memory --enable-bulk-memory-opt --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals"
have_opt=0; command -v wasm-opt >/dev/null && have_opt=1 || echo "wasm-opt not found; shipping unoptimized wasm" >&2

export RUSTFLAGS="-C target-feature=+simd128"
to_ts() { printf 'export const %s = "%s";\n' "$2" "$(base64 < "$1" | tr -d '\n')" > "$3"; }
report() { echo "$1: $(wc -c < "$2") bytes ($(gzip -c "$2" | wc -c) gz)"; }

# Worklet (no model inside).
cargo build --release --target $TARGET -p auricle-worklet
WORKLET=target/$TARGET/release/auricle_worklet.wasm
if [ $have_opt = 1 ]; then wasm-opt $OPT_FLAGS -o "$WORKLET.opt" "$WORKLET" && WORKLET="$WORKLET.opt"; fi
to_ts "$WORKLET" WORKLET_WASM_B64 "$OUT/worklet_wasm.ts"
report worklet "$WORKLET"

# Brain, one build per detector. The glue (auricle_brain.js, .d.ts) comes from the last build.
for model in ${MODELS:-auricle}; do  # MODELS="silero auricle" also builds the Silero brain for bench/browser
  cargo build --release --target $TARGET -p auricle-brain --no-default-features --features "$model"
  wasm-bindgen --target web --omit-default-module-path --out-dir "$OUT/brain" --out-name auricle_brain "target/$TARGET/release/auricle_brain.wasm"
  WASM="$OUT/brain/auricle_brain_bg.wasm"
  if [ $have_opt = 1 ]; then wasm-opt $OPT_FLAGS -o "$WASM.opt" "$WASM" && mv "$WASM.opt" "$WASM"; fi
  to_ts "$WASM" BRAIN_WASM_B64 "$OUT/brain_wasm_$model.ts"
  report "brain ($model)" "$WASM"
done
rm -f "$OUT/brain_wasm.ts"
