#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEEDBANK_DIR="$ROOT/src/circuits/seedbank"
BUILD_DIR="$SEEDBANK_DIR/build"
NODE_MODULES="$ROOT/node_modules"   # <— add this
mkdir -p "$BUILD_DIR"

build_one() {
  local name="$1"; local dir="$2"; local main="$3"; local input="$4"
  echo "[build] $name"
  pushd "$dir" >/dev/null
  args=(-l "$NODE_MODULES" -l node_modules --r1cs --wasm)
  circom "$main" "${args[@]}"
  snarkjs r1cs export json "${main%.circom}.r1cs" "$BUILD_DIR/${name}.r1cs.json"
  if [ -f "$input" ]; then
    node ${main%.circom}_js/generate_witness.js ${main%.circom}_js/${main%.circom}.wasm "$input" "$BUILD_DIR/${name}.wtns"
    snarkjs wtns export json "$BUILD_DIR/${name}.wtns" "$BUILD_DIR/${name}.witness.json"
  fi
  popd >/dev/null
}

# (Optional) ensure dep exists at the root
if [ ! -f "$NODE_MODULES/circomlib/circuits/poseidon.circom" ] && \
   [ ! -f "$NODE_MODULES/circomlib/src/poseidon.circom" ]; then
  echo "[deps] installing circomlib@^2 under $NODE_MODULES"
  (cd "$ROOT" && npm i circomlib@^2)
fi

build_one "merkle_path_d3" "$SEEDBANK_DIR/merkle" merkle_path.circom input.small.json
build_one "range_check_32" "$SEEDBANK_DIR/range" range_check.circom input.small.json
build_one "poseidon_chain_t5" "$SEEDBANK_DIR/poseidon" poseidon_chain.circom input.small.json
build_one "arith_link" "$SEEDBANK_DIR/arith" arith_link.circom input.small.json
echo "[done] artifacts in $BUILD_DIR"
