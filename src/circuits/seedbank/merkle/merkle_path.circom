pragma circom 2.2.2;
include "circomlib/circuits/poseidon.circom";

template MerklePath(depth) {
    signal input leaf;
    signal input root;
    signal input pathElements[depth];
    signal input pathIndex[depth];   // 0 = leaf on left, 1 = leaf on right

    // Boolean bits
    for (var i = 0; i < depth; i++) {
        pathIndex[i] * (1 - pathIndex[i]) === 0;
    }

    // One signal per tree level
    signal level[depth + 1];
    level[0] <== leaf;

    component h[depth];
    for (var i = 0; i < depth; i++) {
        h[i] = Poseidon(2);

        // One-mul-per-constraint forms (keeps it quadratic)
        // left  = level[i] + b*(path - level[i])
        // right = path + b*(level[i] - path)
        h[i].inputs[0] <== level[i]           + pathIndex[i] * (pathElements[i] - level[i]);
        h[i].inputs[1] <== pathElements[i]    + pathIndex[i] * (level[i] - pathElements[i]);

        level[i + 1] <== h[i].out;
    }

    level[depth] === root;
}

// If you want `root` public in snarkjs:
component main { public [root] } = MerklePath(3);
// Otherwise, plain:
// component main = MerklePath(3);
