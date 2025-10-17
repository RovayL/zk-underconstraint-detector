pragma circom 2.2.2;
include "circomlib/circuits/poseidon.circom";

template PoseidonChain(T) {
    signal input x;
    signal output y;

    signal level[T + 1];
    level[0] <== x;

    component h[T];
    for (var i = 0; i < T; i++) {
        h[i] = Poseidon(2);
        h[i].inputs[0] <== level[i];
        h[i].inputs[1] <== 0;      // fixed second limb
        level[i + 1]   <== h[i].out;
    }

    y <== level[T];
}

component main = PoseidonChain(5);
