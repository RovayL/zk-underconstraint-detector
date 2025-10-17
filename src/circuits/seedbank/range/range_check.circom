pragma circom 2.2.2;
include "circomlib/circuits/bitify.circom";

template RangeCheck(n) {
    signal input x;
    signal output ok;

    component n2b = Num2Bits(n);
    n2b.in <== x;

    ok <== 1;
}

component main = RangeCheck(32);
