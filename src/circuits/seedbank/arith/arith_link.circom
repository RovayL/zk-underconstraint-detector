pragma circom 2.2.2;

template ArithLink() {
    signal input a;
    signal input b;
    signal input c;
    signal input y_pub;     // intended to be public at the protocol level
    signal output y_out;

    y_out <== a * b + c;
    // enforce linkage to the external/public value
    y_out === y_pub;
}

component main = ArithLink();
