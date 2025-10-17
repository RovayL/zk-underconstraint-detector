# zk-underconstraint-detector


Lightweight toolkit to **extract features** from R1CS (Circom/snarkjs export JSON) and compute **structural statistics** to help detect *under‑constrained* zero‑knowledge circuits.



## High Level Model Description (Version 1)
```mathematica
R1CS JSON  ─┐
            ├─► Structural features       ┐
(seedbank & │                             ├─► Feature vector X ─► Standardize ─► PCA(d) ─► GMM(k=2)
 originals) ┘  Probe-time features (rank) ┘                                      │            │
                                                                                 │            ├─► {buggy_post, llr}
                                                                                 └────────────┘
                                      ┌──────────────────────────────────────────────────────────────┐
                                      │  Calibrator (LogReg/Tree): uses {buggy_post, llr, PCA comps} │
                                      └──────────────────────────────────────────────────────────────┘
                                                                                   │
                                                                                   └─► score = P(UC | features)
                                                                                   │
                                                                       thresholds (FPR≈1%, 5%)
                                                                                   │
                                                                                 decision

```



## Quickstart


1) (Optional) install editable:


```bash
pip install -e .


