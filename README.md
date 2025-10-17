# zk-underconstraint-detector


Lightweight toolkit to **extract features** from R1CS (Circom/snarkjs export JSON) and compute **structural statistics** to help detect *under‑constrained* zero‑knowledge circuits.


This repository contains the **M0/M1 milestones**:
- M0: Environment, repo layout, quickstart.
- M1: R1CS parsing + structural features, and a minimal CLI.


> Next milestones (M2+) will add probe‑time Jacobian rank, PCA+GMM modeling, calibration, and evaluation.


## Quickstart


1) (Optional) install editable:


```bash
pip install -e .
