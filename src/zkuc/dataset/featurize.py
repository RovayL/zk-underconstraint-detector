from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from zkuc.core.r1cs_io import load_r1cs_json
from zkuc.core.witness_io import assemble_full_z
from zkuc.core.jacobian import matvec_rows_modp, jacobian_submatrix_dense_modp
from zkuc.core.rank_modp import algebraic_rank_modp, structural_rank

# ---- structural features from row support only ----
def structural_from_rows(A_rows, B_rows, C_rows, n_vars: int) -> Dict[str, Any]:
    m = len(A_rows)
    # per-row variable sets
    row_sets: List[set] = []
    for i in range(m):
        s = set(A_rows[i].keys()) | set(B_rows[i].keys()) | set(C_rows[i].keys())
        row_sets.append(s)
    fanin = [len(s) for s in row_sets]
    avg_fanin = float(np.mean(fanin)) if fanin else 0.0
    # per-var counts
    counts = np.zeros(n_vars, dtype=int)
    for s in row_sets:
        for v in s:
            if 0 <= v < n_vars:
                counts[v] += 1
    nonzero = counts[counts > 0]
    avg_fanout = float(np.mean(nonzero)) if nonzero.size else 0.0
    mult_share = float(np.mean([(len(A_rows[i]) > 0) and (len(B_rows[i]) > 0) for i in range(m)])) if m else 0.0
    # duplicate row support
    sigs = []
    for i in range(m):
        sigs.append((
            tuple(sorted(A_rows[i].keys())),
            tuple(sorted(B_rows[i].keys())),
            tuple(sorted(C_rows[i].keys())),
        ))
    dup_rows_frac = float(1.0 - len(set(sigs)) / m) if m else 0.0
    return {
        "avg_fanin": avg_fanin,
        "avg_fanout": avg_fanout,
        "mult_share": mult_share,
        "dup_rows_frac": dup_rows_frac,
    }

# ---- probe-time stats without requiring a satisfying witness ----
def probe_stats(
    R, rng: random.Random, trials: int = 10, subset: int = 32,
    freeze_const: bool = True, freeze_pub_inputs: bool = True, seed: int = 0,
    rank_mode: str = "algebraic",
) -> Dict[str, Any]:
    p = R.prime
    # freeze policy
    frozen = set()
    if freeze_const and R.n_vars > 0:
        frozen.add(0)
    if freeze_pub_inputs and R.n_inputs > 0:
        for v in range(1, min(1 + R.n_inputs, R.n_vars)):
            frozen.add(v)
    unfrozen = [j for j in range(R.n_vars) if j not in frozen]
    if not unfrozen:
        return {"rank_mean":0.0,"rank_std":0.0,"nullity_mean":0.0,"nullity_std":0.0,
                "dead_rows_frac_mean":0.0,"wiggle_rate_mean":0.0,"trials":trials,"subset":0}

    # build a random witness vector z (does NOT need to satisfy constraints)
    rng_np = np.random.default_rng(seed)
    z = np.zeros(R.n_vars, dtype=object)
    z[0] = 1  # constant-1
    for j in range(1, R.n_vars):
        z[j] = int(rng_np.integers(low=1, high=min(2**32, p-1))) % p

    Az = matvec_rows_modp(R.A_rows, z, p)
    Bz = matvec_rows_modp(R.B_rows, z, p)

    ranks, nulls, dead_fracs, wiggles = [], [], [], []
    for _ in range(trials):
        k = min(subset, len(unfrozen))
        cols = sorted(rng.sample(unfrozen, k)) if k > 0 else []
        J = jacobian_submatrix_dense_modp(R.A_rows, R.B_rows, R.C_rows, Az, Bz, cols, p)
        if rank_mode == "algebraic":
            r = algebraic_rank_modp(J, p)
        else:
            # structural_rank over reals is the float/SVD control
            r = structural_rank(J, 2)
        ranks.append(r)
        nulls.append(max(J.shape[1] - r, 0))
        # dead rows & wiggle proxy
        row_nnz = (np.count_nonzero(J, axis=1))
        col_nnz = (np.count_nonzero(J, axis=0))
        dead_fracs.append(float((row_nnz == 0).sum()) / float(J.shape[0] if J.shape[0] else 1))
        wiggles.append(0.0 if J.shape[1] == 0 else float((col_nnz > 0).sum()) / float(J.shape[1]))
    def ms(x):
        if not x: return 0.0, 0.0
        arr = np.array(x, dtype=float)
        return float(arr.mean()), float(arr.std())
    rm, rs = ms(ranks)
    nm, ns = ms(nulls)
    dm, _ = ms(dead_fracs)
    wm, _ = ms(wiggles)
    return {
        "rank_mean": rm, "rank_std": rs,
        "nullity_mean": nm, "nullity_std": ns,
        "dead_rows_frac_mean": dm,
        "wiggle_rate_mean": wm,
        "trials": trials, "subset": k if unfrozen else 0, "rank_mode": rank_mode
    }

def compute_all_features(R, rng: random.Random, probe_cfg: Dict[str,Any]) -> Dict[str,Any]:
    struct = structural_from_rows(R.A_rows, R.B_rows, R.C_rows, R.n_vars)
    cfg = dict(probe_cfg)
    rank_mode = cfg.pop("rank_mode", "algebraic")
    probe = probe_stats(R, rng, rank_mode=rank_mode, **cfg)
    base = {
        "circuit_id": Path(r1cs_path).stem,
        "n_constraints": int(R.n_constraints),
        "n_vars": int(R.n_vars),
        "n_inputs": int(R.n_inputs),
        "n_outputs": int(R.n_outputs),
        "ratio_c_v": float(R.n_constraints / max(1, R.n_vars)),
    }
    base.update(struct); base.update(probe)
    return base

def featurize_file(r1cs_path: str, rng: random.Random, probe_cfg: Dict[str,Any]) -> Dict[str,Any]:
    R = load_r1cs_json(r1cs_path)
    return compute_all_features(R, rng, probe_cfg)
