from __future__ import annotations
from typing import Dict, List, Tuple
import random
import numpy as np
from scipy.sparse import csr_matrix
from ..core.r1cs_io import R1CS, BN254_PRIME
from ..core.fieldla import rank_from_row_dicts

def _row_dot(mat: csr_matrix, i: int, w: List[int], p: int) -> int:
    start, end = mat.indptr[i], mat.indptr[i+1]
    idx = mat.indices[start:end]
    dat = mat.data[start:end]
    s = 0
    for j, a in zip(idx, dat):
        s = (s + int(a) * w[j]) % p
    return s

def _row_items(mat: csr_matrix, i: int):
    start, end = mat.indptr[i], mat.indptr[i+1]
    idx = mat.indices[start:end]
    dat = mat.data[start:end]
    return [(int(j), int(dat[k])) for k, j in enumerate(idx)]

def jacobian_rows_dict(r: R1CS, w: List[int], p: int, freeze_prefix: int):
    A, B, C = r.A, r.B, r.C
    m, n = A.shape
    n_free = max(0, n - (freeze_prefix + 1))
    dead = 0
    rows: List[Dict[int,int]] = []

    for i in range(m):
        sA = _row_dot(A, i, w, p)
        sB = _row_dot(B, i, w, p)
        d: Dict[int,int] = {}

        for j, aij in _row_items(A, i):
            if j <= freeze_prefix: continue
            if aij != 0 and sB != 0:
                d[j] = (d.get(j, 0) + aij * sB) % p
        for j, bij in _row_items(B, i):
            if j <= freeze_prefix: continue
            if bij != 0 and sA != 0:
                d[j] = (d.get(j, 0) + bij * sA) % p
        for j, cij in _row_items(C, i):
            if j <= freeze_prefix: continue
            if cij != 0:
                v = (d.get(j, 0) - cij) % p
                if v == 0:
                    d.pop(j, None)
                else:
                    d[j] = v
        if len(d) == 0:
            dead += 1
        rows.append(d)

    dead_frac = float(dead) / m if m > 0 else 0.0
    return rows, n_free, dead, dead_frac

def rank_nullity_stats(rows: List[Dict[int,int]], n_free: int, p: int):
    rank, _ = rank_from_row_dicts(rows, p)
    rank = int(rank)
    nullity = max(0, n_free - rank)
    return rank, nullity

def wiggle_trials(rows: List[Dict[int,int]], n_free: int, p: int, freeze_prefix: int, trials: int = 10, subset: int = 32):
    free_cols = list(range(freeze_prefix + 1, freeze_prefix + 1 + n_free))
    if not free_cols:
        return {"wiggle_rate": 0.0, "trials": trials, "subset": subset}
    k = min(subset, len(free_cols))
    successes = 0
    for _ in range(trials):
        S = set(random.sample(free_cols, k))
        sub_rows = []
        for d in rows:
            if not d:
                sub_rows.append({}); continue
            sub_rows.append({c:v for c,v in d.items() if c in S})
        rS, _ = rank_from_row_dicts(sub_rows, p)
        if rS < k:
            successes += 1
    return {"wiggle_rate": successes / max(1, trials), "trials": trials, "subset": k}
