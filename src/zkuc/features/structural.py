from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy.sparse import csr_matrix
from ..core.r1cs_io import R1CS

def _constraint_variable_sets(A: csr_matrix, B: csr_matrix, C: csr_matrix):
    m, n = A.shape
    for mat in (A, B, C):
        mat.sort_indices()
    vars_per_row = []
    for i in range(m):
        s = set(A.indices[A.indptr[i]:A.indptr[i+1]])
        s.update(B.indices[B.indptr[i]:B.indptr[i+1]])
        s.update(C.indices[C.indptr[i]:C.indptr[i+1]])
        vars_per_row.append(s)
    return vars_per_row

def _variable_constraint_counts(vars_per_row, n_vars: int):
    counts = np.zeros(n_vars, dtype=int)
    for s in vars_per_row:
        for v in s:
            if v < n_vars:
                counts[v] += 1
    return counts

def structural_features(r: R1CS) -> Dict[str, Any]:
    A, B, C = r.A, r.B, r.C
    m, n = A.shape
    vars_per_row = _constraint_variable_sets(A, B, C)
    var_counts = _variable_constraint_counts(vars_per_row, n_vars=n)
    var_counts_wo0 = var_counts[1:] if n > 1 else var_counts

    mult_mask = (A.getnnz(axis=1) > 0) & (B.getnnz(axis=1) > 0)
    mult_share = float(np.mean(mult_mask)) if m > 0 else 0.0

    fanins = np.array([len(s) for s in vars_per_row], dtype=float)
    avg_fanin = float(np.mean(fanins)) if fanins.size else 0.0

    nonzero_vars = var_counts_wo0[var_counts_wo0 > 0]
    avg_fanout = float(np.mean(nonzero_vars)) if nonzero_vars.size else 0.0

    A_nz = A.getnnz(axis=1)
    B_nz = B.getnnz(axis=1)
    deg_proxy = float(np.mean((A_nz * B_nz) > 1)) if m > 0 else 0.0

    def row_signature(i: int) -> tuple:
    	def seg(mat):
        	idx = mat.indices[mat.indptr[i]:mat.indptr[i+1]]
        	# support-only signature: just the sorted variable indices
        	return tuple(sorted(int(j) for j in idx))
    	return (seg(A), seg(B), seg(C))

    if m > 0:
        sigs = [row_signature(i) for i in range(m)]
        unique = len(set(sigs))
        dup_rows_frac = float(1.0 - unique / m)
    else:
        dup_rows_frac = 0.0

    ratio_c_v = float(m / r.n_vars) if r.n_vars else 0.0

    feats = {
        "circuit_id": "unknown",
        "n_constraints": int(r.n_constraints),
        "n_vars": int(r.n_vars),
        "n_inputs": int(r.n_inputs),
        "n_outputs": int(r.n_outputs),
        "ratio_c_v": ratio_c_v,
        "mult_share": mult_share,
        "avg_fanin": avg_fanin,
        "avg_fanout": avg_fanout,
        "deg_proxy": deg_proxy,
        "dup_rows_frac": dup_rows_frac,
        "has_var0_one": int(var_counts[0] > 0) if r.n_vars > 0 else 0
    }
    return feats
