from __future__ import annotations
import numpy as np
from typing import List, Dict

def matvec_rows_modp(rows: List[Dict[int,int]], z: np.ndarray, p: int) -> np.ndarray:
    """Compute (Rows @ z) mod p, where rows[i] is {col: coeff}."""
    m = len(rows)
    out = np.zeros(m, dtype=object)
    for i in range(m):
        acc = 0
        row = rows[i]
        for j, c in row.items():
            acc += c * z[j]
        out[i] = acc % p
    return out

def jacobian_submatrix_dense_modp(
    A_rows: List[Dict[int,int]],
    B_rows: List[Dict[int,int]],
    C_rows: List[Dict[int,int]],
    Az: np.ndarray,
    Bz: np.ndarray,
    cols: List[int],
    p: int,
) -> np.ndarray:
    """
    Build dense J_sub = J[:, cols] over F_p, where
      J = diag(Bz) @ A + diag(Az) @ B - C .
    Returns object dtype (exact ints mod p).
    """
    m = len(A_rows)
    k = len(cols)
    J = np.zeros((m, k), dtype=object)
    for i in range(m):
        ai = A_rows[i]; bi = B_rows[i]; ci = C_rows[i]
        bz = Bz[i]; az = Az[i]
        for t, jcol in enumerate(cols):
            aij = ai.get(jcol, 0)
            bij = bi.get(jcol, 0)
            cij = ci.get(jcol, 0)
            J[i, t] = (bz * aij + az * bij - cij) % p
    return J

def jacobian_at_witness_modp(A_rows, B_rows, C_rows, z, p):
    """
    Back-compat wrapper: returns a *dense* full Jacobian J over F_p.
    New code should prefer jacobian_submatrix_dense_modp(...) for sampled columns.
    """
    Az = matvec_rows_modp(A_rows, z, p)
    Bz = matvec_rows_modp(B_rows, z, p)
    cols = list(range(len(z)))
    return jacobian_submatrix_dense_modp(A_rows, B_rows, C_rows, Az, Bz, cols, p)