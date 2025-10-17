from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

def _gauss_rank_dense_modp(A: np.ndarray, p: int) -> int:
    A = A.copy()
    m, n = A.shape
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if A[i, c] % p != 0:
                pivot = i; break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot], :] = A[[pivot, r], :]
        inv = pow(int(A[r, c]) % p, -1, p)
        A[r, :] = (A[r, :] * inv) % p
        for i in range(r + 1, m):
            if A[i, c] % p != 0:
                f = A[i, c] % p
                A[i, :] = (A[i, :] - f * A[r, :]) % p
        r += 1
        if r == m:
            break
    return r

def algebraic_rank_modp(J_sub, p: int) -> int:
    """
    Exact rank over F_p. Accepts either a dense numpy array (preferred)
    or a scipy CSR (will densify).
    """
    if hasattr(J_sub, "toarray"):
        A = J_sub.toarray().astype(object)
    else:
        A = np.array(J_sub, dtype=object, copy=True)
    return _gauss_rank_dense_modp(A, p)

def structural_rank(J_sub, p_for_calc: int = 2) -> int:
    """
    Structural rank proxy: treat nonzeros as 1 and compute rank mod 2.
    Works with dense or CSR.
    """
    if hasattr(J_sub, "toarray"):
        B = (J_sub.toarray() != 0).astype(int)
    else:
        B = (np.array(J_sub) != 0).astype(int)
    return _gauss_rank_dense_modp(B.astype(object), p_for_calc)
