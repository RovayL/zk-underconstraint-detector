from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Set

def modp(x: int, p: int) -> int:
    r = x % p
    return r if r >= 0 else r + p

def inv_modp(a: int, p: int) -> int:
    a = a % p
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 mod p")
    return pow(a, p - 2, p)

def rank_from_row_dicts(rows: Iterable[Dict[int,int]], p: int) -> Tuple[int, dict]:
    """
    Sparse modular Gaussian elimination on a list of row dicts: {col: coeff_mod_p}.
    Returns (rank, pivots) where pivots maps pivot_col -> normalized_row_dict (pivot coeff = 1).
    """
    pivots: dict[int, Dict[int,int]] = {}
    rank = 0
    for row in rows:
        r = dict(row)
        for pc in sorted(pivots.keys()):
            if not r:
                break
            coeff = r.get(pc, 0)
            if coeff == 0:
                continue
            piv = pivots[pc]
            for c, pv in list(piv.items()):
                if c == pc:
                    continue
                r[c] = (r.get(c, 0) - coeff * pv) % p
                if r[c] == 0:
                    del r[c]
            if pc in r:
                del r[pc]
        if not r:
            continue
        pivot_col = min(r.keys())
        pivot_val = r.pop(pivot_col, 0) % p
        if pivot_val == 0:
            continue
        inv = inv_modp(pivot_val, p)
        r_norm = {c: (v * inv) % p for c, v in r.items() if v % p != 0}
        pivots[pivot_col] = r_norm
        rank += 1
    return rank, pivots
