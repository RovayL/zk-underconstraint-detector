from __future__ import annotations
import json, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from pathlib import Path

@dataclass
class R1CSRows:
    A_rows: List[Dict[int,int]]
    B_rows: List[Dict[int,int]]
    C_rows: List[Dict[int,int]]
    n_constraints: int
    n_vars: int
    n_inputs: int
    n_outputs: int
    prime: int
    var_map: List[int] | None

# ---- helpers: IO in the dict-based snarkjs JSON style ----
def rows_to_snarkjs_json(r: R1CSRows) -> dict:
    def dictify_row(row: Dict[int,int]) -> Dict[str,str]:
        # snarkjs-export JSON sometimes encodes constraints as dict var->coeff (strings)
        return {str(k): str(v) for k, v in row.items()}
    constraints = []
    for i in range(r.n_constraints):
        constraints.append([
            dictify_row(r.A_rows[i]),
            dictify_row(r.B_rows[i]),
            dictify_row(r.C_rows[i]),
        ])
    return {
        "prime": str(r.prime),
        "nConstraints": r.n_constraints,
        "nVars": r.n_vars,
        "nInputs": r.n_inputs,
        "nOutputs": r.n_outputs,
        "constraints": constraints,
        "map": r.var_map,
    }

def clone_rows(r: R1CSRows) -> R1CSRows:
    def deep_copy(lst: List[Dict[int,int]]): return [dict(x) for x in lst]
    return R1CSRows(
        A_rows=deep_copy(r.A_rows), B_rows=deep_copy(r.B_rows), C_rows=deep_copy(r.C_rows),
        n_constraints=r.n_constraints, n_vars=r.n_vars,
        n_inputs=r.n_inputs, n_outputs=r.n_outputs, prime=r.prime, var_map=(list(r.var_map) if r.var_map else None)
    )

# ---- mutators producing under-constraint variants ----
def drop_rows(r: R1CSRows, frac: float, rng: random.Random) -> Tuple[R1CSRows, dict]:
    m = r.n_constraints
    k = max(1, int(frac * m))
    idx = list(range(m))
    rng.shuffle(idx)
    keep = sorted(idx[k:])
    r2 = clone_rows(r)
    r2.A_rows = [r2.A_rows[i] for i in keep]
    r2.B_rows = [r2.B_rows[i] for i in keep]
    r2.C_rows = [r2.C_rows[i] for i in keep]
    r2.n_constraints = len(keep)
    return r2, {"type": "drop_rows", "params": {"frac": frac, "dropped": k}}

def zero_cols(r: R1CSRows, frac: float, rng: random.Random) -> Tuple[R1CSRows, dict]:
    n = r.n_vars
    k = max(1, int(frac * n))
    cols = list(range(n))
    rng.shuffle(cols)
    cols = set(cols[:k])
    r2 = clone_rows(r)
    for i in range(r2.n_constraints):
        for d in (r2.A_rows[i], r2.B_rows[i], r2.C_rows[i]):
            for c in list(d.keys()):
                if c in cols:
                    d.pop(c, None)
    return r2, {"type": "zero_cols", "params": {"frac": frac, "cols": sorted(list(cols))}}

def linearize_mult_rows(r: R1CSRows, frac: float, rng: random.Random) -> Tuple[R1CSRows, dict]:
    # for constraints with both A and B non-empty, clear one side to make it linear
    ids = [i for i in range(r.n_constraints) if r.A_rows[i] and r.B_rows[i]]
    k = max(1, int(frac * len(ids))) if ids else 0
    rng.shuffle(ids)
    chosen = ids[:k]
    r2 = clone_rows(r)
    for i in chosen:
        if rng.random() < 0.5:
            r2.B_rows[i].clear()
        else:
            r2.A_rows[i].clear()
    return r2, {"type": "linearize_mult_rows", "params": {"frac": frac, "rows": chosen}}

# ---- controls (not UC) ----
def duplicate_rows(r: R1CSRows, frac: float, rng: random.Random) -> Tuple[R1CSRows, dict]:
    m = r.n_constraints
    k = max(1, int(frac * m))
    idx = list(range(m))
    rng.shuffle(idx)
    take = idx[:k]
    r2 = clone_rows(r)
    for i in take:
        r2.A_rows.append(dict(r.A_rows[i]))
        r2.B_rows.append(dict(r.B_rows[i]))
        r2.C_rows.append(dict(r.C_rows[i]))
    r2.n_constraints = len(r2.A_rows)
    return r2, {"type": "duplicate_rows", "params": {"frac": frac, "duped": take}}

def permute_rows(r: R1CSRows, rng: random.Random) -> Tuple[R1CSRows, dict]:
    order = list(range(r.n_constraints))
    rng.shuffle(order)
    r2 = clone_rows(r)
    r2.A_rows = [r2.A_rows[i] for i in order]
    r2.B_rows = [r2.B_rows[i] for i in order]
    r2.C_rows = [r2.C_rows[i] for i in order]
    return r2, {"type": "permute_rows", "params": {"perm": order}}
