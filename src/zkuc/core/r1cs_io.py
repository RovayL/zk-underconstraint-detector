from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict
import numpy as np
from scipy.sparse import csr_matrix

BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

@dataclass
class Term:
    coeff: int
    var: int

@dataclass
class R1CS:
    # coefficient rows as Python-int mod-p maps
    A_rows: List[Dict[int,int]]
    B_rows: List[Dict[int,int]]
    C_rows: List[Dict[int,int]]
    # 0/1 patterns, for structural features if needed
    Apat: csr_matrix
    Bpat: csr_matrix
    Cpat: csr_matrix
    n_constraints: int
    n_vars: int
    n_inputs: int
    n_outputs: int
    prime: int
    var_map: Optional[List[int]]
    raw_constraints: Any

def _normalize_constraint_entry(entry) -> List[Term]:
    if entry is None:
        return []
    if isinstance(entry, dict):
        return [Term(int(v), int(k)) for k, v in entry.items()]
    if isinstance(entry, list):
        out = []
        for t in entry:
            if isinstance(t, dict) and "coeff" in t and "var" in t:
                out.append(Term(int(t["coeff"]), int(t["var"])))
            elif isinstance(t, (list, tuple)) and len(t) == 2:
                c, v = t
                out.append(Term(int(c), int(v)))
            else:
                raise ValueError(f"Unrecognized term format element: {t!r}")
        return out
    raise ValueError(f"Unrecognized term container: {type(entry)}")

def _constraints_from_json(obj) -> List[Tuple[List[Term], List[Term], List[Term]]]:
    cons = obj.get("constraints")
    if cons is None:
        raise ValueError("R1CS JSON missing 'constraints'")
    out = []
    for i, c in enumerate(cons):
        if isinstance(c, list) and len(c) == 3:
            A_raw, B_raw, C_raw = c
        elif isinstance(c, dict) and all(k in c for k in ("A", "B", "C")):
            A_raw, B_raw, C_raw = c["A"], c["B"], c["C"]
        else:
            raise ValueError(f"Constraint {i} unexpected format: {type(c)}")
        out.append((
            _normalize_constraint_entry(A_raw),
            _normalize_constraint_entry(B_raw),
            _normalize_constraint_entry(C_raw),
        ))
    return out

def _build_rows_maps(constraints, n_rows, n_cols, p: int):
    def to_map(terms: List[Term]) -> Dict[int,int]:
        d: Dict[int,int] = {}
        for t in terms:
            if 0 <= t.var < n_cols:
                d[t.var] = (d.get(t.var, 0) + (t.coeff % p)) % p
        # strip zeros (rare after mod p)
        return {k:v for k,v in d.items() if v != 0}
    A_rows, B_rows, C_rows = [], [], []
    for (A_terms, B_terms, C_terms) in constraints:
        A_rows.append(to_map(A_terms))
        B_rows.append(to_map(B_terms))
        C_rows.append(to_map(C_terms))
    assert len(A_rows) == n_rows
    return A_rows, B_rows, C_rows

def _build_patterns(constraints, n_rows, n_cols):
    def build_from(terms_list_index):
        data, rows, cols = [], [], []
        for i, trip in enumerate(constraints):
            terms = trip[terms_list_index]
            for t in terms:
                if 0 <= t.var < n_cols:
                    data.append(1); rows.append(i); cols.append(t.var)
        M = csr_matrix((np.ones(len(data), dtype=np.int8),
                        (np.array(rows), np.array(cols))), shape=(n_rows, n_cols))
        M.data[:] = 1
        return M
    Apat = build_from(0)
    Bpat = build_from(1)
    Cpat = build_from(2)
    return Apat, Bpat, Cpat

def load_r1cs_json(path: str | Path) -> R1CS:
    obj = json.loads(Path(path).read_text())
    n_constraints = int(obj.get("nConstraints") or len(obj.get("constraints", [])))
    n_vars = int(obj.get("nVars") or obj.get("nWitness") or 0)
    n_inputs = int(obj.get("nInputs") or obj.get("nPubInputs") or 0)
    n_outputs = int(obj.get("nOutputs") or obj.get("publicOutputs") or 0)
    prime = int(obj.get("prime") or BN254_PRIME)
    var_map = obj.get("map")

    constraints = _constraints_from_json(obj)
    if n_constraints == 0:
        n_constraints = len(constraints)
    if n_vars == 0:
        maxv = 0
        for A,B,C in constraints:
            for t in (A+B+C):
                maxv = max(maxv, t.var)
        n_vars = maxv + 1

    A_rows, B_rows, C_rows = _build_rows_maps(constraints, n_constraints, n_vars, prime)
    Apat, Bpat, Cpat = _build_patterns(constraints, n_constraints, n_vars)

    return R1CS(
        A_rows=A_rows, B_rows=B_rows, C_rows=C_rows,
        Apat=Apat, Bpat=Bpat, Cpat=Cpat,
        n_constraints=n_constraints, n_vars=n_vars,
        n_inputs=n_inputs, n_outputs=n_outputs,
        prime=prime, var_map=var_map, raw_constraints=obj.get("constraints")
    )


def summarize_r1cs(r: R1CS):
    mult_rows = int(((r.A.getnnz(axis=1) > 0) & (r.B.getnnz(axis=1) > 0)).sum())
    return {
        "n_constraints": int(r.n_constraints),
        "n_vars": int(r.n_vars),
        "n_inputs": int(r.n_inputs),
        "n_outputs": int(r.n_outputs),
        "prime_bits": int(r.prime.bit_length()),
        "multiplicative_rows": mult_rows,
        "linear_rows": int(r.n_constraints - mult_rows),
    }
