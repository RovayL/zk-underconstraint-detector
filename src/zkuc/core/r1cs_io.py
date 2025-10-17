from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

BN254_PRIME = int("21888242871839275222246405745257275088548364400416034343698204186575808495617")

@dataclass
class Term:
    coeff: int
    var: int

@dataclass
class R1CS:
    A: csr_matrix
    B: csr_matrix
    C: csr_matrix
    n_constraints: int
    n_vars: int
    n_inputs: int
    n_outputs: int
    prime: int
    raw_constraints: Any

def _parse_term(t: Dict[str, Any]) -> Term:
    coeff = int(t.get("coeff", 0))
    var = int(t.get("var", 0))
    return Term(coeff=coeff, var=var)

def _normalize_constraint_entry(entry) -> List[Term]:
    out = []
    for t in entry:
        if isinstance(t, dict):
            out.append(_parse_term(t))
        elif isinstance(t, list) and len(t) == 2:
            out.append(Term(int(t[0]), int(t[1])))
        else:
            raise ValueError(f"Unrecognized term format: {t}")
    return out

def _constraints_from_json(obj) -> List[Tuple[List[Term], List[Term], List[Term]]]:
    cons = obj.get("constraints")
    if cons is None:
        raise ValueError("R1CS JSON missing 'constraints' field")

    out = []
    for i, c in enumerate(cons):
        if isinstance(c, list) and len(c) == 3:
            A_raw, B_raw, C_raw = c
        elif isinstance(c, dict) and all(k in c for k in ("A", "B", "C")):
            A_raw, B_raw, C_raw = c["A"], c["B"], c["C"]
        else:
            raise ValueError(f"Constraint {i} has unexpected format: {type(c)}")
        A_terms = _normalize_constraint_entry(A_raw)
        B_terms = _normalize_constraint_entry(B_raw)
        C_terms = _normalize_constraint_entry(C_raw)
        out.append((A_terms, B_terms, C_terms))
    return out

def _nz_mod_p(coeff: int, p: int) -> int:
    # 1 if non-zero modulo p, else 0 (works for negative coeffs too)
    return 1 if (coeff % p) != 0 else 0

def _build_sparse(constraints, n_rows, n_cols):
    import numpy as np
    from scipy.sparse import csr_matrix

    # --- A ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in A_terms:
            if t.var < n_cols:
                v = _nz_mod_p(t.coeff, BN254_PRIME)
                if v:  # only store nonzeros
                    data.append(1)
                    rows.append(i)
                    cols.append(t.var)
    A = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    # --- B ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in B_terms:
            if t.var < n_cols:
                v = _nz_mod_p(t.coeff, BN254_PRIME)
                if v:
                    data.append(1)
                    rows.append(i)
                    cols.append(t.var)
    B = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    # --- C ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in C_terms:
            if t.var < n_cols:
                v = _nz_mod_p(t.coeff, BN254_PRIME)
                if v:
                    data.append(1)
                    rows.append(i)
                    cols.append(t.var)
    C = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    return A, B, C


def load_r1cs_json(path: str | Path) -> R1CS:
    obj = json.loads(Path(path).read_text())
    n_constraints = int(obj.get("nConstraints") or obj.get("constraintsLen") or len(obj.get("constraints", [])))
    n_vars = int(obj.get("nVars") or obj.get("nWitness") or obj.get("variables", 0))
    n_inputs = int(obj.get("nInputs") or obj.get("publicInputs", 0))
    n_outputs = int(obj.get("nOutputs") or obj.get("publicOutputs", 0))
    prime_raw = obj.get("prime")
    prime = int(prime_raw) if prime_raw is not None else BN254_PRIME

    constraints = _constraints_from_json(obj)
    if n_constraints == 0:
        n_constraints = len(constraints)
    if n_vars == 0:
        max_var = 0
        for A_terms, B_terms, C_terms in constraints:
            for t in (A_terms + B_terms + C_terms):
                if t.var > max_var:
                    max_var = t.var
        n_vars = max_var + 1

    A, B, C = _build_sparse(constraints, n_rows=n_constraints, n_cols=n_vars)

    return R1CS(
        A=A, B=B, C=C,
        n_constraints=n_constraints,
        n_vars=n_vars,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        prime=prime,
        raw_constraints=obj.get("constraints"),
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
