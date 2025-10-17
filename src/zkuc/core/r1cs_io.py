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

def _normalize_constraint_entry(entry) -> List[Term]:
    """
    Accepts:
      • snarkjs/circom dict form: {"var_index_str": "coeff_str", ...}
      • list of {"coeff": "...", "var": ...}
      • list of [coeff, var]
      • empty dict/list/None
    Returns: List[Term]
    """
    if entry is None:
        return []

    # snarkjs dict form
    if isinstance(entry, dict):
        out = []
        for sig, coeff in entry.items():
            out.append(Term(int(coeff), int(sig)))
        return out

    # list forms
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
    return 1 if (coeff % p) != 0 else 0

# Change signature to accept p
def _build_sparse(constraints, n_rows, n_cols, p: int):
    import numpy as np
    from scipy.sparse import csr_matrix

    # --- A ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in A_terms:
            if t.var < n_cols and _nz_mod_p(t.coeff, p):
                data.append(1); rows.append(i); cols.append(t.var)
    A = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    # --- B ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in B_terms:
            if t.var < n_cols and _nz_mod_p(t.coeff, p):
                data.append(1); rows.append(i); cols.append(t.var)
    B = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    # --- C ---
    data, rows, cols = [], [], []
    for i, (A_terms, B_terms, C_terms) in enumerate(constraints):
        for t in C_terms:
            if t.var < n_cols and _nz_mod_p(t.coeff, p):
                data.append(1); rows.append(i); cols.append(t.var)
    C = csr_matrix((np.array(data, dtype=np.int64),
                    (np.array(rows), np.array(cols))),
                   shape=(n_rows, n_cols))

    return A, B, C

def load_r1cs_json(path: str | Path) -> R1CS:
    obj = json.loads(Path(path).read_text())
    n_constraints = int(obj.get("nConstraints") or obj.get("constraintsLen") or len(obj.get("constraints", [])))
    n_vars = int(obj.get("nVars") or obj.get("nWitness") or obj.get("variables", 0))

    # Be liberal with field names:
    n_inputs = obj.get("nInputs", obj.get("nPubInputs", 0))
    n_inputs = int(n_inputs) if n_inputs is not None else 0
    n_outputs = obj.get("nOutputs", obj.get("publicOutputs", 0))
    n_outputs = int(n_outputs) if n_outputs is not None else 0

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

    # pass prime through
    A, B, C = _build_sparse(constraints, n_rows=n_constraints, n_cols=n_vars, p=prime)

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
