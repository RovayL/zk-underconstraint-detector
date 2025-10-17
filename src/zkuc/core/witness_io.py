from __future__ import annotations
import json
from pathlib import Path
from typing import List
import numpy as np

def load_witness_json(path, p: int) -> List[int]:
    """
    Accept:
      • snarkjs: ["1","..."]
      • zkuc:    {"values":[...]}
      • alt:     {"witness":[...]} / {"data":[...]}
    Return list[int] reduced mod p.
    """
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, list):
        vals = obj
    else:
        vals = obj.get("values") or obj.get("witness") or obj.get("data") or []
    if not isinstance(vals, list):
        raise ValueError("Witness JSON does not contain an array")
    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(v % p)
        elif isinstance(v, str):
            s = v.strip()
            vv = int(s, 16) if s.startswith(("0x", "0X")) else int(s)
            out.append(vv % p)
        else:
            raise ValueError(f"Unsupported witness value type: {type(v)}")
    return out

def assemble_full_z(wit_vals: List[int], var_map: List[int] | None, n_vars: int, p: int) -> np.ndarray:
    """
    Build z (length n_vars) from witness values and optional var mapping.
    """
    z = np.zeros(n_vars, dtype=object)
    if var_map and len(var_map) == len(wit_vals):
        for widx, var in enumerate(var_map):
            if var < n_vars:
                z[var] = wit_vals[widx] % p
    else:
        # assume identity mapping
        lim = min(n_vars, len(wit_vals))
        z[:lim] = np.array(wit_vals[:lim], dtype=object) % p
    return z
