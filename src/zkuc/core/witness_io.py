from __future__ import annotations
import json
from pathlib import Path
from typing import List

def load_witness_json(path, p: int) -> List[int]:
    """
    Reads a witness JSON from:
      • snarkjs:          ["1","..."]
      • zkuc-style:       {"values": ["1","..."]}
      • alt compatible:   {"witness": ["1","..."]}

    Returns a list of ints reduced mod p.
    """
    obj = json.loads(Path(path).read_text())

    if isinstance(obj, list):
        vals = obj
    else:
        vals = obj.get("values") or obj.get("witness") or obj.get("data") or []
    if not isinstance(vals, list):
        raise ValueError("Witness JSON does not contain an array of values")

    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(v % p)
        elif isinstance(v, str):
            s = v.strip()
            if s.startswith(("0x", "0X")):
                vv = int(s, 16)
            else:
                vv = int(s)  # decimal
            out.append(vv % p)
        else:
            raise ValueError(f"Unsupported witness value type: {type(v)}")
    return out
