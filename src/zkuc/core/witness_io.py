from __future__ import annotations
import json
from pathlib import Path
from typing import List

def load_witness_json(path, p: int) -> List[int]:
    """
    Reads a snarkjs‑exported witness JSON (`snarkjs wtns export json X.wtns X.json`).
    Returns a list of ints reduced mod p.
    """
    obj = json.loads(Path(path).read_text())
    vals = obj.get("values") or obj.get("witness") or []
    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(v % p)
        elif isinstance(v, str):
            vv = int(v, 0)  # detects 0x hex or decimal
            out.append(vv % p)
        else:
            raise ValueError(f"Unsupported witness value type: {type(v)}")
    return out
