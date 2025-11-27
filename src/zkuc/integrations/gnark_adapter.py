from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def gnark_json_to_r1cs(gnark_json: str) -> Dict[str, Any]:
    """
    Convert a gnark-exported constraint JSON (with rows of A/B/C terms) into our R1CS JSON schema.
    Expected gnark JSON shape:
      {
        "NInputs": int, "NOutputs": int, "NVars": int, "NConstraints": int, "Prime": str,
        "constraints": [ { "A":[{"Coeff":c,"Var":i}], "B":[...], "C":[...] }, ... ]
      }
    """
    o = json.loads(Path(gnark_json).read_text())

    def norm_terms(lst):
        out = []
        for t in lst:
            coeff = t.get("Coeff")
            var = t.get("Var")
            out.append({"coeff": str(coeff), "var": int(var)})
        return out

    constraints = []
    for row in o.get("constraints", []):
        constraints.append({
            "A": norm_terms(row.get("A", [])),
            "B": norm_terms(row.get("B", [])),
            "C": norm_terms(row.get("C", [])),
        })

    return {
        "nInputs": int(o.get("NInputs", 0)),
        "nOutputs": int(o.get("NOutputs", 0)),
        "nVars": int(o.get("NVars", 0)),
        "nConstraints": int(o.get("NConstraints", len(constraints))),
        "prime": str(o.get("Prime")),
        "constraints": constraints,
    }
