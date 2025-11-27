from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

def noir_json_to_r1cs(acir_json: str) -> Dict[str, Any]:
    """
    Convert a minimal ACIR-style JSON (add/mul/const) into our R1CS JSON.
    Expected JSON shape (simplified):
    {
      "field": "<prime>",
      "public_inputs": <int>,
      "variables": <int>,
      "constraints": [
         {"op":"add", "lhs":{"var":i,"coeff":c}, "rhs":{"var":j,"coeff":d}, "out":{"var":k,"coeff":e}},
         {"op":"mul", "lhs":{"var":i,"coeff":c}, "rhs":{"var":j,"coeff":d}, "out":{"var":k,"coeff":e}},
         {"op":"const", "out":{"var":k,"coeff":c}}
      ]
    }
    This is a deliberately tiny subset to let us exercise end-to-end import; real ACIR has richer ops.
    """
    obj = json.loads(Path(acir_json).read_text())
    p = obj.get("field")
    n_vars = int(obj.get("variables", 0))
    n_inputs = int(obj.get("public_inputs", 0))
    cons: List[Dict[str, Any]] = []

    def term(t):
        return {"coeff": str(t.get("coeff", 1)), "var": int(t.get("var", 0))}

    for c in obj.get("constraints", []):
        op = c.get("op")
        if op == "add":
            A = [term(c["lhs"])]
            B = [{"coeff": "1", "var": 0}]
            C = [term(c["out"]), {"coeff": "-1", "var": c["rhs"]["var"]}]
        elif op == "mul":
            A = [term(c["lhs"])]
            B = [term(c["rhs"])]
            C = [term(c["out"])]
        elif op == "const":
            A = [{"coeff": "1", "var": 0}]
            B = [{"coeff": "1", "var": 0}]
            C = [term(c["out"])]
        else:
            # fallback: treat as equality LHS - RHS = 0 with identity
            A = [term(c.get("lhs", {"var":0,"coeff":0}))]
            B = [{"coeff": "1", "var": 0}]
            C = [term(c.get("rhs", {"var":0,"coeff":0}))]
        cons.append({"A": A, "B": B, "C": C})

    return {
        "nInputs": n_inputs,
        "nOutputs": 0,
        "nVars": n_vars,
        "nConstraints": len(cons),
        "prime": str(p),
        "constraints": cons,
    }
