from __future__ import annotations
import json, random
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from zkuc.core.r1cs_io import load_r1cs_json
from zkuc.seed.mutators import R1CSRows, rows_to_snarkjs_json, drop_rows, zero_cols, linearize_mult_rows, duplicate_rows, permute_rows
from zkuc.dataset.featurize import featurize_file

@dataclass
class SeedSpec:
    kind: str
    params: Dict[str, Any]
    label_uc: int  # 1 for UC, 0 for control

def _from_loader(R) -> R1CSRows:
    return R1CSRows(R.A_rows, R.B_rows, R.C_rows, R.n_constraints, R.n_vars, R.n_inputs, R.n_outputs, R.prime, R.var_map)

def apply_mutations(base: R1CSRows, specs: List[SeedSpec], rng: random.Random) -> (R1CSRows, List[dict], int):
    r = base
    trace = []
    label = 0
    for s in specs:
        if s.kind == "drop_rows":
            r, m = drop_rows(r, float(s.params.get("frac", 0.1)), rng)
        elif s.kind == "zero_cols":
            r, m = zero_cols(r, float(s.params.get("frac", 0.05)), rng)
        elif s.kind == "linearize_mult_rows":
            r, m = linearize_mult_rows(r, float(s.params.get("frac", 0.2)), rng)
        elif s.kind == "duplicate_rows":
            r, m = duplicate_rows(r, float(s.params.get("frac", 0.1)), rng)
        elif s.kind == "permute_rows":
            r, m = permute_rows(r, rng)
        else:
            raise ValueError(f"unknown mutator {s.kind}")
        trace.append({"kind": s.kind, "meta": m})
        label = max(label, s.label_uc)
    return r, trace, label

def seed_from_file(
    src_path: str, out_dir: str, per_src: int, rng: random.Random,
    uc_specs: List[List[SeedSpec]], ctrl_specs: List[List[SeedSpec]],
    probe_cfg: Dict[str,Any], jsonl_path: str
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(per_src):
        # alternate UC and control
        for is_uc, spec_list in [(1, uc_specs), (0, ctrl_specs)]:
            if not spec_list: continue
            R = load_r1cs_json(src_path)
            base = _from_loader(R)
            specs = rng.choice(spec_list)
            r_mut, trace, label = apply_mutations(base, specs, rng)
            label = 1 if is_uc else 0  # force label by bucket
            # write JSON
            obj = rows_to_snarkjs_json(r_mut)
            out_name = f"{Path(src_path).stem}.seed{str(i).zfill(3)}.{('uc' if label else 'ctrl')}.r1cs.json"
            out_path = out_dir / out_name
            out_path.write_text(json.dumps(obj))
            # featurize
            feats = featurize_file(str(out_path), rng, probe_cfg)
            rows.append({
                "id": out_name,
                "parent_id": Path(src_path).name,
                "label_uc": label,
                "mutations": [ {"type": t['kind'], **t['meta']} for t in trace ],
                "features": feats,
                "probe_cfg": probe_cfg
            })
    # append to dataset JSONL
    with open(jsonl_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
