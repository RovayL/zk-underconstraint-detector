from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def merge_circomspect(zkuc_jsonl: str, spect_json: str, out_csv: str):
    """
    Join zkuc dataset rows with circomspect rule flags for quick comparison.
    spect_json expected format: [{"id": "...", "rule": "R1", "flag": true}, ...]
    """
    zk_rows = [json.loads(line) for line in Path(zkuc_jsonl).read_text().splitlines() if line.strip()]
    df_zk = pd.DataFrame([{
        "id": r.get("id") or r.get("features", {}).get("circuit_id"),
        "label": r.get("label_uc"),
        "rank_mean": r.get("features", {}).get("rank_mean"),
        "score_proxy": r.get("features", {}).get("rank_mean"),  # placeholder for model score if absent
    } for r in zk_rows])

    spect = json.loads(Path(spect_json).read_text())
    df_sp = pd.DataFrame(spect)
    if not df_sp.empty and "flag" in df_sp.columns and "rule" in df_sp.columns:
        df_rules = df_sp.pivot_table(index="id", columns="rule", values="flag", aggfunc="max").reset_index()
    else:
        df_rules = pd.DataFrame(columns=["id"])

    df = df_zk.merge(df_rules, on="id", how="outer")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
