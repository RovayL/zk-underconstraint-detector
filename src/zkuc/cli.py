import json
import click
from pathlib import Path
from .core.r1cs_io import load_r1cs_json, summarize_r1cs
from .features.structural import structural_features

@click.group()
def cli():
    """zkuc command line interface"""
    pass

@cli.command(name="parse")
@click.option("--r1cs", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to snarkjs-exported R1CS JSON")
def parse_cmd(r1cs):
    """Parse and summarize an R1CS JSON."""
    r = load_r1cs_json(r1cs)
    summary = summarize_r1cs(r)
    click.echo(json.dumps(summary, indent=2))

@cli.command(name="features")
@click.option("--r1cs", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to snarkjs-exported R1CS JSON")
@click.option("--out", type=click.Path(dir_okay=False), required=False,
              help="Append features row to this JSONL file")
def features_cmd(r1cs, out):
    """Compute structural features for a circuit and optionally write to JSONL."""
    r = load_r1cs_json(r1cs)
    feats = structural_features(r)
    feats["circuit_id"] = Path(r1cs).stem
    pretty_keys = [
        "circuit_id","n_constraints","n_vars","n_inputs","n_outputs",
        "ratio_c_v","mult_share","avg_fanin","avg_fanout","deg_proxy","dup_rows_frac"
    ]
    click.echo(json.dumps({k: feats[k] for k in pretty_keys}, indent=2))

    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("a") as f:
            f.write(json.dumps(feats) + "\n")

@cli.command(name="probe")
@click.option("--r1cs", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to snarkjs-exported R1CS JSON")
@click.option("--witness", "witness_paths", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True,
              help="One or more snarkjs-exported witness JSON files")
@click.option("--pub", "pub_override", type=int, default=None, help="Override number of public inputs (default: use n_inputs from R1CS)")
@click.option("--trials", type=int, default=10, help="Number of wiggle trials (default: 10)")
@click.option("--subset", type=int, default=32, help="Subset size for wiggle test (default: 32)")
@click.option("--out", type=click.Path(dir_okay=False), required=False, help="Append probe stats row to this JSONL file")
def probe_cmd(r1cs, witness_paths, pub_override, trials, subset, out):
    from .core.witness_io import load_witness_json
    from .features.probe import jacobian_rows_dict, rank_nullity_stats, wiggle_trials
    from .core.r1cs_io import load_r1cs_json, BN254_PRIME
    import numpy as np
    r = load_r1cs_json(r1cs)
    p = BN254_PRIME
    n_inputs = pub_override if pub_override is not None else r.n_inputs
    freeze_prefix = n_inputs  # 0..n_inputs frozen (0=one)

    ranks, nullities, dead_fracs, wiggle_rates = [], [], [], []
    for wpath in witness_paths:
        w = load_witness_json(wpath, p)
        if len(w) < r.n_vars: w = w + [0]*(r.n_vars-len(w))
        elif len(w) > r.n_vars: w = w[:r.n_vars]
        rows, n_free, dead_cnt, dead_frac = jacobian_rows_dict(r, w, p, freeze_prefix)
        rank, nullity = rank_nullity_stats(rows, n_free, p)
        wr = wiggle_trials(rows, n_free, p, freeze_prefix, trials=trials, subset=subset)
        ranks.append(rank); nullities.append(nullity); dead_fracs.append(dead_frac); wiggle_rates.append(wr["wiggle_rate"])

    summary = {
        "circuit_id": Path(r1cs).stem,
        "n_inputs_frozen": int(freeze_prefix),
        "rank_mean": float(np.mean(ranks)),
        "rank_std": float(np.std(ranks)),
        "nullity_mean": float(np.mean(nullities)),
        "nullity_std": float(np.std(nullities)),
        "dead_rows_frac_mean": float(np.mean(dead_fracs)),
        "wiggle_rate_mean": float(np.mean(wiggle_rates)),
        "trials": int(trials),
        "subset": int(subset),
        "witness_count": len(witness_paths),
    }
    click.echo(json.dumps(summary, indent=2))
    if out:
        outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("a") as f: f.write(json.dumps(summary) + "\n")

def main():
    cli()

if __name__ == "__main__":
    main()
