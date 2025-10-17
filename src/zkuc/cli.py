import json, random
import click
import numpy as np
from pathlib import Path
# from .core.r1cs_io import load_r1cs_json, summarize_r1cs
# from .features.structural import structural_features
from zkuc.core.jacobian import matvec_rows_modp, jacobian_submatrix_dense_modp
from zkuc.core.rank_modp import algebraic_rank_modp, structural_rank
from zkuc.core.witness_io import load_witness_json, assemble_full_z
from zkuc.core.r1cs_io import load_r1cs_json

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

@click.command("probe")
@click.option("--r1cs", required=True, type=click.Path(exists=True))
@click.option("--witness", "witness_files", multiple=True, type=click.Path(exists=True),
              help="One or more witness JSON files (repeat flag).")
@click.option("--witness-dir", type=click.Path(exists=True), help="Directory containing *.witness.json")
@click.option("--trials", default=10, show_default=True)
@click.option("--subset", default=32, show_default=True, help="Number of columns to sample from unfrozen set")
@click.option("--seed", default=0, show_default=True)
@click.option("--freeze-const/--no-freeze-const", default=True, show_default=True)
@click.option("--freeze-pub-inputs/--no-freeze-pub-inputs", default=True, show_default=True)
@click.option("--freeze-cols", default="", show_default=False, help="Comma-separated var indices to freeze")
@click.option("--rank-mode", type=click.Choice(["algebraic", "structural"]), default="algebraic", show_default=True)
@click.option("--out", "out_path", type=click.Path(), required=False)
def probe_cmd(r1cs, witness_files, witness_dir, trials, subset, seed,
              freeze_const, freeze_pub_inputs, freeze_cols, rank_mode, out_path):
    R = load_r1cs_json(r1cs)
    p = R.prime

    # collect witness files
    all_wfiles = list(witness_files)
    if witness_dir:
        for fp in sorted(Path(witness_dir).glob("*.witness.json")):
            all_wfiles.append(str(fp))
    if not all_wfiles:
        raise click.UsageError("Provide at least one --witness or a --witness-dir")

    rng = random.Random(seed)

    # build freeze set
    frozen = set()
    if freeze_const and R.n_vars > 0:
        frozen.add(0)  # constant-1 wire
    if freeze_pub_inputs and R.n_inputs > 0:
        # Heuristic: freeze first n_inputs variables after constant (works for your tiny_add; for generality use .sym)
        for v in range(1, min(1 + R.n_inputs, R.n_vars)):
            frozen.add(v)
    if freeze_cols.strip():
        for tok in freeze_cols.split(","):
            if tok.strip():
                frozen.add(int(tok.strip()))

    # unfrozen col indices
    unfrozen = [j for j in range(R.n_vars) if j not in frozen]
    if not unfrozen:
        raise click.ClickException("No unfrozen columns remain after freeze policy.")

    # gather stats across witnesses
    rank_vals, nullity_vals, wiggle_rates, dead_rows_frac_vals = [], [], [], []
    n_samples_total = 0

    for wf in all_wfiles:
        wit_vals = load_witness_json(wf, p)
        z = assemble_full_z(wit_vals, R.var_map, R.n_vars, p)

        # validate witness: (A z) ⊙ (B z) == (C z)
        Az = matvec_rows_modp(R.A_rows, z, p)
        Bz = matvec_rows_modp(R.B_rows, z, p)
        Cz = matvec_rows_modp(R.C_rows, z, p)
        residual = (Az * Bz - Cz) % p
        if int(np.max(residual)) != 0:
            bad = int(np.argmax(residual))
            raise click.ClickException(
                f"Witness does not satisfy R1CS (first failing row {bad}, value {int(residual[bad])})"
            )

        for _ in range(trials):
            k = min(subset, len(unfrozen))
            cols = rng.sample(unfrozen, k) if k > 0 else []
            cols.sort()

            # Build dense J[:, cols] directly over F_p
            J_sub = jacobian_submatrix_dense_modp(
                R.A_rows, R.B_rows, R.C_rows, Az, Bz, cols, p
            )

            # rank
            if rank_mode == "algebraic":
                r = algebraic_rank_modp(J_sub, p)
            else:
                r = structural_rank(J_sub, 2)

            n_rows, n_cols = J_sub.shape
            rank_vals.append(r)
            nullity_vals.append(max(n_cols - r, 0))

            # wiggle: fraction of columns with any nonzero
            col_nnz = (np.count_nonzero(J_sub, axis=0))
            wiggle = 0.0 if n_cols == 0 else float((col_nnz > 0).sum()) / float(n_cols)
            wiggle_rates.append(wiggle)

            # dead rows: rows that are all zeros
            row_nnz = (np.count_nonzero(J_sub, axis=1))
            dead_rows = float((row_nnz == 0).sum()) / float(n_rows if n_rows else 1)
            dead_rows_frac_vals.append(dead_rows)

            n_samples_total += 1

    def meanstd(xs):
        if not xs:
            return 0.0, 0.0
        x = np.array(xs, dtype=float)
        return float(x.mean()), float(x.std())

    out = {
        "circuit_id": Path(r1cs).stem,
        "n_inputs_frozen": (R.n_inputs if freeze_pub_inputs else 0),
        "rank_mean": meanstd(rank_vals)[0],
        "rank_std": meanstd(rank_vals)[1],
        "nullity_mean": meanstd(nullity_vals)[0],
        "nullity_std": meanstd(nullity_vals)[1],
        "dead_rows_frac_mean": meanstd(dead_rows_frac_vals)[0],
        "wiggle_rate_mean": meanstd(wiggle_rates)[0],
        "trials": trials,
        "subset": subset,
        "witness_count": len(all_wfiles),
        "seed": seed,
        "freeze_const": freeze_const,
        "freeze_pub_inputs": freeze_pub_inputs,
        "extra_frozen": sorted([int(x) for x in frozen if x not in (set(range(1, 1+R.n_inputs)) | {0})]),
        "rank_mode": rank_mode,
    }

    s = json.dumps(out, indent=2)
    if out_path:
        Path(out_path).write_text(s)
    print(s)


def main():
    cli()

if __name__ == "__main__":
    main()
