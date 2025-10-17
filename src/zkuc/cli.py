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

# Dataset seeding and featurization ---
@cli.command("seed-dataset")
@click.option("--src-glob", required=True, help="Glob for source R1CS JSONs (originals)")
@click.option("--out-dir", required=True, help="Directory to write seeded R1CS JSONs")
@click.option("--per-src", default=4, show_default=True, help="Number of seeds per source per bucket (UC and control)")
@click.option("--seed", default=0, show_default=True)
@click.option("--jsonl", "jsonl_path", default="data/dataset.jsonl", show_default=True)
@click.option("--trials", default=10, show_default=True)
@click.option("--subset", default=32, show_default=True)
@click.option("--freeze-const/--no-freeze-const", default=True, show_default=True)
@click.option("--freeze-pub-inputs/--no-freeze-pub-inputs", default=True, show_default=True)
def seed_dataset_cmd(src_glob, out_dir, per_src, seed, jsonl_path, trials, subset, freeze_const, freeze_pub_inputs):
    """
    Generate UC/control seeds from originals and write a dataset JSONL with features + ground-truth labels.
    """
    import glob, random, json
    from zkuc.dataset.build import seed_from_file, SeedSpec

    rng = random.Random(seed)
    probe_cfg = dict(trials=int(trials), subset=int(subset),
                     freeze_const=bool(freeze_const), freeze_pub_inputs=bool(freeze_pub_inputs), seed=int(seed))

    # Define mutation recipes (feel free to tweak)
    uc_specs = [
        [SeedSpec("zero_cols", {"frac":0.05}, 1)],
        [SeedSpec("linearize_mult_rows", {"frac":0.2}, 1)],
        [SeedSpec("drop_rows", {"frac":0.10}, 1)],
        [SeedSpec("zero_cols", {"frac":0.02}, 1), SeedSpec("linearize_mult_rows", {"frac":0.1}, 1)],
    ]
    ctrl_specs = [
        [SeedSpec("duplicate_rows", {"frac":0.05}, 0)],
        [SeedSpec("permute_rows", {}, 0)],
        [SeedSpec("duplicate_rows", {"frac":0.03}, 0), SeedSpec("permute_rows", {}, 0)],
    ]

    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    for src in sorted(glob.glob(src_glob)):
        seed_from_file(src, out_dir, per_src, rng, uc_specs, ctrl_specs, probe_cfg, jsonl_path)

    click.echo(f"Seeded dataset written to {jsonl_path} and R1CS files under {out_dir}")

@cli.command("featurize-dir")
@click.option("--r1cs-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--jsonl", "jsonl_path", required=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--trials", default=10, show_default=True)
@click.option("--subset", default=32, show_default=True)
@click.option("--freeze-const/--no-freeze-const", default=True, show_default=True)
@click.option("--freeze-pub-inputs/--no-freeze-pub-inputs", default=True, show_default=True)
def featurize_dir_cmd(r1cs_dir, jsonl_path, seed, trials, subset, freeze_const, freeze_pub_inputs):
    """
    Featurize every *.r1cs.json in a directory and append rows to a JSONL (no ground-truth labels).
    """
    import random, json
    from pathlib import Path
    from zkuc.dataset.featurize import featurize_file
    rng = random.Random(seed)
    probe_cfg = dict(trials=int(trials), subset=int(subset),
                     freeze_const=bool(freeze_const), freeze_pub_inputs=bool(freeze_pub_inputs), seed=int(seed))
    rows = []
    for fp in sorted(Path(r1cs_dir).glob("*.r1cs.json")):
        feats = featurize_file(str(fp), rng, probe_cfg)
        rows.append({"id": fp.name, "parent_id": None, "label_uc": None, "mutations": [], "features": feats, "probe_cfg": probe_cfg})
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    click.echo(f"Wrote {len(rows)} rows to {jsonl_path}")

# Modeling commands (unsupervised + calibration + eval) ---
@cli.command("model-unsup")
@click.option("--jsonl", "jsonl_path", required=True, help="Dataset JSONL from seed-dataset")
@click.option("--model-out", default="models/pca_gmm.joblib", show_default=True)
@click.option("--n-pca", default=6, show_default=True)
@click.option("--seed", default=0, show_default=True)
def model_unsup_cmd(jsonl_path, model_out, n_pca, seed):
    """Fit PCA(d) -> GMM(k=2) on features (unsupervised)."""
    from zkuc.model.pca_gmm import dataset_from_jsonl, fit_pca_gmm
    import numpy as np
    X, feat_names, _ = dataset_from_jsonl(jsonl_path)
    model = fit_pca_gmm(X, feat_names, n_pca=n_pca, random_state=seed)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    click.echo(f"Saved PCA+GMM to {model_out} (buggy_component={model.buggy_component}, n_pca={model.n_pca})")

@cli.command("model-calibrate")
@click.option("--jsonl", "jsonl_path", required=True, help="Dataset JSONL with labels (seeded pairs)")
@click.option("--model-in", default="models/pca_gmm.joblib", show_default=True)
@click.option("--cal-out", default="models/calibrator.joblib", show_default=True)
@click.option("--n-pca-feat", default=3, show_default=True)
@click.option("--model-type", type=click.Choice(["logreg","tree"]), default="logreg", show_default=True)
def model_calibrate_cmd(jsonl_path, model_in, cal_out, n_pca_feat, model_type):
    """Fit lightweight calibrator (LogReg or small Tree) on unsupervised outputs + PCA comps."""
    import numpy as np
    from zkuc.model.pca_gmm import dataset_from_jsonl, PCAGMMModel
    from zkuc.model.calibrate import fit_calibrator
    X, feat_names, y_list = dataset_from_jsonl(jsonl_path)
    y = np.array([yy for yy in y_list if yy is not None], dtype=int)
    # keep only rows with labels
    mask = np.array([yy is not None for yy in y_list])
    X = X[mask]
    model = PCAGMMModel.load(model_in)
    cal = fit_calibrator(model, X, y, n_pca_used=n_pca_feat, model_type=model_type)
    Path(cal_out).parent.mkdir(parents=True, exist_ok=True)
    cal.save(cal_out)
    click.echo(f"Saved calibrator to {cal_out} (thr@1%={cal.threshold_fpr_1pct:.4f}, thr@5%={cal.threshold_fpr_5pct:.4f})")

@cli.command("model-eval")
@click.option("--jsonl", "jsonl_path", required=True, help="Dataset JSONL (labels required for metrics)")
@click.option("--model-in", default="models/pca_gmm.joblib", show_default=True)
@click.option("--cal-in", default="models/calibrator.joblib", show_default=True)
@click.option("--report-out", default="reports/m4_eval.json", show_default=True)
@click.option("--plots-out", default=None, help="Directory to write ROC/PR/Calibration plots (PNGs)")
@click.option("--scores-out", default=None, help="CSV path to write per-row scores (id,score,buggy_post,llr) for labeled rows")
def model_eval_cmd(jsonl_path, model_in, cal_in, report_out, plots_out, scores_out):
    """Evaluate AUROC/PR, TPR at 1% and 5% FP, and a calibration curve; write JSON report."""
    import json
    import numpy as np
    from zkuc.model.pca_gmm import dataset_from_jsonl, PCAGMMModel
    from zkuc.model.calibrate import Calibrator
    from zkuc.metrics.eval import evaluate_scores, save_plots
    import csv
    import json

    X, feat_names, y_list = dataset_from_jsonl(jsonl_path)
    mask = np.array([yy is not None for yy in y_list])
    X, y = X[mask], np.array([yy for yy in y_list if yy is not None], dtype=int)

    pga = PCAGMMModel.load(model_in)
    cal = Calibrator.load(cal_in)
    proba = cal.predict_proba(pga, X)           # calibrated UC probability
    unsup = pga.score(X)                        # {'buggy_post','llr',...}
    report = evaluate_scores(y, proba)
    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    click.echo(json.dumps(report, indent=2))
    if plots_out:
        save_plots(y, proba, plots_out)
        click.echo(f"Saved plots to {plots_out}")
    if scores_out:
        Path(scores_out).parent.mkdir(parents=True, exist_ok=True)
        # recover the IDs in the same order as dataset_from_jsonl()
        ids_all = []
        with open(jsonl_path, "r") as fin:
            for line in fin:
                if line.strip():
                    ids_all.append(json.loads(line).get("id"))
        ids_lab = [ids_all[i] for i, m in enumerate(mask) if m]
        with open(scores_out, "w", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["id", "score", "buggy_post", "llr"])
            for rid, s, bp, l in zip(ids_lab, proba, unsup["buggy_post"], unsup["llr"]):
                w.writerow([rid, float(s), float(bp), float(l)])
        click.echo(f"Saved scores to {scores_out}")


def main():
    cli()

if __name__ == "__main__":
    main()
