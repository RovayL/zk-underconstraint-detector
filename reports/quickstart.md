# Quickstart — reproduce detector results

1) **Fit** the compact detector (PCA→GMM + logistic calibrator):
```bash
python -m zkuc.cli fit --features data/dataset.jsonl --model models/zkuc_detector.pkl
```

2) **Score** any feature JSONL (same set or a new one):
```bash
python -m zkuc.cli score-detector --features data/dataset.jsonl --model models/zkuc_detector.pkl --out reports/scores.csv
```

3) **Evaluate** (AUROC/AP + ROC/PR plots) on labeled rows:
```bash
python -m zkuc.cli eval-detector --features data/dataset.jsonl --model models/zkuc_detector.pkl \
  --report-out reports/quickstart_eval.json --plots-out reports/quickstart_plots
```

To target a non-Circom artifact (e.g., Gnark/Noir), first convert to R1CS JSON, then:
```bash
python -m zkuc.cli extract --r1cs data/gnark_demo.r1cs.json --out data/gnark_features.jsonl
python -m zkuc.cli score-detector --features data/gnark_features.jsonl --model models/zkuc_detector.pkl --out reports/gnark_scores.csv
```
