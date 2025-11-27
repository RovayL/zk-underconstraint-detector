# Metric definitions

- **AUROC (↑)**: Area under the ROC curve; probability a randomly chosen positive scores above a randomly chosen negative (Bradley, 1997).
- **Average Precision (↑)**: Area under the precision–recall curve; summarizes precision at all recall levels, robust to class imbalance (Davis & Goadrich, 2006).
- **TPR@1% / 5% FP (↑)**: True positive rate when the false positive rate is capped at 1% or 5%; derived by selecting the score threshold at the target FP.
- **Accuracy (↑)**: Fraction of correct predictions.
- **Precision / Recall / F1 (↑)**: Standard classification metrics; F1 is the harmonic mean of precision and recall.
- **Brier score (↓)**: Mean squared error between predicted probabilities and true labels; lower is better (Brier, 1950).
- **Calibration curve**: Reliability diagram plotting predicted probability vs. empirical frequency; flat diagonal indicates perfect calibration (Niculescu-Mizil & Caruana, 2005).

References: [Bradley1997], [DavisGoadrich2006], [Brier1950], [NiculescuMizilCaruana2005].
