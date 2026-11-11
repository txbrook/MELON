"""Evaluate MELON embeddings with cross-validated prediction probes."""
import pandas as pd


def evaluate_targets(embedding, targets, seed):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import balanced_accuracy_score

    x = embedding.to_numpy()
    rows = []
    for target in ["R", "U", "C", "Y"]:
        y = targets.loc[embedding.index, target].to_numpy()
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(folds.split(x, y), start=1):
            if target == "Y":
                estimator = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4,
                    learning_rate_init=1e-3, early_stopping=True, max_iter=400,
                    n_iter_no_change=20, random_state=seed + fold)
            else:
                estimator = LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed + fold)
            probe = make_pipeline(StandardScaler(), estimator)
            probe.fit(x[train], y[train])
            rows.append({"seed": seed, "target": target, "fold": fold,
                         "balanced_accuracy": balanced_accuracy_score(y[test], probe.predict(x[test]))})
    return pd.DataFrame(rows)
