"""MELON integration, clustering and intermediate-file utilities."""
from pathlib import Path
import h5py
import pandas as pd


def assay_shape(path):
    path = Path(path)
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            shape = tuple(x.shape)
        else:
            shape = tuple(int(v) for v in x.attrs["shape"])
        if "spot" in f:
            n = len(f["spot"])
            if shape[0] != n:
                shape = shape[::-1]
        return shape


def align_partition(path, embedding, column="MELON"):
    frame = pd.read_csv(path, index_col=0, converters={0: str})
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique or set(frame.index) != set(embedding.index):
        raise ValueError("Partition and embedding barcodes differ")
    labels = frame.loc[embedding.index, column]
    if labels.isna().any():
        raise ValueError("Missing domain assignments")
    return labels


def integrate_modalities(features, coordinates, config):
    """Learn a joint representation from aligned, preprocessed modality arrays."""
    import torch
    import pytorch_lightning as pl
    from _shared.engine import make_model, export_embedding, PaperAugmentations
    from _shared.publication_model.data import DualAugDataModule, SpatialOmicsAugmentations
    from _shared.publication_model.utils import build_knn_mask

    pl.seed_everything(config["training_seed"], workers=True)
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in features]
    xy = torch.as_tensor(coordinates.to_numpy(), dtype=torch.float32)
    mask, weight = build_knn_mask(xy, k=config["training_spatial_k"],
                                  include_self=True, return_weights=True)
    augmentation = SpatialOmicsAugmentations if config["historical_five_modal"] else PaperAugmentations
    aug = dict(augmentations=config.get("augmentations", ["gaussian_noise", "feature_dropout", "neighbor_mix"]),
               p=config["augment_p"])
    data = DualAugDataModule(
        mods_raw=tensors, coords=xy, neighbor_mask=mask, neighbor_weight=weight,
        batch_size=config["batch_size"], augmentA=augmentation(**aug), augmentB=augmentation(**aug),
        spot_tokenization=True, shuffle=config["shuffle_batches"])
    model = make_model([x.shape[1] for x in features], config)
    trainer = pl.Trainer(max_epochs=config["epochs"], accelerator="cpu", devices=1,
                         logger=False, enable_checkpointing=False, deterministic=True)
    trainer.fit(model, datamodule=data)
    z = export_embedding(model, tensors, xy, config["export_spatial_k"])
    return pd.DataFrame(z, index=coordinates.index)


def identify_domains(embedding, config, method="leiden"):
    """Cluster a joint embedding with the dataset's notebook parameters."""
    if method == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=config["target_clusters"], n_init=config["kmeans_n_init"],
                        random_state=config["clustering_seed"]).fit_predict(embedding)
    elif method == "leiden_pca":
        import anndata as ad
        import scanpy as sc
        import igraph as ig
        import leidenalg as la
        from scipy import sparse
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize

        x = normalize(PCA(n_components=config["clustering_pca_components"],
                          random_state=config["clustering_seed"]).fit_transform(embedding))
        a = ad.AnnData(x)
        sc.pp.neighbors(a, n_neighbors=config["clustering_neighbors"], use_rep="X",
                        metric="euclidean", random_state=config["clustering_seed"])
        edges = sparse.triu(a.obsp["connectivities"], k=1).tocoo()
        graph = ig.Graph(n=len(x), edges=list(zip(edges.row.tolist(), edges.col.tolist())), directed=False)
        result = la.find_partition(graph, la.RBConfigurationVertexPartition, weights=edges.data.tolist(),
                                  resolution_parameter=config["leiden_resolution"],
                                  seed=config["clustering_seed"], n_iterations=-1)
        labels = result.membership
    elif method == "leiden":
        from melon_pkg.cli_cluster import leiden_clustering
        labels = leiden_clustering(embedding.to_numpy(), n_clusters=config["target_clusters"],
                                    n_neighbors=config["clustering_neighbors"],
                                    seed=config["clustering_seed"], resolution=config["leiden_resolution"])
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return pd.Series(labels, index=embedding.index, name="MELON")


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


def run_configuration(features, coords, parameters):
    from melon_pkg.cli_cluster import leiden_clustering
    z = integrate_modalities(features, coords, parameters)
    labels = leiden_clustering(z.to_numpy(), n_clusters=parameters["target_clusters"], n_neighbors=parameters["clustering_neighbors"],
                                seed=parameters["clustering_seed"], resolution=parameters["leiden_resolution"])
    return z, pd.Series(labels, index=z.index, name="MELON")
