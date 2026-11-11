"""Dataset-specific partitions, spatial maps and benchmark metrics."""
import argparse
import json
from pathlib import Path
import anndata as ad
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from .preprocessing import unique_index


def spatial_metrics(xy, labels):
    # Remove self by index, even when several coordinates coincide.
    k = min(10, len(labels) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)
    neighbors = np.array([row[row != i][:k] for i, row in enumerate(neighbors)])
    same = labels[neighbors] == labels[:, None]
    result = {"continuity_6nn": float(same[:, :min(6, k)].mean())}
    result["PAS_10nn"] = float(((~same).sum(1) >= 6).mean()) if k == 10 else None
    return result


def main(folder):
    config = json.loads((folder / "settings.json").read_text())
    p = argparse.ArgumentParser(description=f"Cluster {config['name']}")
    p.add_argument("--embedding", type=Path, required=True)
    p.add_argument("--coords", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--truth", type=Path, help="CSV indexed by barcode, with a label column")
    p.add_argument("--truth-column", default="label")
    p.add_argument("--umap", action="store_true")
    args = p.parse_args()
    frame = pd.read_csv(args.embedding, index_col=0, converters={0: str})
    frame.index = unique_index(frame.index, "embedding")
    x = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Embedding must be finite")
    coords = pd.read_csv(args.coords, index_col=0, converters={0: str})
    coords.index = unique_index(coords.index, "coordinates")
    xy = coords.loc[frame.index, ["x", "y"]].to_numpy()
    truth = None
    if args.truth:
        labels = pd.read_csv(args.truth, index_col=0, converters={0: str})
        labels.index = unique_index(labels.index, "truth")
        truth = labels.loc[frame.index, args.truth_column]
        if truth.isna().any():
            raise ValueError("Reference labels are missing for some spots")
    args.out.mkdir(parents=True, exist_ok=True)
    seed = config["clustering_seed"]
    partitions = {}
    if config["dataset"] == "human_tonsil":
        partitions[10] = KMeans(n_clusters=10, n_init=20, random_state=seed).fit_predict(x)
    elif config["dataset"] == "mouse_brain":
        import scanpy as sc
        clustering_input = normalize(PCA(n_components=20, random_state=seed).fit_transform(x))
        a = ad.AnnData(clustering_input)
        sc.pp.neighbors(a, n_neighbors=15, use_rep="X", metric="euclidean", random_state=seed)
        edges = sparse.triu(a.obsp["connectivities"], k=1).tocoo()
        g = ig.Graph(n=len(x), edges=list(zip(edges.row.tolist(), edges.col.tolist())), directed=False)
        g.es["weight"] = edges.data.tolist()
        result = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                   resolution_parameter=config["leiden_resolution"], seed=seed, n_iterations=-1)
        y = np.asarray(result.membership)
        actual = len(np.unique(y))
        # A new model need not reproduce the archived partition exactly.
        partitions[actual] = y
        config["observed_clusters"] = actual
        config["matches_manuscript_cluster_count"] = actual == 17
        np.save(args.out / "clustering_input.npy", clustering_input)
    else:
        # The original benchmark runner: 10-NN Leiden at resolution 1, followed
        # by its documented merge/spectral target-count handling.
        from melon_pkg.cli_cluster import leiden_clustering
        for k in config["cluster_sweep"]:
            partitions[k] = leiden_clustering(x, k, n_neighbors=10, seed=seed, resolution=1.)
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, y in partitions.items():
        pd.DataFrame({"cluster": y}, index=frame.index).to_csv(args.out / f"clusters_k{k}.csv", index_label="barcode")
        metrics = dict(clusters=k, **spatial_metrics(xy, y))
        if truth is not None:
            metrics.update(ARI=adjusted_rand_score(truth, y), NMI=normalized_mutual_info_score(truth, y))
        rows.append(metrics)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xy[:, 0], xy[:, 1], c=y, s=5, cmap="tab20", linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        fig.savefig(args.out / f"spatial_k{k}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(args.out / "metrics.csv", index=False)
    (args.out / "clustering.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.umap:
        import umap
        u = umap.UMAP(n_neighbors=15, min_dist=.3, metric="euclidean", random_state=seed).fit_transform(x)
        pd.DataFrame(u, index=frame.index, columns=["UMAP1", "UMAP2"]).to_csv(args.out / "umap.csv", index_label="barcode")
