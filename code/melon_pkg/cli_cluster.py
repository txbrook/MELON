import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

import igraph as ig
import networkx as nx

try:
    import community as community_louvain

    _HAS_LOUVAIN = True
except Exception:
    _HAS_LOUVAIN = False

try:
    import leidenalg as la

    _HAS_LEIDEN = True
except Exception:
    _HAS_LEIDEN = False


def build_run_invocation_metadata() -> dict[str, object]:
    if getattr(sys, "orig_argv", None):
        argv = [str(arg) for arg in sys.orig_argv]
    else:
        argv = [sys.executable, str(Path(__file__).resolve()), *[str(arg) for arg in sys.argv[1:]]]

    return {
        "argv": argv,
        "command_line": shlex.join(argv),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "script_path": str(Path(__file__).resolve()),
    }


def save_run_invocation(output_dir: str | Path, invocation: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command_path = output_dir / "cluster_melon_command.sh"
    command_path.write_text(f"{invocation['command_line']}\n", encoding="utf-8")

    metadata_path = output_dir / "cluster_melon_invocation.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(invocation, f, indent=2, sort_keys=True)

    print(f"Saved clustering command to {command_path}")


def read_embedding(melon_dir: str, embedding_csv: str | None) -> pd.DataFrame:
    if embedding_csv:
        if not os.path.exists(embedding_csv):
            raise FileNotFoundError(f"Embedding file not found: {embedding_csv}")
        return pd.read_csv(embedding_csv, index_col=0)

    candidates = [
        os.path.join(melon_dir, "melon_embeddings.csv"),
        os.path.join(melon_dir, "latent_melon.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0)
    raise FileNotFoundError(
        f"No embedding CSV found in {melon_dir}. "
        "Expected one of: melon_embeddings.csv, latent_melon.csv"
    )


def walktrap_clustering(X, n_clusters, n_neighbors=10):
    n_obs = int(X.shape[0])
    if n_obs <= 1:
        return np.zeros(n_obs, dtype=int)
    k = min(int(n_neighbors), n_obs - 1)
    A = kneighbors_graph(X, n_neighbors=k, include_self=False, mode="connectivity")
    A = 0.5 * (A + A.T)
    s, t = A.nonzero()
    w = A[s, t].A1
    g = ig.Graph(n=n_obs, edges=list(zip(s.tolist(), t.tolist())), directed=False)
    if w.size > 0:
        g.es["weight"] = w.tolist()
    return np.array(g.community_walktrap(weights="weight").as_clustering(n=n_clusters).membership, dtype=int)


def louvain_clustering(X, n_clusters, n_neighbors=10, seed=2025):
    if not _HAS_LOUVAIN:
        raise RuntimeError("python-louvain ('community') is not available.")

    n_obs = int(X.shape[0])
    if n_obs <= 1:
        return np.zeros(n_obs, dtype=int)
    k = min(int(n_neighbors), n_obs - 1)
    A = kneighbors_graph(X, n_neighbors=k, include_self=False, mode="connectivity")
    A = 0.5 * (A + A.T)
    G = nx.Graph()
    s, t = A.nonzero()
    w = A[s, t].A1
    G.add_nodes_from(range(n_obs))
    for u, v, ww in zip(s, t, w):
        G.add_edge(int(u), int(v), weight=float(ww))

    part = community_louvain.best_partition(G, random_state=seed, weight="weight")
    labels = np.array([part[i] for i in range(n_obs)], dtype=int)
    uniq = np.unique(labels)
    if len(uniq) > n_clusters:
        counts = sorted(((lab, (labels == lab).sum()) for lab in uniq), key=lambda x: x[1])
        target = counts[-1][0]
        for lab, _ in counts[: len(uniq) - n_clusters]:
            labels[labels == lab] = target
    elif len(uniq) < n_clusters:
        spec = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=seed)
        labels = spec.fit_predict(A.toarray())
    return labels


def leiden_clustering(X, n_clusters, n_neighbors=10, seed=2025, resolution=1.0):
    """
    Build a kNN graph on X, run Leiden, then coerce to exactly n_clusters:
    - if too many, merge smallest communities into the largest until k
    - if too few, fallback to spectral to split to k
    """
    if not _HAS_LEIDEN:
        raise RuntimeError("leidenalg is not available.")

    n_obs = int(X.shape[0])
    if n_obs <= 1:
        return np.zeros(n_obs, dtype=int)

    k = min(int(n_neighbors), n_obs - 1)
    A = kneighbors_graph(X, n_neighbors=k, include_self=False, mode="connectivity")
    A = 0.5 * (A + A.T)
    s, t = A.nonzero()
    w = A[s, t].A1
    g = ig.Graph(n=n_obs, edges=list(zip(s.tolist(), t.tolist())), directed=False)
    if w.size > 0:
        g.es["weight"] = w.tolist()

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(part.membership, dtype=int)

    uniq = np.unique(labels)
    if len(uniq) > n_clusters:
        counts = sorted(((lab, (labels == lab).sum()) for lab in uniq), key=lambda x: x[1])
        target = counts[-1][0]
        for lab, _ in counts[: len(uniq) - n_clusters]:
            labels[labels == lab] = target
    elif len(uniq) < n_clusters:
        spec = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=seed)
        labels = spec.fit_predict(A.toarray())
    return labels


def spectral_clustering(X, n_clusters, seed=2025):
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=seed,
    ).fit_predict(X)


def gmm_clustering(X, n_clusters, seed=2025):
    return GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(X)


def kmeans_clustering(X, n_clusters, seed=2025):
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)


def save_cluster_csv(output_dir: str, method: str, n_clusters: int, index, labels):
    key = f"{method}_{n_clusters}"
    path = os.path.join(output_dir, f"clusters_{method}_k{n_clusters}.csv")
    pd.DataFrame({key: labels}, index=index).to_csv(path)
    return path, key


def main():
    parser = argparse.ArgumentParser(description="Cluster MELON embeddings without plotting/ARI.")
    parser.add_argument("--melon_dir", type=str, required=True, help="Directory containing MELON outputs.")
    parser.add_argument(
        "--embedding_csv",
        type=str,
        default=None,
        help="Optional explicit embedding CSV path. Defaults to melon_embeddings.csv/latent_melon.csv in melon_dir.",
    )
    parser.add_argument("--target_k", type=int, required=True, help="Desired number of clusters.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--methods",
        type=str,
        default="spectral,gmm,kmeans,walktrap,louvain,leiden",
        help="Comma-separated methods: spectral,gmm,kmeans,walktrap,louvain,leiden",
    )
    parser.add_argument(
        "--primary_method",
        type=str,
        default="leiden",
        help="Method used to write pipeline-compatible clusters_k{target_k}.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.melon_dir, exist_ok=True)
    invocation = build_run_invocation_metadata()
    save_run_invocation(args.melon_dir, invocation)
    emb_df = read_embedding(args.melon_dir, args.embedding_csv)
    X = emb_df.values
    index = emb_df.index

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    supported = {"spectral", "gmm", "kmeans", "walktrap", "louvain", "leiden"}
    invalid = [m for m in methods if m not in supported]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Supported: {sorted(supported)}")

    results = {}
    for m in methods:
        try:
            if m == "spectral":
                y = spectral_clustering(X, args.target_k, seed=args.seed)
            elif m == "gmm":
                y = gmm_clustering(X, args.target_k, seed=args.seed)
            elif m == "kmeans":
                y = kmeans_clustering(X, args.target_k, seed=args.seed)
            elif m == "walktrap":
                y = walktrap_clustering(X, args.target_k, n_neighbors=args.n_neighbors)
            elif m == "louvain":
                y = louvain_clustering(X, args.target_k, n_neighbors=args.n_neighbors, seed=args.seed)
            elif m == "leiden":
                y = leiden_clustering(
                    X,
                    args.target_k,
                    n_neighbors=args.n_neighbors,
                    seed=args.seed,
                    resolution=args.resolution,
                )
            else:
                continue
            y = np.asarray(y, dtype=int)
            results[m] = y
            out_path, key = save_cluster_csv(args.melon_dir, m, args.target_k, index, y)
            print(f"Saved {m} clusters ({len(np.unique(y))} unique) to {out_path} [{key}]")
        except Exception as exc:
            print(f"Skipping {m}: {exc}")

    if not results:
        raise RuntimeError("No clustering method succeeded.")

    primary = args.primary_method.strip().lower()
    if primary not in results:
        for fallback in ("leiden", "louvain", "walktrap", "spectral", "gmm"):
            if fallback in results:
                primary = fallback
                break

    final_labels = results[primary]
    final_key = f"{primary}_{args.target_k}"
    pipeline_out = os.path.join(args.melon_dir, f"clusters_k{args.target_k}.csv")
    pd.DataFrame({final_key: final_labels}, index=index).to_csv(pipeline_out)
    print(f"Saved primary clusters to {pipeline_out} using method='{primary}'")


if __name__ == "__main__":
    main()
