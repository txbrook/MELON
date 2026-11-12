"""Plot spatial domains, embeddings and evaluation scores."""
import json
import math
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

PAPER_COLORS = ["#7EAF99", "#F39B7D", "#A9CBCA", "#BDB2C8", "#625C70",
                "#5965B0", "#8E9BAE", "#1697A6", "#FAD471", "#CFDDB6",
                "#C98986", "#F4B8D0", "#B5A2C7", "#9F8CAE"]
METRICS = {"ari": ("ARI", "#258f9d"), "nmi": ("NMI", "#ee765e"), "spari": ("SpARI", "#edbf4c")}


def show(fig, *, crop=True):
    """Store a crisp, portable figure output directly in the notebook."""
    from IPython.display import Image, display
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight" if crop else None, facecolor="white")
    display(Image(data=buffer.getvalue()))
    plt.close(fig)


def read_indexed(path):
    frame = pd.read_csv(path, index_col=0, converters={0: str})
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate barcodes in {Path(path).name}")
    return frame


def align_embedding(embedding, labels):
    if set(embedding.index) != set(labels.index):
        raise ValueError("Embedding and labels do not contain the same barcodes")
    return embedding.loc[labels.index]


def default_colors(values):
    keys = sorted(pd.unique(values), key=lambda value: str(value))
    if len(keys) > len(PAPER_COLORS):
        colors = [to_hex(c) for c in plt.get_cmap("tab20").colors]
    else:
        colors = PAPER_COLORS
    return {str(k): colors[i % len(colors)] for i, k in enumerate(keys)}


def saved_palette(path, key="method"):
    palette = pd.read_csv(path, dtype={"cluster": str})
    return {name: dict(zip(group.cluster, group.color_hex)) for name, group in palette.groupby(key)}


def spatial_panels(frame, columns, *, colors=None, ncols=4, point_size=8, x="x", y="y", invert_y=True):
    if not np.isfinite(frame[[x, y]].to_numpy()).all():
        raise ValueError("Coordinates contain missing or nonfinite values")
    rows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(3.2 * ncols, 3.3 * rows), squeeze=False, layout="constrained")
    for ax, column in zip(axes.flat, columns):
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"Missing labels in {column}")
        palette = (colors or {}).get(column) or default_colors(values)
        shade = values.astype(str).map(palette)
        if shade.isna().any():
            raise ValueError(f"Incomplete saved color mapping for {column}")
        ax.scatter(frame[x], frame[y], c=shade, s=point_size, linewidths=0, rasterized=True)
        name = {"MELON_14_15": "MELON", "melon_domain": "MELON", "Full": "MELON full",
                "- image": "MELON without image", "- protein": "MELON without protein",
                "- RNA": "MELON without RNA"}.get(column, column)
        ax.set_title(f"{name} ({values.nunique()} groups)", fontsize=11)
        ax.set_aspect("equal")
        if invert_y:
            ax.invert_yaxis()
        ax.axis("off")
    for ax in list(axes.flat)[len(columns):]:
        ax.axis("off")
    return fig


def tonsil_colors(folder):
    mappings = json.loads((folder / "04_clusters/display_color_mappings.json").read_text())
    return {method: {str(cluster): PAPER_COLORS[int(display_id)] for cluster, display_id in values.items()}
            for method, values in mappings.items()}


def saved_umap_panels(table, palette, methods):
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.2), layout="constrained")
    for ax, method in zip(np.atleast_1d(axes), methods):
        rows = table.loc[table.method == method]
        if rows.empty or rows.barcode.duplicated().any():
            raise ValueError(f"Invalid saved UMAP rows for {method}")
        shade = rows.cluster.astype(str).map(palette[method])
        if shade.isna().any():
            raise ValueError(f"Missing UMAP colors for {method}")
        ax.scatter(rows.UMAP1, rows.UMAP2, c=shade, s=6, linewidths=0, rasterized=True)
        ax.set_title("MELON" if method == "MELON_14_15" else method.replace("_", " "), fontsize=11)
        ax.set_aspect("equal"); ax.axis("off")
    return fig


def sensitivity_panels(neighbors, augmentation):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, frame, column, label in [(axes[0], neighbors, "neighbor_k", "Spatial neighbors, k"),
                                    (axes[1], augmentation, "augment_p", "Augmentation probability, p")]:
        frame = frame.sort_values(column)
        for metric, (name, color) in METRICS.items():
            ax.plot(frame[column], frame[metric], "o-", color=color, label=name, ms=6)
        ax.set(xlabel=label, ylabel="Saved clustering score", ylim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, fontsize=9)
    return fig


def ablation_panels(frame):
    # Summaries of saved scores only. This does not rerun the cluster-number sweep.
    methods = ["full", "no_aug", "no_spatial"]
    names = ["MELON full", "MELON without augmentation", "MELON without spatial bias"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
    centers = np.arange(len(methods))
    for i, (metric, (label, color)) in enumerate(METRICS.items()):
        arrays = [frame.loc[frame.method == method, metric].dropna().to_numpy() for method in methods]
        if any(len(a) == 0 for a in arrays):
            raise ValueError("Missing saved ablation scores")
        median = np.array([np.median(a) for a in arrays])
        low = median - np.array([np.quantile(a, .25) for a in arrays])
        high = np.array([np.quantile(a, .75) for a in arrays]) - median
        ax.bar(centers + (i - 1) * .23, median, width=.23, color=color, label=label,
               yerr=np.array([low, high]), capsize=3)
    ax.set(xticks=centers, xticklabels=names, ylabel="Median saved score (IQR)", ylim=(0, 1))
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=3)
    return fig


def simulation_scores(melon):
    scores = melon.loc[((melon.target == "Y") & (melon.probe == "mlp")) |
                       ((melon.target != "Y") & (melon.probe == "linear_svm"))].copy()
    scores["method"] = "MELON"
    return scores[["target", "method", "probe", "balanced_accuracy_mean", "balanced_accuracy_std"]]


def simulation_score_panels(scores):
    if not scores.method.eq("MELON").all() or scores.target.duplicated().any():
        raise ValueError("Expected one saved MELON score per target")
    part = scores.set_index("target").loc[["R", "U", "C", "Y"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    ax.bar(np.arange(4), part.balanced_accuracy_mean, yerr=part.balanced_accuracy_std,
           color="#ee765e", capsize=3, width=.6)
    ax.set(title="MELON target prediction", xticks=np.arange(4), xticklabels=part.index,
           xlabel="Target", ylim=(0, 1.08), ylabel="Balanced accuracy (mean ± s.d.)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def marker_expression_panels(expression, style):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("marker_expression", ["#F6F1D9", "#A9CBCA", "#278F9B", "#064D59"])
    fig = plt.figure(figsize=(10, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.4])
    order = style["domain_order"]
    for row, gene in enumerate(["Col1a1", "Neurod2"]):
        ax = fig.add_subplot(grid[row, 0])
        points = ax.scatter(expression.x, expression.y, c=expression[gene], s=10,
                            linewidths=0, cmap=cmap, vmin=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(gene)
        fig.colorbar(points, ax=ax, shrink=.8, fraction=.04, pad=.025)
        ax = fig.add_subplot(grid[row, 1])
        values = [expression.loc[expression.domain == d, gene].to_numpy() for d in order]
        parts = ax.violinplot(values, positions=np.arange(len(order)), widths=.85,
                              showmedians=True, showextrema=False)
        for body, domain in zip(parts["bodies"], order):
            body.set_facecolor(style["domain_colors"][str(domain)])
            body.set_edgecolor("#777777")
            body.set_linewidth(.4)
            body.set_alpha(.8)
        parts["cmedians"].set(color="#555555", linewidth=.8)
        ax.set_xticks(range(len(order)), order)
        ax.set_ylim(0, max(1, np.ceil(expression[gene].max())))
        ax.set_ylabel("Log-normalized expression")
        ax.set_title(gene + " by domain")
        if row == 1:
            ax.set_xlabel("MELON domain")
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def domain_enrichment_panel(matrix):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("domain_enrichment", ["#F4F1DE", "#CDD9C2", "#7CB6B4", "#298E9C", "#07535E"])
    labels = {"synapse organization": "Synapse organization",
              "nervous system development": "Nervous system development",
              "neuron differentiation": "Neuron differentiation",
              "generation of neurons": "Generation of neurons",
              "multicellular organism development": "Multicellular development"}
    fig, ax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0,
                   vmax=max(float(matrix.to_numpy().max()), 1e-12))
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), [labels.get(t, t) for t in matrix.columns],
                  rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("MELON domain")
    ax.set_title("Domain enrichment")
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=.045, pad=.03, label=r"$-\log_{10}(P)$")
    return fig


def domain_network_panels(tsne, edges, style):
    import networkx as nx
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="constrained")
    colors = [style["domain_colors"][str(d)] for d in tsne.domain]
    axes[0].scatter(tsne.TSNE1, tsne.TSNE2, c=colors, s=9, alpha=.85, linewidths=0)
    centers = tsne.groupby("domain")[["TSNE1", "TSNE2"]].median()
    for domain, row in centers.iterrows():
        axes[0].text(*row, str(domain), ha="center", va="center", fontsize=10,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": .4})
    graph = nx.Graph()
    for domain, count in tsne.groupby("domain").size().items():
        graph.add_node(domain, n_spots=int(count))
    for row in edges.itertuples():
        if row.source != row.target:
            graph.add_edge(row.source, row.target, weight=row.edge_count)
    positions = {domain: tuple(row) for domain, row in centers.iterrows()}
    weights = np.array([graph.edges[e]["weight"] for e in graph.edges()])
    widths = .5 + 3 * weights / max(weights.max(initial=0), 1)
    nx.draw_networkx_edges(graph, positions, ax=axes[1], width=widths, alpha=.45, edge_color="#ADB4BA")
    nx.draw_networkx_nodes(graph, positions, ax=axes[1],
                           node_size=[100 + graph.nodes[d]["n_spots"] * .9 for d in graph],
                           node_color=[style["domain_colors"][str(d)] for d in graph],
                           edgecolors="white", linewidths=.5)
    nx.draw_networkx_labels(graph, positions, ax=axes[1], font_size=11)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    for ax, title in zip(axes, ["MELON t-SNE", "Spatial domain adjacency"]):
        ax.axis("off")
        ax.set_box_aspect(1)
        ax.set_title(title)
    return fig


def regulatory_roc_panels(curves):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), layout="constrained")
    for ax, name, title in zip(axes, ["scHiCAR", "Brain2023"], ["scHiCAR", "Brain cCRE-gene"]):
        curve, auc = curves[name]
        ax.step(curve.FPR, curve.TPR, where="post", color="#E77963", lw=1.8,
                label=f"MELON AUROC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=.8, ls="--", zorder=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title=title)
        ax.set_box_aspect(1)
        ax.legend(loc="lower right", frameon=False, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def gene_peak_panels(pairs):
    fig, axes = plt.subplots(3, len(pairs), figsize=(3.7 * len(pairs), 8.6),
                             squeeze=False, layout="constrained")
    for col, (gene, (peak, frame)) in enumerate(pairs.items()):
        chrom, start, end = peak.split("-")
        axes[0, col].set_title(f"{gene}\n{chrom}:{start}-{end}", fontsize=10)
        for row, (key, cmap, label) in enumerate([
            ("gene_expr", "Blues", "RNA"), ("peak_acc", "Purples", "ATAC"),
            ("pair_activity", "YlOrBr", "Pair activity")]):
            ax = axes[row, col]
            values = frame[key].to_numpy()
            maximum = max(float(np.percentile(values, 99)), 1e-12)
            points = ax.scatter(frame.x, frame.y, c=values, s=10, linewidths=0,
                                cmap=cmap, vmin=0, vmax=maximum)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.colorbar(points, ax=ax, shrink=.75, fraction=.035, pad=.02)
            if col == 0:
                ax.text(-.07, .5, label, rotation=90, va="center", ha="right", transform=ax.transAxes)
    return fig
