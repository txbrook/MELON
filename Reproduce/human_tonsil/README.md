# Human tonsil

## Results

Open the [analysis notebook](paper_figures.ipynb) to follow the saved MELON results through explanations, code and plots.

![Saved MELON spatial partition](../figures/human_tonsil_saved.png)

## Run from preprocessed data

Input: 4,194 aligned spots in `features.npz`, with `barcodes`, `coords`, `RNA` (normalized log-expression), `Protein` (CLR features), and `Image` (2,048 frozen InceptionV3 features). All features must already be preprocessed.

Train for 100 epochs with batch size 2,048, then cluster the native 128-dimensional embedding with k-means (10 clusters, 20 initializations). Run from the repository root:

```bash
python Reproduce/human_tonsil/run.py --features data/tonsil/features.npz --out results/tonsil
python Reproduce/human_tonsil/cluster.py --embedding results/tonsil/embedding.csv --coords results/tonsil/coords.csv --out results/tonsil/clustering --umap
```

Training parameters are in [settings.json](settings.json).
