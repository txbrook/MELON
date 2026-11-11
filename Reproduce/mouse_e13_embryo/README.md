# Mouse E13 embryo

## Analysis workflow

[Open the notebook](analysis.ipynb). Edit parameters directly in its configuration cell.

![MELON spatial partition](../figures/mouse_e13_embryo_saved.png)

## Run from preprocessed data

Input: 2,187 aligned spots in `features.npz`, with `barcodes`, `coords`, `RNA` (normalized log-expression), and `ATAC` (50-dimensional TF-IDF/LSI features). Both modalities must already be preprocessed.

Train for 100 epochs with batch size 2,048. Clustering evaluates 10, 12, 14, 16, and 18 clusters; the main spatial partition uses 14. Run from the repository root:

```bash
python Reproduce/mouse_e13_embryo/run.py --features data/e13/features.npz --out results/e13
python Reproduce/mouse_e13_embryo/cluster.py --embedding results/e13/embedding.csv --coords results/e13/coords.csv --out results/e13/clustering
```

Add `--truth data/e13/truth.csv` for ARI/NMI evaluation (CSV columns: `barcode,label`), and `--umap` for UMAP coordinates. Edit the model and clustering parameters directly in the configuration cell of the [notebook](analysis.ipynb).
