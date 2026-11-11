# Five-modality mouse brain

## Analysis workflow

[Open the notebook](analysis.ipynb). Edit parameters directly in its configuration cell.

![MELON spatial partition](../figures/mouse_brain_saved.png)

## Run from preprocessed data

Input: 9,567 aligned spots in `features.npz`, with `barcodes`, `coords`, and five preprocessed 30-dimensional matrices: `RNA`, `Protein`, `ATAC`, `H3K27ac`, and `H3K27me3`. RNA and protein are PCA features; the three chromatin inputs are ArchR LSI features.

Train for 10 epochs with batch size 2,048 and seed 2025. Export uses 14 spatial neighbors. Clustering uses 20 principal components, row L2 normalization, a 15-neighbor graph, and Leiden community detection. Run from the repository root:

```bash
python Reproduce/mouse_brain/run.py --features data/mouse_brain/features.npz --out results/mouse_brain
python Reproduce/mouse_brain/cluster.py --embedding results/mouse_brain/embedding.csv --coords results/mouse_brain/coords.csv --out results/mouse_brain/clustering --umap
```

To export an existing model, add `--checkpoint path/to/MELON_10epochs.ckpt`. The saved paper resolution yields 17 clusters for the archived embedding; the observed count and spatial continuity metrics are saved for each run. Edit the model and clustering parameters directly in the configuration cell of the [notebook](analysis.ipynb).
