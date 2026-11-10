# MELON

![MELON framework](figures/MELON_framework.png)

MELON (Multi-modal lEarning framework for spatial muLti-Omics integratioN) is a multimodal learning framework for spatial multi-omics data integration. It encodes each modality separately and combines their representations through a fusion transformer with a learned spatial neighborhood bias. A contrastive objective compares modality-masked views with joint observations to preserve modality-specific and cross-modal synergistic information. MELON produces a joint representation for each spatial spot, supporting spatial domain identification and downstream biological analysis.

## Reproduce

The [`code/`](code/) directory contains the supplementary implementation, training and clustering entry points, and a script for the mouse E15.5 brain RNA–ATAC example. After installation, run the example with a directory containing `mRNA_data.h5`, `ATAC_data.h5`, and `coords.csv` (spot barcodes as the index and columns `x`, `y`).

```bash
bash code/run_mouse_e15.sh /absolute/path/to/Mouse_E15.5
```

The script applies TF-IDF and LSI to ATAC, trains MELON for 100 epochs with seed 42, and clusters the embedding with a target of 12 clusters. Results are saved under `code/results/example_mouse_e15/`. Processed data are available from [MISAR-seq](https://github.com/gpenglab/MISAR-seq), and the repository includes the two modality matrices in [`mouse_E15_brain/`](mouse_E15_brain/).

This supplementary code provides paired-modality embedding and clustering. The previous repository implementation is available on the [`legacy/pre-supplementary-update-20261110`](https://github.com/txbrook/MELON/tree/legacy/pre-supplementary-update-20261110) branch.

## Usage and installation

Use Python 3.12 to create an environment and install the package from this repository.

```bash
git clone https://github.com/txbrook/MELON.git
cd MELON
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./code/melon_pkg

melon-run --help
melon-cluster --help
```

For `.h5ad` inputs, install the optional dependencies with `python -m pip install -e './code/melon_pkg[h5ad]'`. The Python API and testing instructions are in the [package README](code/melon_pkg/README.md). Data preparation and a worked example are described in the [Tutorials](https://txbrook.github.io/MELON_website/tutorials/). Use the installation paths and reproduction command above for this repository layout.

## Citation

Zhang, Yuchen, Lisa Huang, Jian Gao, Yuying Xie, Yu Shen, Liuqing Yang, Chunru Lin, and Lulu Shang. “MELON: A Cross Modality Information Aware and Spatial Aware Multimodal Learning Framework for spatial multi-omics data integration.” Manuscript (2026).
