# Reproduce

This directory contains MELON training and clustering workflows for the datasets analyzed in the paper. The four tissue workflows below start from **preprocessed, barcode-aligned data**.

| Dataset | Modalities | Code |
| --- | --- | --- |
| Human tonsil | RNA, protein, H&E image features | [human_tonsil](human_tonsil/) |
| Mouse brain | RNA, protein, ATAC, H3K27ac, H3K27me3 | [mouse_brain](mouse_brain/) |
| Human lymph node | RNA, ADT | [human_lymph_node](human_lymph_node/) |
| Mouse E13 embryo | RNA, ATAC | [mouse_e13_embryo](mouse_e13_embryo/) |
| Mouse E15.5 brain | RNA, ATAC | [clustering](clustering/) |

Each tissue directory provides `run.py`, `cluster.py`, and `settings.json`. Training writes `embedding.csv`, `model.ckpt`, and `run.json`; clustering writes barcode-indexed labels, spatial maps, and metrics. Optional reference labels enable ARI and NMI evaluation.

Use `features.npz` with a string array `barcodes`, an N × 2 array `coords`, and an N × D array for each modality named in the dataset instructions. All arrays must follow the same spot order. Features are used directly, without additional normalization.

Please follow the [Tutorials](https://txbrook.github.io/MELON_website/tutorials/) for installation and usage. Analysis dependencies are listed in [requirements.txt](requirements.txt); parameter sources are recorded in [PARAMETERS.md](PARAMETERS.md).
