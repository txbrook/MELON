# Figure sources

Paths below are relative to the sharedoc archive `Published_Data_non_NASH/`. The notebooks display MELON results from the archived arrays and labels. Source arrays and cluster IDs are unchanged.

| Figure | Archive folder | Files used by the notebook |
| --- | --- | --- |
| 2a | `01_Mouse_E15_5` | `03_embeddings/MELON.csv`; `04_clusters/final_clusters.csv` |
| 3a | `04_Human_tonsil` | `03_embeddings/MELON.csv`; `04_clusters/final_clusters.csv`; `display_color_mappings.json` |
| 3f | `05_Mouse_brain_five_modalities` | `03_embeddings/aligned_embeddings.npz`; `04_clusters/final_clusters.csv`; `cluster_palette.csv` |
| 3g | `05_Mouse_brain_five_modalities` | `04_clusters/UMAP_coordinates.csv`; `cluster_palette.csv` |
| 4a | `02_Human_lymph_node` | `03_embeddings/MELON.csv`; `04_clusters/final_clusters.csv` |
| 4d | `03_Mouse_E13` | `03_embeddings/MELON.csv`; `04_clusters/final_clusters.csv` |
| 4f input branch | `03_Mouse_E13` | `03_embeddings/decoder_MACS3/MELON.csv`; `02_model_inputs/decoder_MACS3/saved_settings.json` |
| 5a-b | `01_Mouse_E15_5` | `05_Fig5_robustness/neighbor_k_local_report_metrics.csv`; `augment_p_report_metrics.csv` |
| Related ablation sweep (5c requires reconciliation) | `01_Mouse_E15_5` | `05_Fig5_robustness/ablation_multik_metrics_E15.csv` |
| Supplementary 7e / 8e | `02_Human_lymph_node` / `03_Mouse_E13` | `03_embeddings/supplementary_tsne/saved_tsne.csv`; `source_embedding.csv` |
| Supplementary 10 | `06_Controlled_simulation` | `02_model_inputs/seed*/X_*.npy`; `03_embeddings/seed*/MELON.csv`; `04_saved_evaluation/MELON_summary.csv` and MELON ablation summaries |

## Panels with additional source tables

Additional MELON panels require the following saved plotting tables outside this archive:

- Fig. 2b-c and Fig. 4b/e: final multi-resolution benchmark and region-level summaries.
- Fig. 2d-f: final expression, enrichment, t-SNE and interaction display tables.
- Fig. 2g-h, Fig. 3d/j and Fig. 4c/f: external-support and selected decoder pair-activity tables. E13 includes the decoder embedding, settings and MACS3 matrix, but not the final selected pair-activity table.
- Fig. 3b-c: final compartment and paired-bootstrap score tables.
- Fig. 3h-i: final spatial-continuity/PAS summary table.
- Fig. 5d-f: final fusion-gain summary table.

## Fig. 5c discrepancy

The archived full-model median ARI across the six saved cluster counts is 0.383; the supplied Fig. 5c shows approximately 0.48. The notebook labels its plot as the archived sweep rather than the final panel. The final Fig. 5c plotting table needs to be identified before claiming an exact reconstruction.

## Saved-run provenance

E15.5 records seed 42, 100 epochs, batch size 2048 and spatial k = 16. Tonsil records seed 42, 30 epochs, batch size 4096 and k = 6. The E13 main run records seed 2021, shuffled batches and a 24,017-feature second modality. These settings describe the stored results; the generic manuscript-default training scripts are a separate workflow.

The five-modality archive distinguishes the native 128-dimensional MELON embedding, the saved PCA20/L2 clustering representation, and UMAP display coordinates. The simulation notebook selects the saved MELON MLP probe for Y and linear-SVM probes for R/U/C.
