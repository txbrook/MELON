# Parameter sources

The [analysis notebooks](README.md) connect model inputs, training, clustering and evaluation through computational code and intermediate files. Archived run settings can differ from the manuscript-default runners described below.

The three 100-epoch workflows follow Supplementary Table 9 and the dataset descriptions in the supplied manuscript: 128-dimensional embeddings, batch size 2048, spatial k = 16, Adam (learning rate 0.001, weight decay 0.00001), temperature 0.1, and augmentation probability 0.8. Training seed 2027 is the original package default; the manuscript does not specify its numerical value. Clustering seed is 2025.

The five-modality mouse brain workflow follows the saved paper-run metadata: 10 epochs, seed 2025, batch size 2048, training k = 16, augmentation probability 1.0, and spatial export k = 14. It preserves the archived AdamW optimizer, neighbor-mixing implementation, and two-direction contrastive loss. Clustering uses PCA20, row L2 normalization, a 15-neighbor graph, and Leiden resolution 0.70357421875. The archived partition contains 17 clusters; a retrained model may produce a different count, which is reported without forcing labels.

Tonsil uses native embeddings and k-means (k = 10, n_init = 20), with the original global embedding export. Lymph node and E13 use the repository's 10-neighbor Leiden runner at resolution 1, including its existing target-count handling: merge smallest communities when above target, or spectral clustering when below target. Their sweeps are 6/8/10/12/14 and 10/12/14/16/18, respectively.

The shared multimodal components are copied from the original five-modality analysis source; [source_manifest.json](_shared/publication_model/source_manifest.json) records their SHA-256 hashes. The runners apply the manuscript's Adam optimizer and shared-permutation augmentation for the three default workflows, and preserve the recorded five-modality settings separately. These workflows implement the documented settings; they do not include the large input datasets or claim bit-for-bit regeneration of every published result.

The controlled-simulation notebook uses the full-model configuration from `run_v4_2_pairscale30_a1scale_probe.py`: 20 epochs, 256 embedding dimensions, spatial k = 4, augmentation probability 0.3, batch size 2,048, and learning rate 0.002. Its MLP and linear-SVM evaluation follows `phase2_eval.py`, with five stratified folds and fold-specific standardization. The notebook connects these computational stages to the supplied simulation nodes.

Notebook values follow the intermediate outputs: E15.5 uses seed 42; E13 uses seed 2021, shuffled batches and its 24,017-feature ATAC branch; tonsil uses 30 epochs, batch size 4,096, spatial k = 6 and seed 42. The tonsil run used an algebraically equivalent memory-efficient InfoNCE implementation. The original lymph-node training seed is not recorded with the embedding, so its editable configuration uses the package default.

The archive identifies assay input files, but does not contain the runtime-normalized arrays for every dataset. Training functions accept prepared modality arrays; the notebooks continue from the supplied embeddings and partitions.
