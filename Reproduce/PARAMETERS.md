# Parameter sources

The [analysis notebooks](README.md) read existing MELON outputs directly and identify their input files in the code. Archived run settings can differ from the manuscript-default runners described below.

The three 100-epoch workflows follow Supplementary Table 9 and the dataset descriptions in the supplied manuscript: 128-dimensional embeddings, batch size 2048, spatial k = 16, Adam (learning rate 0.001, weight decay 0.00001), temperature 0.1, and augmentation probability 0.8. Training seed 2027 is the original package default; the manuscript does not specify its numerical value. Clustering seed is 2025.

The five-modality mouse brain workflow follows the saved paper-run metadata: 10 epochs, seed 2025, batch size 2048, training k = 16, augmentation probability 1.0, and spatial export k = 14. It preserves the archived AdamW optimizer, neighbor-mixing implementation, and two-direction contrastive loss. Clustering uses PCA20, row L2 normalization, a 15-neighbor graph, and Leiden resolution 0.70357421875. The archived partition contains 17 clusters; a retrained model may produce a different count, which is reported without forcing labels.

Tonsil uses native embeddings and k-means (k = 10, n_init = 20), with the original global embedding export. Lymph node and E13 use the repository's 10-neighbor Leiden runner at resolution 1, including its existing target-count handling: merge smallest communities when above target, or spectral clustering when below target. Their sweeps are 6/8/10/12/14 and 10/12/14/16/18, respectively.

The shared multimodal components are copied from the original five-modality analysis source; [source_manifest.json](_shared/publication_model/source_manifest.json) records their SHA-256 hashes. The runners apply the manuscript's Adam optimizer and shared-permutation augmentation for the three default workflows, and preserve the recorded five-modality settings separately. These workflows implement the documented settings; they do not include the large input datasets or claim bit-for-bit regeneration of every published result.
