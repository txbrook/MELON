# Reproduce

Follow each analysis from multimodal inputs through MELON configuration, representation learning, spatial clustering or target prediction, and visualization. The notebooks include the code for each computational stage and continue from the supplied intermediate files, without repeating model fitting.

**Data:** `Published_Data_non_NASH` will be available through sharedoc; the link will be added after upload. Place the extracted folder beside `Reproduce/`, or set `MELON_PUBLISHED_DATA` to its location. Notebook outputs are included for reading on GitHub.

## Mouse E15.5 brain

[Open the notebook](mouse_e15_brain/analysis.ipynb). Integrates RNA and ATAC, identifies 12 spatial domains, and maps them onto the tissue.

![Mouse E15.5 brain: MELON results](figures/mouse_e15_brain_saved.png)

## Human tonsil

[Open the notebook](human_tonsil/analysis.ipynb). Follows multimodal integration, domain identification and spatial visualization.

![Human tonsil: MELON results](figures/human_tonsil_saved.png)

## Five-modality mouse brain

[Open the notebook](mouse_brain/analysis.ipynb). Integrates five modalities, constructs a joint representation, identifies spatial domains and visualizes them with UMAP.

![Five-modality mouse brain: MELON results](figures/mouse_brain_saved.png)

## Human lymph node

[Open the notebook](human_lymph_node/analysis.ipynb). Follows multimodal integration, domain identification and spatial visualization.

![Human lymph node: MELON results](figures/human_lymph_node_saved.png)

## Mouse E13 embryo

[Open the notebook](mouse_e13_embryo/analysis.ipynb). Follows RNA–ATAC integration, spatial domains, t-SNE and the separate cis-regulatory input branch.

![Mouse E13 embryo: MELON results](figures/mouse_e13_embryo_saved.png)

## MELON robustness

[Open the notebook](robustness/analysis.ipynb). Follows parameter selection, MELON integration, domain identification and sensitivity evaluation.

![MELON robustness: MELON results](figures/robustness_saved.png)

## MELON controlled simulation

[Open the notebook](controlled_simulation/analysis.ipynb). Follows paired inputs, MELON representation learning, target prediction and evaluation.

![MELON controlled simulation: MELON results](figures/controlled_simulation_saved.png)

## Training and clustering

Dataset directories retain their scripts for preprocessed, barcode-aligned input. Parameters are in [PARAMETERS.md](PARAMETERS.md).

Please follow the [Tutorials](https://txbrook.github.io/MELON_website/tutorials/) for installation and usage. Plotting dependencies are listed in [requirements.txt](requirements.txt).
