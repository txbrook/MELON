# Reproduce

Follow the saved MELON results through short explanations, reading code and displayed plots. The notebooks use existing embeddings, partitions and evaluation tables.

**Data:** `Published_Data_non_NASH` will be available through sharedoc; the link will be added after upload. Place the extracted folder beside `Reproduce/`, or set `MELON_PUBLISHED_DATA` to its location. Notebook outputs are included for reading on GitHub.

## Mouse E15.5 brain

[Open the notebook](mouse_e15_brain/paper_figures.ipynb). Reads the saved MELON embedding and places its 12-domain partition on the tissue.

![Mouse E15.5 brain: saved MELON results](figures/mouse_e15_brain_saved.png)

## Human tonsil

[Open the notebook](human_tonsil/paper_figures.ipynb). Shows the saved MELON spatial partition.

![Human tonsil: saved MELON results](figures/human_tonsil_saved.png)

## Five-modality mouse brain

[Open the notebook](mouse_brain/paper_figures.ipynb). Follows the saved MELON inputs and embedding to its 17-domain partition and UMAP.

![Five-modality mouse brain: saved MELON results](figures/mouse_brain_saved.png)

## Human lymph node

[Open the notebook](human_lymph_node/paper_figures.ipynb). Shows the saved MELON spatial partition.

![Human lymph node: saved MELON results](figures/human_lymph_node_saved.png)

## Mouse E13 embryo

[Open the notebook](mouse_e13_embryo/paper_figures.ipynb). Shows the saved MELON spatial partition and t-SNE, with the separate MELON cis-pair input branch.

![Mouse E13 embryo: saved MELON results](figures/mouse_e13_embryo_saved.png)

## MELON robustness

[Open the notebook](robustness/paper_figures.ipynb). Reads the saved MELON neighborhood and augmentation sensitivity scores.

![MELON robustness: saved MELON results](figures/robustness_saved.png)

## MELON controlled simulation

[Open the notebook](controlled_simulation/paper_figures.ipynb). Displays the stored MELON target-prediction scores.

![MELON controlled simulation: saved MELON results](figures/controlled_simulation_saved.png)

## Training and clustering

Dataset directories retain their scripts for preprocessed, barcode-aligned input. Parameters are in [PARAMETERS.md](PARAMETERS.md).

Please follow the [Tutorials](https://txbrook.github.io/MELON_website/tutorials/) for installation and usage. Plotting dependencies are listed in [requirements.txt](requirements.txt).
