# melon-pkg

The supplementary MELON package learns a joint embedding from paired spatial molecular profiles and spot coordinates. It includes Python and command-line interfaces for training and clustering.

## Install

From the MELON repository root, in an activated Python environment, run

```bash
python -m pip install -e ./code/melon_pkg
```

Optional `.h5ad` input support can be installed with

```bash
python -m pip install -e './code/melon_pkg[h5ad]'
```

## Verify

```bash
melon-run --help
melon-cluster --help
```

## Programmatic usage

Provide two pandas DataFrames with spatial spots as rows, molecular features as columns, and matching barcode indices. The coordinate CSV must use the same barcodes as its index and contain columns `x` and `y`.

```python
from melon_pkg import MELON

model, mod1_t, mod2_t, coords_t, neighbor_mask = MELON(
    mod1_df,
    mod2_df,
    coords_path="coords.csv",
    out_dir="melon_output",
    epochs=10,
    augmentations=("gaussian_noise", "feature_dropout", "neighbor_mix"),
)
```

For the mouse E15.5 example, follow the [repository README](../../README.md). Data preparation and the analysis workflow are described in the [Tutorials](https://txbrook.github.io/MELON_website/tutorials/).

## Tests

From the repository root, install the development dependencies and run the bundled tests.

```bash
python -m pip install -e './code/melon_pkg[dev]'
python -m pytest code/melon_pkg/tests
```

## License

The package is distributed under the [MIT License](LICENSE).
