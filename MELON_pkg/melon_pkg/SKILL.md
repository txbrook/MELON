---
name: Run MELON
description: Instructions for training the MELON model on multiomic spatial transcriptomics data or image features with spatial transcriptomics.
---

# Run MELON

This skill describes how to run the MELON (Metric Learning on Omics Networks) model. MELON can handle multiomic spatial transcriptomics data (e.g., RNA + ATAC) or integrate image features (e.g., Conch) with spatial transcriptomics.

## Prerequisites

- **Environment**: Python environment with `scanpy`, `pandas`, `numpy`, `torch`, and the `melon_pkg` installed or available in path.
- **Input Data**: An `.h5ad` file containing:
    - `adata.X`: Gene expression matrix (filtered to top 3000 HVGs by default).
    - `adata.obsm['conch_features']`: Image feature embeddings (Modality 2).
    - `adata.obsm['spatial']`: Spatial coordinates (x, y).

## Workflow

1.  **Load Data**: Load the `.h5ad` file.
2.  **Filter Genes**: Select top 3000 highly variable genes based on variance.
3.  **Prepare Inputs**: Create DataFrames for Gene Expression (Mod1), Image Features (Mod2), and Coordinates.
4.  **Run MELON**: Execute the training pipeline.
5.  **Save Results**: Save the trained model weights (`.pt`) and the generated embeddings (`.csv`).

## Python Implementation

You can use the following script template to run MELON. Update `INPUT_FILE` and `OUTPUT_DIR` as needed.

```python
import sys
import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch

# Ensure melon_pkg is importable (adjust path if necessary)
sys.path.insert(0, os.path.abspath("."))
try:
    from melon_pkg.api import MELON
except ImportError:
    print("❌ Error: 'melon_pkg' not found. Please ensure it is in the Python path.")
    sys.exit(1)

# ================= Configuration =================
INPUT_FILE = "path/to/your/data.h5ad"  # <--- Update this
OUTPUT_DIR = "melon_output"            # <--- Update this
TEMP_COORDS = "temp_coords.csv"
TOP_N_GENES = 3000
# =================================================

def run_melon():
    print("🚀 Script process started...", flush=True)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    print(f"📊 Loading {INPUT_FILE}...", flush=True)
    adata = sc.read_h5ad(INPUT_FILE)
    print(adata)

    # 2. Filter to top N highly variable genes
    print(f"   • Filtering to top {TOP_N_GENES} highly variable genes...", flush=True)
    
    # Handle sparse vs dense matrices
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
        
    # Calculate variance and select top genes
    gene_vars = np.var(X, axis=0)
    if len(gene_vars) > TOP_N_GENES:
        top_indices = np.argsort(gene_vars)[::-1][:TOP_N_GENES]
        top_indices = np.sort(top_indices) # Keep original order
        
        adata = adata[:, top_indices].copy()
        X_data = X[:, top_indices]
        print(f"     Kept {TOP_N_GENES} genes with highest variance.")
    else:
        print(f"     Dataset has fewer than {TOP_N_GENES} genes. Using all.")
        X_data = X

    # 3. Prepare Inputs for MELON
    # Modality 1: Gene Expression
    mod1_df = pd.DataFrame(X_data, index=adata.obs_names, columns=adata.var_names)
    
    # Modality 2: Image Features (expecting 'conch_features')
    if 'conch_features' not in adata.obsm:
        print("❌ Error: 'conch_features' not found in adata.obsm")
        return

    conch = adata.obsm["conch_features"]
    if hasattr(conch, "toarray"):
        conch = conch.toarray()
    conch = np.asarray(conch)
    mod2_df = pd.DataFrame(
        conch,
        index=adata.obs_names,
        columns=[f"feat_{i}" for i in range(conch.shape[1])]
    )
    
    # Coordinates
    coords_df = pd.DataFrame(
        adata.obsm['spatial'],
        index=adata.obs_names,
        columns=['x', 'y']
    )
    coords_df.to_csv(TEMP_COORDS)
    
    # 4. Run MELON
    print("🧠 Executing MELON training...", flush=True)
    try:
        model, mod1_t, mod2_t, coords_t, mask = MELON(
            mod1_df, 
            mod2_df, 
            coords_path=TEMP_COORDS,
            out_dir=OUTPUT_DIR,
            epochs=10,
            # For noise-free evaluation, you can disable augmentations:
            # augmentations=(),
        )
        print("🎉 MELON run completed successfully!")
        
        # 5. Save Outputs
        # Save Model Weights
        model_path = os.path.join(OUTPUT_DIR, "melon_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"   • Model saved to {model_path}")
        
        # Extract and Save Embeddings
        print("   • Computing embeddings...")
        model.eval()
        with torch.no_grad():
            encoder = model.encoder
            embeddings = encoder([mod1_t, mod2_t])
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = embeddings
                
        embed_path = os.path.join(OUTPUT_DIR, "melon_embeddings.csv")
        embed_df = pd.DataFrame(
            embeddings_np,
            index=mod1_df.index,
            columns=[f"Dim_{i}" for i in range(embeddings_np.shape[1])]
        )
        embed_df.to_csv(embed_path)
        print(f"   • Embeddings saved to {embed_path}")
        
    except Exception as e:
        print(f"❌ MELON Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(TEMP_COORDS):
            os.remove(TEMP_COORDS)

if __name__ == "__main__":
    run_melon()
```

