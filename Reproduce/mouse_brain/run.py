"""Run the Five-modality mouse brain analysis."""
from melon_pkg.cli_reproduce import main

config = {'name': 'Five-modality mouse brain',
 'spots': 9567,
 'modalities': ['RNA', 'Protein', 'ATAC', 'H3K27ac', 'H3K27me3'],
 'epochs': 10,
 'training_seed': 2025,
 'clustering_seed': 2025,
 'augment_p': 1.0,
 'export_spatial_k': 14,
 'cluster_sweep': [17],
 'leiden_resolution': 0.70357421875,
 'historical_five_modal': True,
 'dataset': 'mouse_brain',
 'embed_dim': 128,
 'batch_size': 2048,
 'training_spatial_k': 16,
 'optimizer': 'AdamW',
 'learning_rate': 0.001,
 'weight_decay': 1e-05,
 'temperature': 0.1,
 'shuffle_batches': False}

if __name__ == "__main__":
    main(config)
