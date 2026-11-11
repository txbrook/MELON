"""Cluster the Mouse E13 embryo analysis."""
from melon_pkg.cli_reproduce_cluster import main

config = {'name': 'Mouse E13 embryo',
 'spots': 2187,
 'modalities': ['RNA', 'ATAC'],
 'epochs': 100,
 'training_seed': 2027,
 'clustering_seed': 2025,
 'augment_p': 0.8,
 'export_spatial_k': 16,
 'cluster_sweep': [10, 12, 14, 16, 18],
 'historical_five_modal': False,
 'dataset': 'mouse_e13_embryo',
 'embed_dim': 128,
 'batch_size': 2048,
 'training_spatial_k': 16,
 'optimizer': 'Adam',
 'learning_rate': 0.001,
 'weight_decay': 1e-05,
 'temperature': 0.1,
 'shuffle_batches': False}

if __name__ == "__main__":
    main(config)
