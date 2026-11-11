"""Run the Human tonsil analysis."""
from melon_pkg.cli_reproduce import main

config = {'name': 'Human tonsil',
 'spots': 4194,
 'modalities': ['RNA', 'Protein', 'Image'],
 'epochs': 100,
 'training_seed': 2027,
 'clustering_seed': 2025,
 'augment_p': 0.8,
 'export_spatial_k': None,
 'cluster_sweep': [10],
 'historical_five_modal': False,
 'dataset': 'human_tonsil',
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
