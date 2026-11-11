"""Cluster the Human tonsil analysis."""
from pathlib import Path
from melon_pkg.cli_reproduce_cluster import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
