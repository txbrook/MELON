"""Cluster the Human lymph node analysis."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.cluster import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
