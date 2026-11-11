"""Run the Human lymph node analysis."""
from pathlib import Path
from melon_pkg.cli_reproduce import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
