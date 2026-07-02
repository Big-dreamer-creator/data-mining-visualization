import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.datasets import ensure_dataset_files


if __name__ == "__main__":
    ensure_dataset_files()
    print("Dataset CSV files are ready in ../dataset")
