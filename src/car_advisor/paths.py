import os
from pathlib import Path

ROOT = Path(os.environ.get("CAR_ADVISOR_DATA", Path(__file__).parents[2] / "data"))
RAW = ROOT / "raw"
ARTICLES = ROOT / "articles"
CHROMA = ROOT / "chroma"
CATALOG = ROOT / "catalog.db"
CHECKPOINTS = ROOT / "checkpoints.db"
