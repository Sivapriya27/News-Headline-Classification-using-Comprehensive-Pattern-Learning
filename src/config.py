from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"
PLOTS_DIR = ARTIFACTS / "plots"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
