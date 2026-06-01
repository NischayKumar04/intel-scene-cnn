from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "intel-image-classification"

TRAIN_DIR = RAW_DATA_DIR / "seg_train" / "seg_train"
TEST_DIR = RAW_DATA_DIR / "seg_test" / "seg_test"
PRED_DIR = RAW_DATA_DIR / "seg_pred" / "seg_pred"

SAVED_MODEL_DIR = BASE_DIR / "saved_models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 20

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]