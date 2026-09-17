from pathlib import Path
from dataclasses import dataclass

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 64
SEED = 42

INTENTS = ["GET_TOP_CELLS", "GET_CELL_DETAIL", "GET_CELL_COUNT", "UNKNOWN"]

SLOTS = [
    "O",
    "B-LIMIT",
    "B-CELL_ID",
    "B-LOCATION",
    "I-LOCATION",
    "B-METRIC",
    "I-METRIC",
    "B-ORDER",
    "I-ORDER",
]

INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}
ID_TO_INTENT = {i: name for name, i in INTENT_TO_ID.items()}
SLOT_TO_ID = {name: i for i, name in enumerate(SLOTS)}
ID_TO_SLOT = {i: name for name, i in SLOT_TO_ID.items()}

IGNORE_INDEX = -100


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 16
    max_length: int = MAX_LENGTH
    seed: int = SEED


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_VERSION = "multilingual-intent-slot-v2"
