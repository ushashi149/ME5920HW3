

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


DEFAULT_DATA_ROOTS = [
    PROJECT_ROOT
    / "Video Dataset of Sheep Activity (Grazing, Running, Sitting)"
    / "Grazing_Running_Sitting Sheep_Classes",
    PROJECT_ROOT
    / "Video Dataset of Sheep Activity (Standing and Walking)"
    / "Standing_Walking_Misc_Sheep_Classes",
]


CLASS_NAMES = ["Grazing", "Running", "Sitting", "Standing", "Walking"]


CLASS_TO_SUBDIR = {
    "Grazing": "Grazing",
    "Running": "Running",
    "Sitting": "Sitting",
    "Standing": "Standing",
    "Walking": "Walking",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".MP4", ".MOV", ".m4v"}

# Processed outputs
PROCESSED_DIR = PACKAGE_DIR / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
COMPRESSED_DIR = PROCESSED_DIR / "compressed_videos"


SEED = 42
FRAME_SIZE = 112  
TRAIN_FRAMES_PER_VIDEO = 8  
TEST_FRAMES_PER_VIDEO = 1  


CLIP_LENGTH = 8
CLIP_STRIDE = 2


LSTM_SEQ_LEN = 16
EMBED_DIM = 512  

RUNS_DIR = PACKAGE_DIR / "runs"
