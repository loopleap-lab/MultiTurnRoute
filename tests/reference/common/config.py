"""Locations and the model pool. The data folder defaults to ../data; override it with the environment variable MTR_DATA_DIR."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MTR_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
MODELS = ["granite-3b", "gemma-e4b", "gpt-oss-20b", "devstral-24b", "gemma-26b-a4b", "granite-30b", "gemma-31b",
          "nemotron-30b-a3b", "mistral-medium-128b"]
ENCODER_MODEL = "granite-3b"            # the frozen encoder of the precomputed embeddings (its input tokens are charged)
FULL_HISTORY_CHARS = 2000               # characters of "history + query" behind the "h" embedding
