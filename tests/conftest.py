"""Tests use small synthetic data in temporary folders; nothing reads SRX_DATA_DIR, SRX_HIDDEN_DIR or the network."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # synth.py, fakes.py
for _k in ("SRX_DATA_DIR", "SRX_HIDDEN_DIR", "SRX_RUNS", "MTR_DATA_DIR"):
    os.environ.pop(_k, None)
