"""Tests use small synthetic data in temporary folders; nothing reads MTS_DATA_DIR, MTS_HIDDEN_DIR or the network."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # synth.py, fakes.py
for _k in ("MTS_DATA_DIR", "MTS_HIDDEN_DIR", "MTS_RUNS", "MTR_DATA_DIR"):
    os.environ.pop(_k, None)
