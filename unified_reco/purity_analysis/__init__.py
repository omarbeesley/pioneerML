"""PIONEER R_e/mu analysis framework (main PURITY model).

Runs on MC truth (mode='truth') to validate the chain, or on the model's benchmark.py
predictions (mode='reco'). See io.py (standardization), measurement.py (the measurement),
run_analysis.py (CLI + plots).
"""
from . import io, measurement  # noqa: F401
