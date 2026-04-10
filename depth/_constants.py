"""Lightweight constants (no torch/transformers)."""

# Default depth calibration: model tends to over-estimate vs radar GT (~est/gt>1); scale down.
DEFAULT_EST_SCALE = 0.4
