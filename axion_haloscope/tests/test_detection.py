# tests/test_detection.py
import numpy as np
from axion_haloscope.detection import threshold_for_detection

def test_threshold_monotonic():
    t95 = threshold_for_detection(5.0, 0.95)
    t99 = threshold_for_detection(5.0, 0.99)
    assert t99 > t95  # higher confidence ⇒ higher threshold
