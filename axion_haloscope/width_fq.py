import numpy as np
from scipy.stats import norm
from numpy import random

def width_from_fq(fq_range):
    width_fq_ratio = 1e-6
    width = float(fq_range) * width_fq_ratio
    return width