"""
Width from Frequency
====================

Calculates the width of the axion just from the frequency based on results from cosmology
"""


def width_from_fq(
    freq: float
) -> float:
    """
    Calculates the width of the axion

    Parameters
    ----------
    freq: float
        axion frequency
        
    Returns
    -------
    width: float
        width of axion
        
    """
    width_fq_ratio = 1e-6
    width = float(freq) * width_fq_ratio
    return width
