"""
ADC conventions for the FLIR camera: 12-bit ADC delivered as Mono16.

Mono16 output from a 12-bit ADC may be left-shifted by 4 bits (MSB-aligned), so
every value is a multiple of 16 and full scale reads 65520 instead of 4095.
Whether it is shifted is detected from the data, not assumed. Shared by
acquisition and analysis so both use the same rule.
"""

import numpy as np

ADC_BITS = 12
FULL_SCALE = (1 << ADC_BITS) - 1          # 4095, in native DN


def lsb_shift(images: np.ndarray) -> int:
    """4 if the data are 12-bit values left-shifted into 16 bits, else 0.
    A shifted value is always a multiple of 16. An unshifted frame with any noise
    (or any pixel at 4095) contains non-multiples, so this fails only for a frame
    that is constant zero."""
    sample = np.asarray(images).reshape(-1)[:200_000]
    extra = 16 - ADC_BITS
    return extra if np.all(sample % (1 << extra) == 0) else 0


def to_native(images: np.ndarray, shift: int | None = None) -> np.ndarray:
    """Convert to float64 native ADC units (DN, 0..4095)."""
    if shift is None:
        shift = lsb_shift(images)
    return np.asarray(images, dtype=np.float64) / (1 << shift)


def clipped_mask(images: np.ndarray, shift: int | None = None) -> np.ndarray:
    """True where the ADC is at its top code. Catches ADC clipping only; full-well
    saturation below 4095 shows up as the variance rollover in the PTC instead."""
    if shift is None:
        shift = lsb_shift(images)
    return (np.asarray(images) >> shift) >= FULL_SCALE
