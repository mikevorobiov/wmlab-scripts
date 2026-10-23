"""
ADC conventions for the FLIR camera: 12-bit ADC delivered as Mono16.

Two output scalings occur, and which one applies is detected from the data:
  - left-shifted 12-bit: every value a multiple of 16, top code 65520. to_native()
    divides by 16, so units are native 12-bit DN (0..4095).
  - rescaled 16-bit: values span 0..65535 but are not multiples of 16 (seen at high
    gain, where the 12-bit codes are rescaled after the ADC, e.g. by digital gain).
    Units stay 16-bit DN.
Unshifted data that never exceed 4095 are taken as native 12-bit.
A noise model is only valid in the units it was measured in, so keep the PTC and
the measurement in the same pixel format and gain.

Shared by acquisition and analysis so both use the same rule.
"""

import numpy as np

ADC_BITS = 12
FULL_SCALE = (1 << ADC_BITS) - 1          # 4095, native 12-bit DN
CLIP_MARGIN_CODES = 2                     # within this many 12-bit codes of the top = clipped


def lsb_shift(images: np.ndarray) -> int:
    """4 if the data are 12-bit values left-shifted into 16 bits, else 0."""
    sample = np.asarray(images).reshape(-1)[:200_000]
    extra = 16 - ADC_BITS
    return extra if np.all(sample % (1 << extra) == 0) else 0


def full_scale(images: np.ndarray, shift: int | None = None) -> float:
    """Top code in the units to_native() returns. Decide it from a BRIGHT stack:
    a dark stack of rescaled 16-bit data may never exceed 4095."""
    if shift is None:
        shift = lsb_shift(images)
    if shift:
        return float(FULL_SCALE)
    return 65535.0 if np.asarray(images).max() > FULL_SCALE else float(FULL_SCALE)


def clip_level(fs: float) -> float:
    """Values at or above this count as ADC-clipped. A margin is used because
    clipped pixels may sit one code below the nominal top (e.g. 65504 = 4094 * 16)."""
    code = 1.0 if fs <= FULL_SCALE else (65535.0 + 1) / (FULL_SCALE + 1)    # 1 or 16 DN
    return fs - CLIP_MARGIN_CODES * code


def to_native(images: np.ndarray, shift: int | None = None) -> np.ndarray:
    """float64, with a left shift undone (see module docstring for the units)."""
    if shift is None:
        shift = lsb_shift(images)
    return np.asarray(images, dtype=np.float64) / (1 << shift)


def clipped_mask(images: np.ndarray, shift: int | None = None, fs: float | None = None) -> np.ndarray:
    """True where the ADC is at (or within CLIP_MARGIN_CODES of) its top code.
    Pass shift and fs from a bright stack when masking dark frames."""
    if shift is None:
        shift = lsb_shift(images)
    if fs is None:
        fs = full_scale(images, shift)
    return to_native(images, shift) >= clip_level(fs)
