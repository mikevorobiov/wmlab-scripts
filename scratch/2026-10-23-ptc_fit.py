"""
Photon transfer curve (PTC) analysis: camera gain K, read noise, PRNU, saturation.

Method: Janesick, Photon Transfer (SPIE, 2007).
For every (gain, exposure) point, with dark-subtracted mean signal S [DN]:
    total temporal variance   var_total = var(A - B) / 2        (light pair)
    read (+dark) variance     var_read  = var(Ad - Bd) / 2      (dark pair)
Pair differences cancel fixed-pattern noise, so it never inflates the estimate.
Pixels whose temporal variance is far above the typical one (hot, blinking/RTS)
are excluded from the variances; the fraction excluded is reported.

K [e-/DN] is the inverse slope of a straight-line fit
    var_total = a + S / K
over the shot-noise-limited points (5-70 % of the saturation signal). The free
intercept a makes K immune to any signal-independent variance. a should equal
var_read; a mismatch is reported as "excess" variance present only in lit frames.

Saturation is taken from the variance peak, which covers both ADC clipping and
full well filling before the top code.

Units: native 12-bit DN when the camera left-shifts its 12-bit codes into Mono16,
otherwise 16-bit DN (see labtools/sensor.py). Use the model in the same units,
pixel format and gain it was measured at.

Single-pixel noise model for a raw frame:
    sigma^2(S) = read_dn^2 [+ excess] + S / K
The PRNU term (prnu * S)^2 is spatial. It cancels in flat-fielded or ratio images
and is absent from frame-to-frame fluctuations of the same pixel.

Usage:
    python analyze/ptc_fit.py <out_dir> <raw_run_dir> [<raw_run_dir> ...]
One raw run per gain (see acquire/ptc_acquire.py); all runs go into one summary and plot.
Outputs in <out_dir>:
    ptc_points.csv   every (gain, exposure) point
    ptc_summary.csv  one row per gain: K, read noise, offset, PRNU, saturation
    ptc.png          noise vs signal, log-log, with the fitted model
"""

import pathlib
import sys

import numpy as np

from labtools.sensor import ADC_BITS, clip_level, full_scale, lsb_shift

FIT_RANGE = (0.05, 0.70)        # fraction of the saturation signal where K is fitted
OUTLIER_FACTOR = 50.0           # pixel excluded if its temporal var > 50 x the median one
SATURATED_FRACTION = 1e-3       # point counts as ADC-saturated above this ROI fraction clipped

# ------------------------------------------------------------------
# Detect and flag outliers

def central_roi(ny: int, nx: int, fraction: float) -> tuple[slice, slice]:
    """
    Define central region covering `fraction` of each dimension to 
    avoid vignetting effects seeping into the analysis.
    """
    dy = int(ny * fraction / 2) 
    dx = int(nx * fraction / 2)
    return slice(ny // 2 - dy, ny // 2 + dy), slice(nx // 2 - dx, nx // 2 + dx)


def mad_test(stack: np.ndarray, alpha: float = 0.05):
    """
    Median Absolute Deviation (MAD) test for time series of images
    used to detect hot and spurious outlier pixels.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    s = np.abs(stack - np.median(stack, axis=(1, 2), keepdims=True))
    threshold = OUTLIER_FACTOR * 1.4826 * np.max(float(np.median(s)), 1.0)
    out_pixel_mask = s > threshold
    return out_pixel_mask
    
    
def ptc_point(light: np.ndarray, dark: np.ndarray, clip: float,
              bad: np.ndarray | None = None) -> dict:
    n = light.shape[0]
    outlier_mask = mad_test(dark) # get bad pixels mask
    



def main(raw_dirs: list[pathlib.Path], roi_fraction: float = 0.5):
    import h5py
    from labtools.provenance import file_ref, stamp
    
    for h5 in h5_files:                         # one file at a time keeps memory to one run
        with h5py.File(h5, "r") as f:
            
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]])

 
    