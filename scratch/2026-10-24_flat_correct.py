"""
Take an image with the camera and flat-field correct it.
Run cell by cell in IPython / VS Code (#%%).

    corrected = (image - dark) / flat          bad pixels -> NaN

The gain always comes from the flat file: a flat is only valid at its own gain.
The dark is taken right after the image, with the same gain and exposure.
"""

#%% Settings
import numpy as np
import matplotlib.pyplot as plt
from labtools.data_io import load
from labtools.flir import FLIRCamera

FLAT_PATH = r"F:\Vaults\WnM-AMO\__Data_New\derived\runs\2026-10-24_run_002_flat_field_calibration\flat_30.0dB.h5"
CAM_INDEX = 1 
EXPOSURE_US = 250148.0         # None = the flat's exposure; or a number for your scene
N_FRAMES = 10               # frames averaged into the image (and into the dark)

#%% Load the flat
flat_file = load(FLAT_PATH)
flat = flat_file["flat"]
bad = flat_file["bad_pixels"]
gain_db = flat_file["attrs"]["gain_db"]

exposure_us = EXPOSURE_US
if exposure_us is None:
    exposure_us = flat_file["attrs"]["exposure_us"]
print(f"Imaging at {gain_db:.2f} dB, {exposure_us:.0f} us, {N_FRAMES} frames")

#%% Take the image, then the dark at the same settings
with FLIRCamera(cam_index=CAM_INDEX, exposure_value=exposure_us, gain_value=gain_db) as cam:
    cam.disable_fps_limit()                   # long exposures would otherwise be refused
    cam.set_exposure_time(exposure_us)

    image = cam.get_frame_sequence(N_FRAMES)["images"].mean(axis=0, dtype=np.float64)

    input("DARK: cover the lens, then press Enter ")
    dark = cam.get_frame_sequence(N_FRAMES)["images"].mean(axis=0, dtype=np.float64)

#%% Correct
signal = image - dark
corrected = signal / flat
corrected[bad] = np.nan                       # the flat cannot be trusted there

#%% Compare: pixel-to-pixel spread in the central region, before and after
def central_spread(picture):
    """Relative standard deviation of the central half, ignoring NaN."""
    height, width = picture.shape
    centre = picture[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
    return np.nanstd(centre) / np.nanmean(centre)

print(f"spread before: {100 * central_spread(signal):.2f} %")
print(f"spread after:  {100 * central_spread(corrected):.2f} %")

#%% Show before and after on the same colour scale
low, high = np.nanpercentile(corrected, [1, 99])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(signal, cmap="gray", vmin=low, vmax=high)
axes[0].set_title("image - dark")
axes[1].imshow(corrected, cmap="gray", vmin=low, vmax=high)
axes[1].set_title("(image - dark) / flat")
plt.show()
# %%
