"""
FLIR Camera noise calibration
to extract Photon Transfer Function (PTF)
see details in 
Janesick Photon Transfer SPIE (2007)

Abstract:
I could not properly estimate the camera pixel
noise for fluorescence dip measurements in 
Rybderg EIT imaging.

Camera: Teledyne FLIR 

"""

#%%
import pathlib
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from labtools.provenance import file_ref, load_param, load_provenance, stamp

data_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")

from labtools.flir import FLIRCamera

#%%
params = {
    "frames_per_run": 2,
    "expousre_us": np.logspace(2, 4, 3),
    "gain_dB": np.linspace(0, 40, 5)
}

cam = FLIRCamera(cam_index=0)

cam.set_px_format_mono16bit()
pic = cam.get_frame()
cam.deinit()

plt.figure(1)
plt.imshow(pic)
plt.show()
# %%
def acquire_calibration_data(params):
    exposure = params["exposure_us"]
    gain = params["gain_dB"]
    N = params["frames_per_run"]    
    
    nx, ny = cam.resolution

    raw_data = []
    for g in gain:
        cam.set_gain(g)
        raw_data = {"gain_db": g}
        for ex in exposure:
            cam.set_exposure_time(ex)
            img_sequence = cam.get_frame_sequence(10)
            images_dict = {
                "exposure_us": ex,
                "images": img_sequence
            }
        raw_data.append({
            "gain_db": g,
            "exposure_data": images_dict
        })
    return raw_data

def process_calibration_data(raw_data: dict):
    pass

