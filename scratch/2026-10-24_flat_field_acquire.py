
"""
Author: Mykhailo Vorobiov
Date: 2026-10-24

Flat field image frames acquisition for the FLIR camera, 
for multiple gains at a fixed exposure.

Method: Janesick, Photon Transfer (SPIE, 2007).
    Take multiple (50+) images of a flat uniformly lit surface (white paper) 
    at each camera gain setting and fixed exposure: ~80% of saturation value. 
    The images further can be averaged into one to extract fixed pattern noise (FPN)
    of the camera that does not change from frame to frame. Averaging is required to 
    suppress camera and photon nosie and make it negledible than that of the 
    actual reading in the future. 
    
    
    Subsequent script must implement the following.
    The flat field image can be stored to remove FPN according 
    to the fomula (see Janesick 2007, page 111, formula (8.2))

            Sᵣ =  Sᵢ(μ₀/S₀),

    where Sᵣ is the pixel value with FPN normalized out,
    Sᵢ is the raw image from camera requiring FPN correction, 
    μ₀ is the mean vaue of the flat field image,
    S₀ is the flat field image pixel value. 

Before running:
  - Stable, flat illumination (LED + diffuser or integrating sphere), warmed up and on.
  - Camera exposure time is set such that camera signal is ~80% of saturation value. 
  - Fill in LIGHT_SOURCE below; it goes into the provenance record.

"""

import argparse
import pathlib

import h5py
import numpy as np

from labtools.flir import FLIRCamera
from labtools.provenance import stamp
from labtools.sensor import clipped_mask, full_scale, to_native
from labtools.data_io import save_h5
from labtools.find_flat_exposure import find_flat_exposure

LIGHT_SOURCE = ""      # e.g. "625 nm LED, 20 mA, opal diffuser"



def acquire_phase(cam: FLIRCamera, 
                  gain_db: float, 
                  n_frames: int, label: str):
    """n_frames at every exposure, at the current gain. Leading gain axis of length 1."""
    cam.set_gain(gain_db)
    nx, ny = cam.resolution
    images = cam.get_frame_sequence(n_frames)["images"]
    gain_rb = cam.gain_value # actual values, as quantized by the camera
    exp_rb = cam.exposure_value
    print(f"  {label} done")
    return images, gain_rb, exp_rb

def main(run_dir: pathlib.Path, gain_db: np.ndarray | float, 
         n_frames: int = 2,
         cam_index: int = 1, fov_mm: float = 11.5) -> None:
    if n_frames < 2:
        raise ValueError("Need at least 2 frames per point (pair difference)")
    params = {
        "cam_index": cam_index,
        "n_frames": n_frames,
        "fov_horizontal_mm": fov_mm,
        "frames_per_point": n_frames,
        "gain_db": gain_db,
        "light_source": LIGHT_SOURCE,
    }

    with FLIRCamera(cam_index=cam_index, fov_horizontal=fov_mm) as cam:
        cam.disable_fps_limit()     # otherwise the frame rate caps the max exposure

        print(f"Looking for exposures of the given gains: {gain_db} ...")
        exposures = find_flat_exposure(cam, gains_db=params["gain_db"], target=0.5)

        print(f"Start image acquisition for exposures: {exposures}")
        images_stack = np.zeros(gain_db.shape[0], )
        images = cam.get_frame_sequence(n_frames) 

    data_dict = {
        "image_stack": images

    }

    with h5py.File(run_dir / "ptc.h5", "w-") as f:
        f.create_dataset("dark", data=dark)
        f.create_dataset("light", data=light)
        f.create_dataset("gain_db", data=g_l)
        f.create_dataset("exposure_us", data=e_l)
        f.attrs["axes"] = "gain, exposure, frame, y, x"
    print(f"Saved {run_dir / 'ptc.h5'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--gain", type=float, required=True, help="gain [dB]")
    ap.add_argument("--max-exposure", type=float, required=True, help="longest exposure [us]")
    ap.add_argument("--min-exposure", type=float, default=20.0, help="shortest exposure [us]")
    ap.add_argument("--n-exp", type=int, default=25, help="number of exposures")
    ap.add_argument("--frames", type=int, default=2, help="frames per point")
    ap.add_argument("--cam", type=int, default=1, help="camera index")
    ap.add_argument("--fov", type=float, default=11.5, help="horizontal FOV [mm]")
    ap.add_argument("--preview", action="store_true",
                    help="show the max-exposure test frames before asking to start")
    a = ap.parse_args()
    main(a.run_dir, a.gain, a.max_exposure, a.min_exposure, a.n_exp, a.frames, a.cam, a.fov,
         preview=a.preview)