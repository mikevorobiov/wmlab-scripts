"""
Examples of labtools.provenance usage.
Run from anywhere:  python scratch/2026-10-22_provenance_examples.py /path/to/data_root
"""

import pathlib
import sys

import numpy as np

from labtools.provenance import file_ref, load_param, load_provenance, stamp

data_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")


# --------------------------------------------------------------------------- #
# 1. Acquisition run: scalars, a short array (inline) and a long ramp (CSV)
# --------------------------------------------------------------------------- #
run_dir = data_root / "2026-10-22_Ar_iv_run003"

ramp_V = np.concatenate([np.linspace(-40, 20, 301), np.linspace(20, -40, 301)])
params = {
    "gas": "Ar",
    "pressure_setpoint_mTorr": 15.0,
    "scope": {"vdiv_V": 0.5, "tdiv_s": 1e-6, "termination_ohm": 50, "bwl": False},
    "bias_checkpoints_V": np.array([-40, -20, 0, 20]),   # 4 elements -> inline
    "ramp_V": ramp_V,                                     # 602 elements -> CSV
    "n_avg": np.int64(16),                                # numpy scalar -> plain int
}
stamp(run_dir, params, note="probe cleaned before run")

# ... acquisition loop would go here, using the same `params` dict ...
np.savetxt(run_dir / "iv_raw.csv", np.c_[ramp_V, np.tanh(ramp_V / 10)],
           delimiter=",", header="V,I")


# --------------------------------------------------------------------------- #
# 2. Analysis run: inputs referenced by path + hash, output in its own subfolder
# --------------------------------------------------------------------------- #
out_dir = run_dir / "analysis" / "fit_iv_v1"
fit_params = {
    "input": file_ref(run_dir / "iv_raw.csv"),
    "fit_window_V": [-30.0, 5.0],
    "model": "exp_retarding",
}
stamp(out_dir, fit_params)
# ... fit, save results into out_dir ...


# --------------------------------------------------------------------------- #
# 3. Reading a record back later
# --------------------------------------------------------------------------- #
rec = load_provenance(run_dir)                  # most recent record in the folder
print("commit:", rec["commit"][:10], "dirty:", rec["dirty"])
print("pressure:", rec["params"]["pressure_setpoint_mTorr"], "mTorr")

ramp_back = load_param(run_dir, rec["params"]["ramp_V"])   # hash-checked
assert np.array_equal(ramp_back, ramp_V)                    # bit-exact round trip
print("ramp restored:", ramp_back.shape, ramp_back.dtype)

fit_rec = load_provenance(out_dir)
src = load_param(out_dir, fit_rec["params"]["input"])       # raises if file changed
print("analysis input verified:", src.name)
