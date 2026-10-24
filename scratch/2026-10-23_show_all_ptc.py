"""
Plot photon transfer curves from one or more ptc_fit.py output folders on one figure.

Usage:
    python analyze/plot_ptc.py                         # every PTC under $LABDATA/derived
    python analyze/plot_ptc.py <folder>                # every PTC under <folder>, recursively
    python analyze/plot_ptc.py --match flir-cam1_ptc   # only paths containing this text
    options: --electrons  --save out.png  --show-saturated

A "PTC" is any folder holding ptc_summary.csv (written by ptc_fit.py). Folders
are searched recursively, and glob patterns work too (PowerShell does not expand
them for Python, so the script does).

--electrons  plot in e- (signal*K, noise*K). Curves taken at different gains, or in
             different units (12- vs 16-bit DN), then fall onto one comparable scale.
Filled markers: points used in the K fit. Lines: fitted model sqrt(a + S/K).
"""

import argparse
import csv
import glob
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_ptc_dirs(roots: list[str], match: str) -> list[pathlib.Path]:
    if not roots:
        if not os.environ.get("LABDATA"):
            sys.exit("No folder given and LABDATA is not set")
        roots = [str(pathlib.Path(os.environ["LABDATA"]) / "derived")]
    found = set()
    for r in roots:
        for p in (glob.glob(r, recursive=True) or [r]):
            p = pathlib.Path(p)
            if p.is_dir():
                found.update(f.parent for f in p.rglob("ptc_summary.csv"))
    dirs = sorted(d for d in found if match in str(d))
    if not dirs:
        sys.exit(f"No ptc_summary.csv found under {roots}" + (f" matching '{match}'" if match else ""))
    for d in dirs:
        print("  ", d)
    return dirs


ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="*", help="folders or glob patterns (default: $LABDATA/derived)")
ap.add_argument("--match", default="", help="keep only PTC folders whose path contains this")
ap.add_argument("--electrons", action="store_true")
ap.add_argument("--save", type=pathlib.Path)
ap.add_argument("--show-saturated", action="store_true", help="include points past saturation")
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(7, 5.5))
for d in find_ptc_dirs(a.roots, a.match):
    points = read_csv(d / "ptc_points.csv")
    for s in read_csv(d / "ptc_summary.csv"):
        K, icpt = float(s["K_e_per_dn"]), 0.0
        scale = K if a.electrons else 1.0
        rows = [p for p in points if p["run"] == s["run"]
                and (a.show_saturated or p["past_saturation"] != "True")]
        S = np.array([float(p["signal_dn"]) for p in rows])
        sig = np.sqrt([float(p["var_total"]) for p in rows])
        used = np.array([p["used_in_fit"] == "True" for p in rows])
        pos = S > 0

        label = f"{s['run']}  ({float(s['gain_db']):.0f} dB, K={K:.3g} e-/DN)"
        line, = ax.loglog(S[used] * scale, sig[used] * scale, "o", ms=4, label=label)
        ax.loglog(S[~used & pos] * scale, sig[~used & pos] * scale, "o", ms=4,
                  mfc="none", color=line.get_color())
        if np.isfinite(K):
            Sm = np.geomspace(S[pos].min(), S[pos].max(), 200)
            ax.loglog(Sm * scale, np.sqrt(icpt + Sm / K) * scale, "-", lw=1, color=line.get_color())

unit = "e-" if a.electrons else "DN"
if a.electrons:
    x = np.array(ax.get_xlim())
    ax.loglog(x, np.sqrt(x), "k:", lw=1, label="shot noise only, sqrt(N)")
ax.set_xlabel(f"dark-subtracted signal [{unit}]")
ax.set_ylabel(f"temporal noise [{unit}]")
ax.set_title("Photon transfer curves (filled: used in K fit)")
ax.legend(fontsize=7)
ax.grid(which="both", alpha=0.2)
fig.tight_layout()
if a.save:
    fig.savefig(a.save, dpi=150)
else:
    plt.show()
