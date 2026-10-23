"""
Photon transfer curve (PTC) analysis: camera gain K, read noise, PRNU, saturation.

Method: Janesick, Photon Transfer (SPIE, 2007).
For every (gain, exposure) point, with dark-subtracted mean signal S [DN]:
    total temporal variance   var_total = var(A - B) / 2        (light pair)
    read (+dark) variance     var_read  = var(Ad - Bd) / 2      (dark pair)
    shot variance             var_shot  = var_total - var_read = S / K
Pair differences cancel fixed-pattern noise, so it never inflates the estimate.
Saturation is taken from the variance peak, which covers both ADC clipping and
full well filling below 4095 DN. K [e-/DN] comes from the shot-noise-limited
points between 5 and 70 % of that saturation signal.

Resulting single-pixel noise model, in DN, for a raw frame:
    sigma^2(S) = read_dn^2 + S / K + (prnu * S)^2
The PRNU term is spatial. It cancels in flat-fielded or ratio images, and it is
absent from frame-to-frame fluctuations of the same pixel.

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

from labtools.sensor import ADC_BITS, FULL_SCALE, lsb_shift

FIT_RANGE = (0.05, 0.70)        # fraction of the saturation signal where K is fitted


# --------------------------------------------------------------------------- #
# core (pure NumPy, no I/O)
# --------------------------------------------------------------------------- #


def central_roi(ny: int, nx: int, fraction: float) -> tuple[slice, slice]:
    """Central region covering `fraction` of each dimension (avoids vignetting)."""
    dy, dx = int(ny * fraction / 2), int(nx * fraction / 2)
    return slice(ny // 2 - dy, ny // 2 + dy), slice(nx // 2 - dx, nx // 2 + dx)


def temporal_var(stack: np.ndarray) -> float:
    """Pixel-averaged temporal variance of a (frame, y, x) float stack.
    Each frame's mean is removed first, so slow illumination drift is not counted.
    For two frames this equals var(A - B) / 2."""
    s = stack - stack.mean(axis=(1, 2), keepdims=True)
    return float(s.var(axis=0, ddof=1).mean())


def ptc_point(light: np.ndarray, dark: np.ndarray, full_scale: float) -> dict:
    """One PTC point from float (frame, y, x) stacks already cut to the ROI, in DN."""
    n = light.shape[0]
    offset = dark.mean()
    signal = light.mean() - offset
    var_total = temporal_var(light)
    var_read = temporal_var(dark)
    # spatial variance of the frame-averaged, dark-subtracted image, minus its temporal part
    flat = light.mean(axis=0) - dark.mean(axis=0)
    var_fpn = flat.var(ddof=1) - (var_total + var_read) / n
    return {
        "offset_dn": offset,
        "signal_dn": signal,
        "var_total": var_total,
        "var_read": var_read,
        "var_shot": var_total - var_read,
        "prnu": np.sqrt(max(var_fpn, 0.0)) / signal if signal > 0 else np.nan,
        "saturated": bool(light.max() >= full_scale),
        "dark_clipped": bool(dark.min() <= 0),
    }


def analyze(dark, light, gain_db, exposure_us, roi_fraction=0.5):
    """dark, light: uint16 (n_gain, n_exp, n_frames, y, x).
    Returns (points, summary): lists of dicts, ready for CSV."""
    if light.shape[2] < 2:
        raise ValueError("Need at least 2 frames per point")
    shift = lsb_shift(light)
    scale = float(1 << shift)
    full_scale = FULL_SCALE
    roi = (slice(None),) + central_roi(*light.shape[-2:], roi_fraction)

    points, summary = [], []
    for i in range(light.shape[0]):
        rows = []
        for j in range(light.shape[1]):
            p = ptc_point(light[i, j][roi].astype(np.float64) / scale,
                          dark[i, j][roi].astype(np.float64) / scale, full_scale)
            p.update(gain_db=gain_db[i, j], exposure_us=exposure_us[i, j])
            rows.append(p)

        S = np.array([r["signal_dn"] for r in rows])
        v_shot = np.array([r["var_shot"] for r in rows])
        v_total = np.array([r["var_total"] for r in rows])
        sat = np.array([r["saturated"] for r in rows])          # ADC top code in ROI
        offset = float(np.median([r["offset_dn"] for r in rows]))
        adc_span = full_scale - offset

        # Saturation of either kind compresses the noise: variance peaks, then drops.
        # Points above the peak signal are past saturation (Janesick's full well).
        # The peak is the last point BEFORE saturation, so the saturation signal is a
        # lower bound, good to one exposure step; use more --n-exp near the top to tighten it.
        k_peak = int(np.argmax(v_total))
        S_peak = float(S[k_peak])
        past = S > S_peak
        if not past.any():
            limit, span = "not reached", adc_span
        else:
            first_past = int(np.argmin(np.where(past, S, np.inf)))
            limit = "ADC" if sat[first_past] else "full well"
            span = S_peak

        use = (~sat) & (~past) & (S > FIT_RANGE[0] * span) & (S < FIT_RANGE[1] * span) & (v_shot > 0)
        for r, u, p in zip(rows, use, past):
            r["used_in_fit"] = bool(u)
            r["past_saturation"] = bool(p)

        K_i = S[use] / v_shot[use]
        K = float(np.median(K_i)) if use.sum() >= 3 else np.nan
        read_dn = float(np.sqrt(rows[int(np.argmin(exposure_us[i]))]["var_read"]))  # shortest exposure
        prnu = float(np.nanmedian([r["prnu"] for r, u in zip(rows, use) if u])) if use.any() else np.nan

        summary.append({
            "gain_db": float(np.median(gain_db[i])),
            "K_e_per_dn": K,
            "K_rel_spread": float(np.std(K_i) / K) if use.sum() >= 3 else np.nan,
            "read_noise_dn": read_dn,
            "read_noise_e": read_dn * K,
            "offset_dn": offset,
            "saturation_dn": span,
            "saturation_e": span * K,
            "saturation_limit": limit,
            "prnu": prnu,
            "n_fit_points": int(use.sum()),
            "lsb_shift": shift,
            "dark_clipped": any(r["dark_clipped"] for r in rows),
        })
        points.extend(rows)
    return points, summary


def noise_dn(signal_dn, K, read_noise_dn, prnu=0.0):
    """1-sigma single-pixel noise [DN] at dark-subtracted signal [DN].
    Use prnu=0 for frame-to-frame noise or for flat-fielded / ratio images."""
    S = np.clip(signal_dn, 0, None)
    return np.sqrt(read_noise_dn**2 + S / K + (prnu * S) ** 2)


# --------------------------------------------------------------------------- #
# I/O and plotting
# --------------------------------------------------------------------------- #

def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(f"{r[k]:.10g}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def plot(points, summary, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in summary:
        rows = [p for p in points if p.get("run") == s.get("run")
                and abs(p["gain_db"] - s["gain_db"]) < 0.05]
        S = np.array([p["signal_dn"] for p in rows])
        sig = np.sqrt([p["var_total"] for p in rows])
        used = np.array([p["used_in_fit"] for p in rows])
        line, = ax.loglog(S[used], sig[used], "o", label=f"{s['gain_db']:.0f} dB, "
                          f"K={s['K_e_per_dn']:.3g} e-/DN, read={s['read_noise_e']:.2g} e-")
        ax.loglog(S[~used & (S > 0)], sig[~used & (S > 0)], "o", mfc="none", color=line.get_color())
        if np.isfinite(s["K_e_per_dn"]):
            Sm = np.geomspace(max(S[S > 0].min(), 1e-1), S.max(), 200)
            ax.loglog(Sm, noise_dn(Sm, s["K_e_per_dn"], s["read_noise_dn"]), "-",
                      color=line.get_color(), lw=1)
    ax.set_xlabel("dark-subtracted signal [DN]")
    ax.set_ylabel("temporal noise [DN]")
    ax.set_title("Photon transfer curve (filled: used in K fit)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(out_dir: pathlib.Path, raw_dirs: list[pathlib.Path], roi_fraction: float = 0.5) -> None:
    import h5py
    from labtools.provenance import file_ref, stamp

    h5_files = [d / "ptc.h5" for d in raw_dirs]
    stamp(out_dir, {"inputs": [file_ref(h) for h in h5_files], "roi_fraction": roi_fraction,
                    "fit_range": list(FIT_RANGE), "adc_bits": ADC_BITS})

    points, summary = [], []
    for h5 in h5_files:                         # one file at a time keeps memory to one run
        with h5py.File(h5, "r") as f:
            p, s = analyze(f["dark"][...], f["light"][...],
                           f["gain_db"][...], f["exposure_us"][...], roi_fraction)
        for row in p + s:
            row["run"] = h5.parent.name
        points += p
        summary += s
    summary.sort(key=lambda s: s["gain_db"])

    write_csv(out_dir / "ptc_points.csv", points)
    write_csv(out_dir / "ptc_summary.csv", summary)
    plot(points, summary, out_dir / "ptc.png")

    for s in summary:
        print(f"{s['gain_db']:5.1f} dB  K={s['K_e_per_dn']:.4g} e-/DN (±{100*s['K_rel_spread']:.1f}%)  "
              f"read={s['read_noise_dn']:.3g} DN = {s['read_noise_e']:.3g} e-  "
              f"sat={s['saturation_e']:.4g} e- ({s['saturation_limit']})  PRNU={100*s['prnu']:.2f}%  "
              f"fit pts={s['n_fit_points']}")
        if s["dark_clipped"]:
            print("   WARNING: dark pixels at 0 DN; noise is clipped. Raise BlackLevel.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(pathlib.Path(sys.argv[1]), [pathlib.Path(p) for p in sys.argv[2:]])
