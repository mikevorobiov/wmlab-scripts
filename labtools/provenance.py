"""
Provenance stamping for lab scripts.

Call stamp(outdir, params) once at the start of every script run. It writes into
`outdir`:

    provenance_<tag>.json     commit, dirty flag, host, script, argv, params, ...
    uncommitted_<tag>.diff    `git diff HEAD`, only if the working tree is dirty
    script_<tag>.py           copy of the running script (covers untracked scripts)
    param_<tag>_<key>.csv     one CSV per large array parameter (> INLINE_MAX elements)

<tag> is a timestamp (YYYYmmddTHHMMSS), so repeated runs into the same folder
never overwrite each other's records.

Requires the repo to be installed in editable mode (`pip install -e .`) so that
REPO below resolves to the git working tree.
"""

import datetime
import hashlib
import json
import pathlib
import re
import shutil
import socket
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
INLINE_MAX = 64  # arrays with more elements than this go to a sidecar CSV


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {REPO}:\n{r.stderr.strip()}\n"
            "Is labtools installed with `pip install -e .` from inside the repo?")
    return r.stdout.strip()


def _sha256(b):
    return hashlib.sha256(b).hexdigest()[:16]


def file_ref(path):
    """Reference an input file (calibration table, background frame, ...) by
    absolute path plus content hash. Use this for anything loaded from disk."""
    path = pathlib.Path(path).resolve()
    return {"__file__": str(path), "sha256": _sha256(path.read_bytes())}


def _safe(key):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def _prepare(obj, key, outdir, tag):
    """Return a JSON-safe version of obj; write large arrays to sidecar CSVs."""
    if isinstance(obj, dict):
        return {str(k): _prepare(v, f"{key}.{k}" if key else str(k), outdir, tag)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare(v, f"{key}.{i}", outdir, tag) for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        if obj.size <= INLINE_MAX:
            return obj.tolist()
        if obj.ndim > 2:
            raise TypeError(f"'{key}': {obj.ndim}-D array can't go in CSV; "
                            "save it as data and pass file_ref(path) instead")
        name = f"param_{tag}_{_safe(key)}.csv"
        path = outdir / name
        if obj.dtype == bool:
            arr, fmt = obj.astype(int), "%d"
        elif np.issubdtype(obj.dtype, np.integer):
            arr, fmt = obj, "%d"
        elif np.issubdtype(obj.dtype, np.floating):
            arr, fmt = obj, "%.17g"          # exact float64 round-trip
        else:
            raise TypeError(f"'{key}': unsupported array dtype {obj.dtype}")
        np.savetxt(path, arr, fmt=fmt, delimiter=",",
                   header=f"key={key} shape={list(obj.shape)} dtype={obj.dtype}")
        return {"__csv__": name, "shape": list(obj.shape), "dtype": str(obj.dtype),
                "sha256": _sha256(path.read_bytes())}
    if isinstance(obj, np.generic):          # np.float64, np.int64, np.bool_
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"Can't serialize {type(obj).__name__} at '{key}' in provenance")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def stamp(outdir, params=None, note=""):
    """Record what produced the contents of `outdir`. Call once, at run start."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t = datetime.datetime.now()
    tag = base = t.strftime("%Y%m%dT%H%M%S")
    n = 1
    while (outdir / f"provenance_{tag}.json").exists():   # two runs in one second
        n += 1
        tag = f"{base}_{n}"

    script = pathlib.Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    status = _git("status", "--porcelain")

    info = {
        "tag": tag,
        "time": t.isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "repo": str(REPO),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "untracked": _git("ls-files", "--others", "--exclude-standard").splitlines(),
        "script": str(script) if script else None,
        "argv": sys.argv[1:],
        "note": note,
    }
    # params last, so a serialization error fails before anything is half-written
    info["params"] = _prepare(params or {}, "", outdir, tag)

    if info["dirty"]:
        (outdir / f"uncommitted_{tag}.diff").write_text(_git("diff", "HEAD"))
    if script and script.suffix == ".py" and script.is_file():
        shutil.copy2(script, outdir / f"script_{tag}.py")

    (outdir / f"provenance_{tag}.json").write_text(json.dumps(info, indent=2))
    return info


def load_provenance(outdir, tag=None):
    """Load a provenance record; the most recent one if tag is None."""
    outdir = pathlib.Path(outdir)
    if tag is None:
        files = sorted(outdir.glob("provenance_*.json"))
        if not files:
            raise FileNotFoundError(f"No provenance records in {outdir}")
        path = files[-1]
    else:
        path = outdir / f"provenance_{tag}.json"
    return json.loads(path.read_text())


def load_param(outdir, entry, verify=True):
    """Turn a params entry back into a value: CSV refs become arrays,
    file refs are hash-checked, everything else is returned as-is."""
    outdir = pathlib.Path(outdir)
    if isinstance(entry, dict) and "__csv__" in entry:
        path = outdir / entry["__csv__"]
        if verify and _sha256(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"{path.name}: hash mismatch, file was modified")
        a = np.loadtxt(path, delimiter=",", ndmin=len(entry["shape"]))
        return a.astype(entry["dtype"]).reshape(entry["shape"])
    if isinstance(entry, dict) and "__file__" in entry:
        path = pathlib.Path(entry["__file__"])
        if verify and _sha256(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"{path}: hash mismatch, input file changed since the run")
        return path
    return entry
