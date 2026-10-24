"""
Find data files, load them, and run the same processing on each one.

Three pieces, meant to be used together:

    find_files()   search a folder tree for files matching a name pattern
                   (optionally only runs whose provenance record fits conditions)
    load()         read one file (HDF5, CSV or JSON) into plain Python/NumPy objects
    save_h5()      write a nested dict to HDF5; load() gives the same dict back
    process_all()  apply your own function to every file and collect the results

Example
-------
    files = find_files(pattern="ptc.h5", match="2026-10-23")
    files = find_files(pattern="ptc.h5", where={"params.gain_db": 34})
    results, errors = process_all(files, mean_light_level)

Status: SCRATCH. Once it has proven itself, move it to labtools/datafiles.py so
other scripts can `from labtools.datafiles import find_files, load, process_all`.
(A file whose name starts with a date cannot be imported as a module.)
"""

import csv
import json
import os
import pathlib
import traceback

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Finding files
# --------------------------------------------------------------------------- #

def default_root() -> pathlib.Path:
    """The raw-data folder, taken from the LABDATA environment variable."""
    labdata = os.environ.get("LABDATA")
    if labdata is None:
        raise RuntimeError("LABDATA is not set; pass a folder to find_files() instead")
    return pathlib.Path(labdata) / "raw"


def find_files(root=None, pattern="*.h5", match="", where=None) -> list[pathlib.Path]:
    """Return all files under `root` whose name fits `pattern`, sorted by path.

    root     folder to search, including all subfolders (default: $LABDATA/raw)
    pattern  file-name pattern, e.g. "ptc.h5", "*.csv", "provenance_*.json"
    match    keep only files whose full path contains this text,
             e.g. a date "2026-10-23" or a run tag "FLIR_flat"
    where    keep only files whose run's provenance record fits ALL these
             conditions; see provenance_matches() for how they are written
    """
    if root is None:
        root = default_root()
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    found = []
    for path in root.rglob(pattern):              # rglob = search all subfolders
        if path.is_file() and match in str(path):
            found.append(path)

    # Sorting by path also sorts by date, because run folders start with YYYY-MM-DD.
    found.sort()
    print(f"Found {len(found)} file(s) matching '{pattern}' under {root}")

    if where:
        found = filter_by_provenance(found, where)
    return found


# --------------------------------------------------------------------------- #
# 1b. Selecting files by what their provenance record says
# --------------------------------------------------------------------------- #

def get_field(record: dict, dotted_key: str):
    """Look up a nested value with a dotted key.

    get_field(record, "params.camera.serial_number")
    is the same as record["params"]["camera"]["serial_number"].
    Raises KeyError if any part of the path is missing.
    """
    value = record
    for part in dotted_key.split("."):
        value = value[part]
    return value


def value_matches(actual, wanted) -> bool:
    """Compare one provenance value with one condition.

    wanted can be
      - a function:  called with the actual value, must return True/False,
                     e.g. lambda gain: gain >= 20   or   lambda s: "ptc" in s
      - a number:    equal within floating-point rounding only. Values read back
                     from the camera are quantized (34 dB may read 33.996), so
                     test those with a range: lambda g: abs(g - 34) < 0.1
      - anything else (text, True/False, lists): must be exactly equal
    """
    if callable(wanted):
        return bool(wanted(actual))

    # True/False first: Python treats True as 1, so without this a condition
    # True would also match a stored 1.
    if isinstance(actual, bool) or isinstance(wanted, bool):
        return type(actual) is type(wanted) and actual == wanted

    both_numbers = isinstance(actual, (int, float)) and isinstance(wanted, (int, float))
    if both_numbers:
        return bool(np.isclose(actual, wanted, rtol=1e-9, atol=1e-9))

    return actual == wanted


def provenance_matches(record: dict, where: dict) -> bool:
    """True if the record fits every condition in `where`.

    where maps dotted keys to conditions, for example
        {"params.gain_db": 34,
         "script": lambda s: "ptc_acquire" in s,
         "dirty": False}
    A record that lacks one of the keys does not match.
    """
    for dotted_key, wanted in where.items():
        try:
            actual = get_field(record, dotted_key)
        except (KeyError, TypeError):
            return False
        if not value_matches(actual, wanted):
            return False
    return True


def filter_by_provenance(files: list[pathlib.Path], where: dict) -> list[pathlib.Path]:
    """Keep the files whose run folder holds a provenance record fitting `where`.

    Files without any provenance record are dropped, and how many were dropped
    is printed, so a missing record never disappears unnoticed.
    """
    kept = []
    without_record = 0

    for path in files:
        record = load_provenance(path)
        if record is None:
            without_record += 1
        elif provenance_matches(record, where):
            kept.append(path)

    print(f"  {len(kept)} of {len(files)} fit the provenance conditions")
    if without_record:
        print(f"  {without_record} file(s) skipped: no provenance record in their folder")
    return kept


# --------------------------------------------------------------------------- #
# 2. Loading one file
# --------------------------------------------------------------------------- #

def load_h5(path, names=None) -> dict:
    """Read an HDF5 file into a nested dict: groups become dicts, datasets
    become arrays (or plain numbers/text for single values).

    names  list of datasets to read, written as paths, e.g. ["light"] or
           ["fit/K"]; None reads everything. Reading only what you need saves
           memory with large image stacks.

    Attributes come back under an "attrs" key at each level (see save_h5):
        data["attrs"]["gain_db"]            file attribute
        data["fit"]["attrs"]["method"]      attribute of group /fit
        data["attrs"]["light.units"]        attribute "units" of dataset /light
    A level without attributes has no "attrs" key.

    This is the exact mirror of save_h5(): save a dict, load it, get it back.
    """
    import h5py                                   # imported here: only needed for .h5

    with h5py.File(path, "r") as f:
        data = read_group(f, names, prefix="")

    if names is not None:
        missing = [name for name in names if not has_path(data, name)]
        if missing:
            raise KeyError(f"{pathlib.Path(path).name} has no dataset(s) {missing}")
    return data


def read_group(group, names, prefix) -> dict:
    """Read one HDF5 group, and everything below it, into a dict."""
    import h5py

    result = {}
    attrs = read_attrs(group)                     # the group's own attributes

    for key, item in group.items():
        full_name = prefix + key                  # e.g. "fit/K"

        if isinstance(item, h5py.Group):
            subgroup = read_group(item, names, prefix=full_name + "/")
            # Keep a group if any data in it was wanted (or if reading everything).
            has_data = any(k != "attrs" for k in subgroup)
            if has_data or names is None:
                result[key] = subgroup

        elif names is None or full_name in names:
            result[key] = read_dataset(item)
            # Dataset attributes are listed with the group as "<dataset>.<name>".
            for attr_name, value in read_attrs(item).items():
                attrs[f"{key}.{attr_name}"] = value

    if attrs:
        result["attrs"] = attrs
    return result


def read_attrs(item) -> dict:
    """All attributes of one group, dataset or file, as plain Python values."""
    attrs = {}
    for name, value in item.attrs.items():
        attrs[name] = to_python(value)
    return attrs


def read_dataset(dataset):
    """Read one dataset, turning text back into str and single values back
    into plain Python numbers."""
    import h5py

    is_text = h5py.check_string_dtype(dataset.dtype) is not None
    if is_text:
        value = dataset.asstr()[()]               # str, or array of str
    else:
        value = dataset[()]                       # [()] reads the whole dataset
    return to_python(value)


def to_python(value):
    """numpy.float64(3.0) -> 3.0 and bytes -> str; arrays are left as they are."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def has_path(data: dict, path: str) -> bool:
    """True if data["a"]["b"] exists for path "a/b"."""
    value = data
    for part in path.split("/"):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def load_csv(path) -> dict:
    """Read a CSV with a header row into {column name: NumPy array}.

    Numeric columns become float arrays, "True"/"False" columns become bool
    arrays, and anything else stays as an array of strings.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    columns = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        columns[name] = convert_column(values)
    return columns


def convert_column(values: list[str]) -> np.ndarray:
    """Turn a list of strings into the most specific array type that fits."""
    if all(v in ("True", "False") for v in values):
        return np.array([v == "True" for v in values])
    try:
        return np.array(values, dtype=float)      # also handles "nan" and "1e-3"
    except ValueError:
        return np.array(values)                   # leave as text


def load_json(path) -> dict:
    """Read a JSON file, e.g. a provenance record."""
    with open(path) as f:
        return json.load(f)


# Which function reads which file type. Add new types here.
LOADERS = {
    ".h5": load_h5,
    ".hdf5": load_h5,
    ".csv": load_csv,
    ".json": load_json,
}


def load(path, **options) -> dict:
    """Read any supported file. Extra options go to the specific loader,
    e.g. load(path, names=["light"]) for HDF5."""
    path = pathlib.Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader for '{path.suffix}' files ({path.name})")
    return loader(path, **options)


def load_provenance(path) -> dict | None:
    """The newest provenance record in the same folder as `path`, or None.
    Provenance files are named provenance_<timestamp>.json, so the last one
    in sorted order is the newest."""
    records = sorted(pathlib.Path(path).parent.glob("provenance_*.json"))
    if not records:
        return None
    return load_json(records[-1])


# --------------------------------------------------------------------------- #
# 2b. Saving a dict to HDF5
# --------------------------------------------------------------------------- #

def save_h5(path, data: dict, overwrite=False) -> None:
    """Write a nested dict to an HDF5 file.

    Rules
      - a dict becomes a group, so nesting in the dict = nesting in the file
      - everything else becomes a dataset: arrays, lists, numbers, True/False, text
      - the key "attrs" is special: it holds ATTRIBUTES of that level, i.e. small
        metadata (units, descriptions, settings) readable without loading data
          "attrs" at the top level      -> file attributes
          "attrs" inside a nested dict  -> attributes of that group
          "<dataset>.<name>" in "attrs" -> attribute <name> of that dataset

    Example
        save_h5(out_dir / "result.h5", {
            "images": stack,
            "fit": {
                "K_e_per_dn": 0.0067,
                "attrs": {"method": "slope of var vs signal"},
            },
            "attrs": {
                "description": "PTC fit, 34 dB",
                "images.units": "DN",
                "images.axes": "frame, y, x",
            },
        })
    gives datasets /images and /fit/K_e_per_dn, a file attribute "description",
    a group attribute "method" on /fit, and "units" and "axes" on /images.

    overwrite=False refuses to replace an existing file, so results are never
    lost by accident. Pass overwrite=True only for files you mean to regenerate.
    """
    import h5py

    # Check everything BEFORE creating the file. Otherwise a bad value halfway
    # through would leave a half-written file behind, which "w-" would then
    # refuse to replace on the next, corrected attempt.
    validate(data, location="")

    mode = "w" if overwrite else "w-"             # "w-" fails if the file exists
    with h5py.File(path, mode) as f:
        write_group(f, data, location="")


def validate(data: dict, location: str) -> None:
    """Walk the dict the same way write_group() does, raising on the first
    value that cannot be stored, without writing anything."""
    for key, value in data.items():
        here = f"{location}/{key}"
        if key == "attrs":
            validate_attrs(value, data, here)
            continue
        check_key(key, here)
        if isinstance(value, dict):
            validate(value, location=here)
        else:
            to_storable(value, here)


def validate_attrs(attrs, siblings: dict, here: str) -> None:
    """Check one "attrs" dict. `siblings` is the dict it sits in, used to
    confirm that "<dataset>.<name>" refers to a dataset that is really there."""
    if not isinstance(attrs, dict):
        raise TypeError(f"{here}: must be a dict of attribute names to values")

    for name, value in attrs.items():
        if not isinstance(name, str) or name == "":
            raise TypeError(f"{here}: attribute names must be non-empty strings, got {name!r}")
        if isinstance(value, dict):
            raise TypeError(f"{here}/{name}: attributes cannot be dicts; use a group instead")
        to_storable(value, f"{here}/{name}")

        if "." in name:
            dataset_name = name.split(".", 1)[0]
            target = siblings.get(dataset_name)
            if target is None or isinstance(target, dict):
                raise KeyError(f"{here}/{name}: '{dataset_name}' is not a dataset at this "
                               "level (a dot in an attribute name means <dataset>.<attribute>)")


def write_group(group, data: dict, location: str) -> None:
    """Write one dict into one HDF5 group, recursing into nested dicts.
    `location` is only used to say where a problem is in error messages."""
    for key, value in data.items():
        if key == "attrs":
            continue                              # written last, once datasets exist
        here = f"{location}/{key}"

        if isinstance(value, dict):
            subgroup = group.create_group(key)
            write_group(subgroup, value, location=here)
        else:
            array, dtype = to_storable(value, here)
            group.create_dataset(key, data=array, dtype=dtype)

    write_attrs(group, data.get("attrs", {}), location)


def write_attrs(group, attrs: dict, location: str) -> None:
    """Attach attributes to the group, or to one of its datasets for names
    written as "<dataset>.<attribute>"."""
    for name, value in attrs.items():
        if "." in name:
            dataset_name, attribute_name = name.split(".", 1)
            target = group[dataset_name]
        else:
            attribute_name = name
            target = group

        array, dtype = to_storable(value, f"{location}/attrs/{name}")
        target.attrs.create(attribute_name, data=array, dtype=dtype)


def check_key(key, here: str) -> None:
    """Reject keys that would silently change the file layout."""
    if not isinstance(key, str) or key == "":
        raise TypeError(f"{here}: keys must be non-empty strings, got {key!r}")
    if "/" in key:
        # HDF5 would read "a/b" as group a containing b: a surprise layout.
        raise ValueError(f"{here}: '/' is not allowed in a key; use a nested dict instead")
    if "." in key:
        # A dot would make "<key>.<name>" in attrs ambiguous.
        raise ValueError(f"{here}: '.' is not allowed in a dataset or group name; use '_'")


def to_storable(value, here: str):
    """Return (data, dtype) that h5py can store. dtype is None except for text."""
    import h5py

    if value is None:
        raise TypeError(f"{here}: None cannot be stored in HDF5; leave the key out instead")

    if isinstance(value, str):
        return value, h5py.string_dtype()

    ragged_message = (f"{here}: cannot store {type(value).__name__} with mixed or "
                      "ragged contents (e.g. lists of different lengths); "
                      "split it into separate datasets")
    try:
        array = np.asarray(value)
    except ValueError:                            # NumPy refuses ragged lists outright
        raise TypeError(ragged_message) from None

    if array.dtype.kind == "U":                   # array of text
        return array.astype(object), h5py.string_dtype()

    if array.dtype == object:                     # anything NumPy could only box up
        raise TypeError(ragged_message)
    return array, None


def check_round_trip(folder) -> None:
    """Save a sample dict, load it back, and confirm nothing changed.
    Run this once with the real h5py on the lab PC:
        python scratch/2026-10-24_data_files.py --selftest
    """
    sample = {
        "images": np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
        "exposure_us": [10.0, 20.0, 40.0],
        "label": "flat, 34 dB",
        "n_frames": 50,
        "dirty": False,
        "fit": {
            "K_e_per_dn": 0.0067,
            "names": ["a", "bb"],
            "deeper": {"x": 1.5},
            "attrs": {"method": "slope", "range": [0.05, 0.7]},
        },
        "attrs": {
            "note": "self-test",
            "gain_db": 34.0,
            "images.units": "DN",
            "images.axes": "frame, y, x",
        },
    }
    path = pathlib.Path(folder) / "round_trip_check.h5"
    save_h5(path, sample, overwrite=True)
    loaded = load_h5(path)

    # attributes at all three levels: file, group, dataset
    assert loaded["attrs"]["note"] == "self-test" and loaded["attrs"]["gain_db"] == 34.0
    assert loaded["attrs"]["images.units"] == "DN"
    assert loaded["attrs"]["images.axes"] == "frame, y, x"
    assert loaded["fit"]["attrs"]["method"] == "slope"
    assert list(loaded["fit"]["attrs"]["range"]) == [0.05, 0.7]
    assert "attrs" not in loaded["fit"]["deeper"], "empty attrs should not appear"

    # data
    assert np.array_equal(loaded["images"], sample["images"])
    assert loaded["images"].dtype == np.uint16, "dtype changed"
    assert np.array_equal(loaded["exposure_us"], sample["exposure_us"])
    assert loaded["label"] == "flat, 34 dB" and isinstance(loaded["label"], str)
    assert loaded["n_frames"] == 50 and loaded["dirty"] is False
    assert loaded["fit"]["K_e_per_dn"] == 0.0067
    assert list(loaded["fit"]["names"]) == ["a", "bb"]
    assert loaded["fit"]["deeper"]["x"] == 1.5
    # reading one dataset keeps the attributes of the groups on its path,
    # but not the attributes of datasets that were not read
    partial = load_h5(path, names=["fit/deeper/x"])
    assert partial["fit"]["deeper"] == {"x": 1.5}
    assert partial["fit"]["attrs"]["method"] == "slope"
    assert "images.units" not in partial["attrs"]
    path.unlink()
    print("Round trip OK: save_h5 and load_h5 are exact mirrors")


# --------------------------------------------------------------------------- #
# 3. Batch processing
# --------------------------------------------------------------------------- #

def process_all(files, function, stop_on_error=False):
    """Call function(path) for every file and collect what it returns.

    The function receives the file PATH, not loaded data, so it decides itself
    what to load (and can read only part of a large file).

    Returns (results, errors):
        results  {path: return value} for files that worked
        errors   {path: error message} for files that failed

    By default one bad file does not stop the batch, but every failure is
    printed at the end, so nothing fails silently. stop_on_error=True raises
    at the first failure instead, which is handy while debugging the function.
    """
    results = {}
    errors = {}

    for number, path in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] {path.name}  ({path.parent.name})")
        try:
            results[path] = function(path)
        except Exception as error:
            if stop_on_error:
                raise
            errors[path] = f"{type(error).__name__}: {error}"
            traceback.print_exc()                 # full detail, right where it happened

    print(f"\nDone: {len(results)} succeeded, {len(errors)} failed")
    for path, message in errors.items():
        print(f"  FAILED {path}\n         {message}")
    return results, errors


# --------------------------------------------------------------------------- #
# Example: run this file directly to try it out
#   python scratch/2026-10-24_data_files.py
# --------------------------------------------------------------------------- #

def mean_light_level(path) -> float:
    """Example processing step: mean of all lit frames in a PTC or flat file."""
    data = load(path, names=["light"])            # read only the dataset we need
    return float(data["light"].mean())


if __name__ == "__main__":
    import sys
    import tempfile

    if "--selftest" in sys.argv:
        with tempfile.TemporaryDirectory() as folder:
            check_round_trip(folder)
        sys.exit()

    files = find_files(pattern="*.h5")
    results, errors = process_all(files, mean_light_level)

    for path, value in results.items():
        print(f"{path.parent.name:45s} mean light level {value:10.1f}")
