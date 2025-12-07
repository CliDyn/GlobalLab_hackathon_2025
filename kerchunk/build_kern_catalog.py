#!/usr/bin/env python3

"""Kerchunk each requested FESOM variable separately and merge refs."""

from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path
from typing import Dict, List

import fsspec
import numpy as np
import xarray as xr
from tqdm import tqdm

from kerchunk import hdf
import warnings
import zarr
warnings.filterwarnings(
    "ignore",
    category=zarr.errors.ZarrUserWarning,
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
)

DEFAULT_EPOCH = np.datetime64("1940-01-01T00:00:00")


def encode_value(v):
    if isinstance(v, (bytes, bytearray)):
        raw = v
    elif isinstance(v, memoryview):
        raw = v.tobytes()
    elif hasattr(v, "tobytes"):
        raw = v.tobytes()
    elif hasattr(v, "to_bytes"):
        raw = v.to_bytes()
    else:
        try:
            raw = memoryview(v).tobytes()
        except TypeError:
            return v

    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return "base64:" + base64.b64encode(raw).decode("ascii")


def get_time_seconds(nc_path: str, epoch=DEFAULT_EPOCH, time_var: str = "time"):
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        vals = ds[time_var].values
    return ((vals - epoch) / np.timedelta64(1, "s")).astype("int32")


def build_time_refs(time_sequences: List[np.ndarray], chunk_hint: int, time_var: str = "time"):
    combined = np.concatenate(time_sequences)
    ds_time = xr.Dataset({time_var: (time_var, combined)})
    ds_time[time_var].attrs["units"] = f"seconds since {str(DEFAULT_EPOCH)[:10]}"
    ds_time[time_var].encoding = {}

    store: Dict[str, bytes] = {}
    ds_time.to_zarr(
        store,
        encoding={
            time_var: {
                "dtype": "i4",
                "chunks": (chunk_hint,),
            }
        },
        mode="w",
        consolidated=False,
        zarr_format=2,
    )
    return {k: encode_value(v) for k, v in store.items() if k.startswith(f"{time_var}/")} 


def fix_time(single_ref, nc_path: str, time_var: str = "time"):
    ds_time = xr.Dataset({time_var: xr.DataArray(get_time_seconds(nc_path, time_var=time_var), dims=(time_var,))})
    mem_store: Dict[str, bytes] = {}
    ds_time.to_zarr(
        mem_store,
        encoding={
            time_var: {
                "dtype": "i4",
            }
        },
    )

    patched_refs = dict(single_ref["refs"])
    for key, value in mem_store.items():
        if key.startswith(f"{time_var}/"):
            patched_refs[key] = encode_value(value)

    return {
        "version": single_ref.get("version", 1),
        "templates": single_ref.get("templates", {}),
        "refs": patched_refs,
    }


def combine_variable_refs(var_name: str, files: List[str], time_var: str = "time"):
    combined = None
    time_sequences = []
    time_offset = 0
    zarray_meta = None

    for f in tqdm(files, desc=f"kerchunk {var_name}", unit="file"):
        try:
            with fsspec.open(f) as inf:
                single = hdf.SingleHdf5ToZarr(inf, f, inline_threshold=100).translate()
            single = fix_time(single, f, time_var=time_var)
            times = get_time_seconds(f, time_var=time_var)
        except Exception as exc:
            print(f"[WARN] Failed to kerchunk {var_name} file '{f}': {exc}")
            continue

        time_sequences.append(times)

        if combined is None:
            combined = {"version": single["version"], "templates": single.get("templates", {}), "refs": {}}

        for key, value in single["refs"].items():
            if key.startswith("time/"):
                continue
            if key == f"{var_name}/.zarray":
                zarray_meta = json.loads(value)
                continue
            if key.startswith(f"{var_name}/") and not key.endswith(".zattrs"):
                suffix = key.split("/", 1)[1]
                parts = suffix.split(".", 1)
                try:
                    idx = int(parts[0])
                except ValueError:
                    combined["refs"][key] = value
                    continue
                new_idx = idx + time_offset
                new_key = f"{var_name}/{new_idx}"
                if len(parts) > 1:
                    new_key += "." + parts[1]
                combined["refs"][new_key] = value
            else:
                combined["refs"][key] = value

        time_offset += len(times)

    if combined is None:
        raise RuntimeError(f"No references were produced for {var_name}; all input files failed.")

    if zarray_meta:
        zarray_meta["shape"][0] = time_offset
        chunks = zarray_meta.get("chunks")
        if chunks:
            chunks[0] = min(chunks[0], time_offset)
            zarray_meta["chunks"] = chunks
        combined["refs"][f"{var_name}/.zarray"] = encode_value(json.dumps(zarray_meta).encode("utf-8"))

    return combined, time_sequences


def merge_refs(per_var_refs, time_refs):
    merged = {"version": per_var_refs[0]["version"], "templates": per_var_refs[0].get("templates", {}), "refs": {}}

    for ref in per_var_refs:
        for key, value in ref["refs"].items():
            merged["refs"][key] = value

    merged["refs"].update(time_refs)
    return merged


def main(data_dir: Path, out_json: Path, variables: List[str], per_var_dir: Path, file_pattern: str, time_var: str):
    per_var_refs: List[dict] = []
    global_time_sequences = None
    reference_time_len = None

    per_var_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        pattern = file_pattern.format(var=var)
        files = sorted(glob.glob(str(data_dir / pattern)))
        if not files:
            continue
        refs, var_times = combine_variable_refs(var, files, time_var=time_var)
        total_len = sum(seq.shape[0] for seq in var_times)

        # Establish global time axis from the first successful variable
        if global_time_sequences is None:
            global_time_sequences = var_times
            reference_time_len = total_len
            per_var_refs.append(refs)
        else:
            # Enforce consistent time length across variables in the merged catalog
            if total_len != reference_time_len:
                print(
                    f"[WARN] Skipping variable '{var}' in merged catalog: "
                    f"time length {total_len} != reference {reference_time_len} (likely missing/corrupted files)."
                )
            else:
                per_var_refs.append(refs)

        per_var_path = per_var_dir / f"{var}_combined.json"
        with open(per_var_path, "w") as f:
            json.dump(refs, f)
        print(f"Wrote per-variable refs to {per_var_path}")

    if not per_var_refs:
        raise RuntimeError("No variables were processed; check input glob patterns.")

    time_refs = build_time_refs(global_time_sequences, chunk_hint=global_time_sequences[0].shape[0], time_var=time_var)
    merged = merge_refs(per_var_refs, time_refs)

    total_time = sum(seq.shape[0] for seq in global_time_sequences)
    per_file_lengths = {seq.shape[0] for seq in global_time_sequences}

    for key, value in list(merged["refs"].items()):
        if not key.endswith("/.zarray"):
            continue
        try:
            meta = json.loads(value)
        except Exception:
            continue
        shape = meta.get("shape")
        if not shape or shape[0] not in per_file_lengths:
            continue
        shape[0] = total_time
        meta["shape"] = shape
        chunks = meta.get("chunks")
        if chunks:
            chunks[0] = min(chunks[0], total_time)
            meta["chunks"] = chunks
        merged["refs"][key] = encode_value(json.dumps(meta).encode("utf-8"))

    # Clean up problematic time-related helpers so CF decoding focuses only on the
    # primary time coordinate. This avoids issues with NaN bounds variables.
    if time_var:
        # 1) Drop auxiliary time variables that are not needed for analysis
        drop_prefixes = [
            f"{time_var}_bounds/",
            "time_instant/",
            "time_instant_bounds/",
            "time_centered/",
            "time_centered_bounds/",
        ]
        for k in list(merged["refs"].keys()):
            if any(k.startswith(pref) for pref in drop_prefixes):
                merged["refs"].pop(k, None)

        # 2) Strip 'bounds' attribute from the primary time coordinate, so
        # xarray does not try to decode now-missing bounds.
        zattrs_key = f"{time_var}/.zattrs"
        zattrs_raw = merged["refs"].get(zattrs_key)
        if zattrs_raw is not None:
            try:
                zattrs = json.loads(zattrs_raw)
            except Exception:
                zattrs = None
            if isinstance(zattrs, dict) and "bounds" in zattrs:
                zattrs.pop("bounds", None)
                merged["refs"][zattrs_key] = encode_value(json.dumps(zattrs).encode("utf-8"))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(merged, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Kerchunk FESOM variables and merge refs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/work/ab0995/ICCP_AWI_hackthon_2025/TCO95L91-CORE2-ctl1950d/outdata/fesom"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("./FESOM_13_tropo_age_interpolated_05_deg_3d_interfaces.json"),
    )
    parser.add_argument(
        "--per-var-dir",
        type=Path,
        default=Path("./per_variable_refs"),
        help="Directory to store per-variable JSON refs",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=["v1-31", "temp1-31"],
        help="Variables to process",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="{var}.fesom.*.nc",
        help="Glob pattern (relative to data-dir) for input files; use {var} as placeholder for variable name",
    )
    parser.add_argument(
        "--time-var",
        type=str,
        default="time",
        help="Name of the time coordinate variable in the input NetCDF files (e.g. time, time_counter)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.out_json, args.vars, args.per_var_dir, args.pattern, args.time_var)
