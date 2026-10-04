#!/usr/bin/env python3
"""
Detect bright targets in a Sentinel-1 IW GRD SAFE product.

Usage
-----
python src/detect.py /path/to/S1C_....SAFE \
    --output-dir results/S1C_....

The script:
  1. Finds the VV measurement TIFF and VV annotation XML in the SAFE.
  2. Reads a reduced-resolution VV image.
  3. Geolocates pixels using the Sentinel-1 annotation geolocation grid.
  4. Creates a land/sea mask and expands the land mask to exclude
     coastal pixels.
  5. Runs 2-D CA-CFAR over the sea.
  6. Groups detections into connected components.
  7. Writes candidate locations to CSV.
  8. Writes a PNG overview with detection locations.

This is intentionally a simple operational detector. Shape/area analysis
and VH validation are left for a later stage.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, find_objects, label, uniform_filter
from global_land_mask import globe

# geo.py is expected to be in the same src/ directory.
from geo import Sentinel1GeoLocator


# ----------------------------------------------------------------------
# Default parameters
# ----------------------------------------------------------------------

DEFAULT_SCALE = 16
DEFAULT_TRAIN_SIZE = 31
DEFAULT_GUARD_SIZE = 5
DEFAULT_THRESHOLD_DB = 8.0
DEFAULT_COAST_EXCLUSION_PIXELS = 5
DEFAULT_MIN_PIXELS = 3
DEFAULT_MAX_PIXELS = 1000


# ----------------------------------------------------------------------
# SAFE file discovery
# ----------------------------------------------------------------------

def find_vv_files(safe_dir: Path) -> tuple[Path, Path]:
    """Find VV measurement TIFF and matching VV annotation XML."""

    measurement = sorted(
        safe_dir.glob("measurement/*-vv-*.tiff")
    )

    annotation = sorted(
        safe_dir.glob("annotation/*-vv-*.xml")
    )

    if not measurement:
        raise FileNotFoundError(
            f"VV measurement TIFF not found in {safe_dir}"
        )

    # There can be several XML files containing VV in other directories.
    # The main annotation XML is directly under annotation/.
    main_annotation = [
        p for p in annotation
        if "calibration" not in p.parts
        and "noise" not in p.name.lower()
    ]

    if not main_annotation:
        raise FileNotFoundError(
            f"VV annotation XML not found in {safe_dir / 'annotation'}"
        )

    # Usually there is exactly one VV measurement and one main VV annotation.
    # If multiple exist, use the first deterministic result.
    return measurement[0], main_annotation[0]


def acquisition_time_from_name(safe_dir: Path) -> str:
    """Extract Sentinel-1 acquisition start time from SAFE name."""

    m = re.search(
        r"_(\d{8}T\d{6})_\d{8}T\d{6}_",
        safe_dir.name,
    )

    if m:
        return m.group(1)

    return ""


# ----------------------------------------------------------------------
# CFAR
# ----------------------------------------------------------------------

def run_cfar(
    power: np.ndarray,
    valid: np.ndarray,
    sea: np.ndarray,
    train_size: int,
    guard_size: int,
    threshold_db: float,
) -> np.ndarray:
    """Run a simple 2-D CA-CFAR."""

    if train_size <= guard_size:
        raise ValueError("train_size must be larger than guard_size")

    if train_size % 2 == 0 or guard_size % 2 == 0:
        raise ValueError("train_size and guard_size must be odd")

    x = power.copy()
    x[~valid] = 0.0

    valid_float = valid.astype(np.float32)

    # uniform_filter returns the local mean.
    total_mean = uniform_filter(
        x,
        size=train_size,
        mode="nearest",
    )

    total_count = uniform_filter(
        valid_float,
        size=train_size,
        mode="nearest",
    )

    guard_mean = uniform_filter(
        x,
        size=guard_size,
        mode="nearest",
    )

    guard_count = uniform_filter(
        valid_float,
        size=guard_size,
        mode="nearest",
    )

    total_area = train_size ** 2
    guard_area = guard_size ** 2

    training_sum = (
        total_mean * total_area
        - guard_mean * guard_area
    )

    training_count = (
        total_count * total_area
        - guard_count * guard_area
    )

    background = (
        training_sum
        / np.maximum(training_count, 1.0)
    )

    ratio = power / np.maximum(background, 1e-12)

    threshold = 10.0 ** (threshold_db / 10.0)

    detections = (
        sea
        & valid
        & (background > 0)
        & (ratio > threshold)
    )

    return detections, ratio


# ----------------------------------------------------------------------
# Candidate extraction
# ----------------------------------------------------------------------

def extract_candidates(
    detections: np.ndarray,
    ratio: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    """Group detections and return one representative point per group."""

    structure = np.ones((3, 3), dtype=np.uint8)

    labels, nlabels = label(
        detections,
        structure=structure,
    )

    objects = find_objects(labels)

    candidates = []

    for label_id, slc in enumerate(objects, start=1):

        if slc is None:
            continue

        region = labels[slc] == label_id
        n_pixels = int(region.sum())

        if n_pixels < min_pixels:
            continue

        if n_pixels > max_pixels:
            continue

        rows, cols = np.where(region)

        rows = rows + slc[0].start
        cols = cols + slc[1].start

        response = ratio[rows, cols]
        imax = int(np.argmax(response))

        row = int(rows[imax])
        col = int(cols[imax])

        if not np.isfinite(lat[row, col]) or not np.isfinite(lon[row, col]):
            continue

        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "row": row,
                "col": col,
                "latitude": float(lat[row, col]),
                "longitude": float(lon[row, col]),
                "cfar_db": float(
                    10.0 * np.log10(
                        max(float(ratio[row, col]), 1e-12)
                    )
                ),
                "pixels": n_pixels,
            }
        )

    return candidates


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def write_csv(
    path: Path,
    candidates: list[dict],
    safe_name: str,
    acquisition_time: str,
) -> None:
    """Write candidate table."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "product",
        "acquisition_time",
        "candidate_id",
        "row",
        "col",
        "latitude",
        "longitude",
        "cfar_db",
        "pixels",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for candidate in candidates:
            writer.writerow(
                {
                    "product": safe_name,
                    "acquisition_time": acquisition_time,
                    **candidate,
                }
            )


def write_png(
    path: Path,
    db: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    land: np.ndarray,
    valid: np.ndarray,
    candidates: list[dict],
) -> None:
    """Write an overview PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)

    sea = (~land) & valid & np.isfinite(db)

    if np.any(sea):
        vmin = float(np.nanpercentile(db[sea], 2))
        vmax = float(np.nanpercentile(db[sea], 98))
    else:
        vmin = float(np.nanpercentile(db, 2))
        vmax = float(np.nanpercentile(db, 98))

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(
        lon,
        lat,
        db,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        cmap='gray',
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Relative SAR intensity [dB]",
        pad=0.01,
    )

    # Coastline from the land mask.
    ax.contour(
        lon,
        lat,
        land.astype(float),
        levels=[0.5],
        linewidths=0.5,
    )

    if candidates:
        cand_lon = [c["longitude"] for c in candidates]
        cand_lat = [c["latitude"] for c in candidates]

        ax.scatter(
            cand_lon,
            cand_lat,
            s=35,
            facecolors="none",
            edgecolors="red",
            linewidths=1,
#            label=f"CFAR candidates ({len(candidates)})",
        )

#        ax.legend()

    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title("Sentinel-1 VV CA-CFAR detection")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Detect bright sea-surface targets in Sentinel-1 IW GRD."
    )

    parser.add_argument(
        "safe",
        type=Path,
        help="Sentinel-1 .SAFE directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<SAFE name>)",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=f"Read every Nth pixel (default: {DEFAULT_SCALE})",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=f"CA-CFAR training window (default: {DEFAULT_TRAIN_SIZE})",
    )

    parser.add_argument(
        "--guard-size",
        type=int,
        default=DEFAULT_GUARD_SIZE,
        help=f"CA-CFAR guard window (default: {DEFAULT_GUARD_SIZE})",
    )

    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_THRESHOLD_DB,
        help=f"CFAR threshold above background in dB (default: {DEFAULT_THRESHOLD_DB})",
    )

    parser.add_argument(
        "--coast-exclusion-pixels",
        type=int,
        default=DEFAULT_COAST_EXCLUSION_PIXELS,
        help=(
            "Expand land mask by this many pixels "
            f"(default: {DEFAULT_COAST_EXCLUSION_PIXELS})"
        ),
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
    )

    args = parser.parse_args()

    safe_dir = args.safe.resolve()

    if not safe_dir.is_dir() or safe_dir.suffix != ".SAFE":
        raise ValueError(
            f"Not a SAFE directory: {safe_dir}"
        )

    if args.output_dir is None:
        output_dir = Path("results") #/ safe_dir.name
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SAFE: {safe_dir}")

    # --------------------------------------------------------------
    # Locate files
    # --------------------------------------------------------------

    tif_path, xml_path = find_vv_files(safe_dir)

    print(f"VV TIFF: {tif_path}")
    print(f"VV XML : {xml_path}")

    # --------------------------------------------------------------
    # Read reduced image
    # --------------------------------------------------------------

    print("Reading VV image...")

    with rasterio.open(tif_path) as src:

        original_width = src.width
        original_height = src.height

        out_height = max(1, original_height // args.scale)
        out_width = max(1, original_width // args.scale)

        img = src.read(
            1,
            out_shape=(out_height, out_width),
        ).astype(np.float32)

    print(
        f"Original size: "
        f"{original_width} x {original_height}"
    )

    print(
        f"Reduced size: "
        f"{img.shape[1]} x {img.shape[0]}"
    )

    # --------------------------------------------------------------
    # Power
    # --------------------------------------------------------------

    valid = np.isfinite(img) & (img > 0)

    power = img ** 2

    # --------------------------------------------------------------
    # Geolocation
    # --------------------------------------------------------------

    print("Calculating geolocation...")

    geo = Sentinel1GeoLocator(str(xml_path))

    rows = np.arange(img.shape[0], dtype=np.float64) * args.scale
    cols = np.arange(img.shape[1], dtype=np.float64) * args.scale

    rr, cc = np.meshgrid(
        rows,
        cols,
        indexing="ij",
    )

    lat, lon = geo.pixel_to_latlon(rr, cc)

    valid &= np.isfinite(lat) & np.isfinite(lon)

    # --------------------------------------------------------------
    # Land / sea mask
    # --------------------------------------------------------------

    print("Creating land mask...")

    land = globe.is_land(lat, lon)

    land &= np.isfinite(lat) & np.isfinite(lon)

    if args.coast_exclusion_pixels > 0:

        structure_size = (
            2 * args.coast_exclusion_pixels + 1
        )

        structure = np.ones(
            (structure_size, structure_size),
            dtype=bool,
        )

        land_for_detection = binary_dilation(
            land,
            structure=structure,
        )

    else:
        land_for_detection = land

    sea = (~land_for_detection) & valid

    print(f"Original land pixels: {land.sum()}")
    print(
        "Expanded land pixels: "
        f"{land_for_detection.sum()}"
    )
    print(f"Detection sea pixels: {sea.sum()}")

    # --------------------------------------------------------------
    # CFAR
    # --------------------------------------------------------------

    print("Running CA-CFAR...")

    detections, ratio = run_cfar(
        power=power,
        valid=valid,
        sea=sea,
        train_size=args.train_size,
        guard_size=args.guard_size,
        threshold_db=args.threshold_db,
    )

    print(
        f"Raw detections: {int(detections.sum())}"
    )

    # --------------------------------------------------------------
    # Candidates
    # --------------------------------------------------------------

    candidates = extract_candidates(
        detections=detections,
        ratio=ratio,
        lat=lat,
        lon=lon,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    print(
        f"Candidates after filtering: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    safe_name = safe_dir.name
    acquisition_time = acquisition_time_from_name(
        safe_dir
    )

    csv_path = output_dir / "candidates.csv"
    png_path = output_dir / "detection.png"

    # write_csv(
    #     csv_path,
    #     candidates,
    #     safe_name,
    #     acquisition_time,
    # )

    db = np.full_like(power, np.nan)

    db[valid] = 10.0 * np.log10(
        power[valid]
    )

    write_png(
        png_path,
        db,
        lon,
        lat,
        land,
        valid,
        candidates,
    )

    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")

    for candidate in candidates:
        print(
            f"  {candidate['latitude']:.5f}, "
            f"{candidate['longitude']:.5f}, "
            f"CFAR={candidate['cfar_db']:.1f} dB"
        )


if __name__ == "__main__":
    main()
