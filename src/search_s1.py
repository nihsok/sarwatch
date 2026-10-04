#!/usr/bin/env python3
"""
Search CDSE for the newest Sentinel-1 IW GRDH dual-polarization product
intersecting the Okinawa monitoring area.

The CDSE OData productType is IW_GRDH_1S.
The "_1SDV_" part is checked in the product name to select VV+VH products.

Outputs for GitHub Actions are written to $GITHUB_OUTPUT when that
environment variable is present.

Local usage:
    python src/search_s1.py

Optional:
    python src/search_s1.py --hours 48 --top 100

Required:
    okinawa.geojson in the repository root
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)

DEFAULT_PROCESSED = Path("data/processed.yaml")
DEFAULT_GEOJSON = Path("data/targetarea.geojson")
DEFAULT_HOURS = 120
DEFAULT_TOP = 100
MAX_PROCESSED = 10

def load_polygon_wkt(path: Path) -> str:
    """Read a GeoJSON Polygon and return OGC WKT."""

    with path.open(encoding="utf-8") as f:
        geojson = json.load(f)

    geometry = geojson.get("geometry", geojson)

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Feature":
        geometry = geometry["geometry"]
        geometry_type = geometry["type"]
        coordinates = geometry["coordinates"]

    if geometry_type == "Polygon":
        # Use the exterior ring.
        ring = coordinates[0]

    elif geometry_type == "MultiPolygon":
        # For the simple monitoring workflow, use the first polygon.
        # If the AOI is a MultiPolygon with several important islands,
        # use separate searches or extend this function later.
        ring = coordinates[0][0]

    else:
        raise ValueError(
            f"Expected Polygon or MultiPolygon, got {geometry_type}"
        )

    if len(ring) < 4:
        raise ValueError("Polygon has too few coordinates")

    # Ensure closed polygon.
    if ring[0] != ring[-1]:
        ring = list(ring) + [ring[0]]

    points = ", ".join(
        f"{float(lon)} {float(lat)}"
        for lon, lat in ring
    )

    return f"POLYGON(({points}))"


def write_github_output(values: dict[str, str]) -> None:
    """Write key=value pairs to GitHub Actions GITHUB_OUTPUT."""

    output_file = os.environ.get("GITHUB_OUTPUT")

    if not output_file:
        return

    with open(output_file, "a", encoding="utf-8") as f:
        for key, value in values.items():
            # Product names/IDs do not contain newlines, but use the
            # multiline-safe form anyway.
            f.write(f"{key}={value}\n")


def parse_start(product: dict) -> datetime:
    """Parse ContentDate.Start into a timezone-aware datetime."""

    value = product["ContentDate"]["Start"]

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value)

def load_processed(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f: data = yaml.safe_load(f) or {}
    processed = data.get("processed", [])
    if not isinstance(processed, list): raise ValueError(f"Invalid processed.yaml: 'processed' must be a list: {path}")
    return processed

def save_processed(path: Path, processed: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    processed = processed[-MAX_PROCESSED:]
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"processed": processed}, f, allow_unicode=True, sort_keys=False)

def mark_processed(path: Path, product_id: str, product_name=None, acquisition_time=None, status="success") -> None:
    processed = [item for item in load_processed(path) if item.get("id") != product_id]
    entry = {"id": product_id, "status": status}#, "processed_at": datetime.now(timezone.utc).isoformat()}
    if product_name: entry["name"] = product_name
#    if acquisition_time: entry["acquisition_time"] = acquisition_time
    processed.append(entry)
    save_processed(path, processed)
    print(f"Updated: {path}")
    print(f"  id:      {product_id}")
    print(f"  status:  {status}")
    print(f"  entries: {len(processed)} / {MAX_PROCESSED}")

def search_products(
    aoi_wkt: str,
    hours: int,
    top: int,
) -> list[dict]:
    """Query CDSE OData."""

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # IMPORTANT:
    # productType is IW_GRDH_1S.
    # _1SDV_ is not a productType; it is part of the product name.
    product_filter = (
        "Collection/Name eq 'SENTINEL-1' "
        "and Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S'"
        ")"
    )

    spatial_filter = (
        "OData.CSC.Intersects("
        f"area=geography'SRID=4326;{aoi_wkt}'"
        ")"
    )

    time_filter = (
        f"ContentDate/Start gt {start_str} "
        f"and ContentDate/Start lt {end_str}"
    )

    filter_query = (
        f"{product_filter} and "
        f"{spatial_filter} and "
        f"{time_filter}"
    )

    params = {
        "$filter": filter_query,
        "$orderby": "ContentDate/Start desc",
        "$top": str(top),
        "$select": "Id,Name,ContentDate,PublicationDate",
    }

    print("Searching CDSE...")
    print(f"Time window: {start_str} -> {end_str}")
    print(f"AOI: {aoi_wkt}")

    response = requests.get(
        CATALOGUE_URL,
        params=params,
        timeout=60,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.text, file=sys.stderr)
        raise

    data = response.json()

    products = data.get("value", [])

    # Select dual-pol VV+VH GRDH products.
    products = [
        product
        for product in products
        if "_1SDV_" in product.get("Name", "")
    ]

    products.sort(
        key=parse_start,
        reverse=True,
    )

    return products


def select_new_product(products: list[dict], processed: list[dict]):
    successful_ids = {item.get("id") for item in processed if item.get("status", "success") == "success"}
    return next((p for p in products if p["Id"] not in successful_ids), None)

def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--mark-processed", metavar="PRODUCT_ID")
    parser.add_argument("--status", choices=("success", "failed"), default="success")
    parser.add_argument("--product-name", default="")
    parser.add_argument("--acquisition-time", default="")
    args = parser.parse_args()

    if args.mark_processed:
        mark_processed(args.processed, args.mark_processed, args.product_name or None, args.acquisition_time or None, args.status)
        return 0

    if not args.geojson.exists():
        raise FileNotFoundError(
            f"GeoJSON not found: {args.geojson}"
        )

    aoi_wkt = load_polygon_wkt(args.geojson)

    products = search_products(
        aoi_wkt=aoi_wkt,
        hours=args.hours,
        top=args.top,
    )
    processed = load_processed(args.processed)
    successful = sum(item.get("status", "success") == "success" for item in processed)

    print(f"Matching dual-pol products: {len(products)}")

    print(f"Previously processed successfully: {successful}")
    print("\nRecent matching products:")
    successful_ids = {item.get("id") for item in processed if item.get("status", "success") == "success"}
    for p in products[:10]:
        marker = "[processed]" if p["Id"] in successful_ids else "[new]"
        print(f"  {marker} {p['ContentDate']['Start']}  {p['Name']}  {p['Id']}")
    product = select_new_product(products, processed)
    if product is None:
        print("\nNo new Sentinel-1 IW GRDH 1SDV products found.")
        write_github_output({"found":"false", "product_id":"", "product_name":"", "acquisition_time":""})
        return 0
    product_id, product_name = product["Id"], product["Name"]
    acquisition_time = product["ContentDate"]["Start"]
    print("\nSelected new product:")
    print(f"  Name: {product_name}"); print(f"  ID:   {product_id}"); print(f"  Start:{acquisition_time}")

    write_github_output(
        {
            "found": "true",
            "product_id": product_id,
            "product_name": product_name,
            "acquisition_time": acquisition_time,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
