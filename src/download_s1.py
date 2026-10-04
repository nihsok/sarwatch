#!/usr/bin/env python3
"""Download a Sentinel-1 product ZIP from CDSE."""
import argparse
import os
import sys
from pathlib import Path
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


def get_access_token(username, password):
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=60)
    if not r.ok:
        try:
            detail = r.json()
        except ValueError:
            detail = r.text
        raise RuntimeError(f"CDSE authentication failed (HTTP {r.status_code}): {detail}")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("No access_token returned by CDSE.")
    return token


def get_product_name(product_id, token):
    url = f"{CATALOGUE_URL}/Products({product_id})?$select=Name"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Could not retrieve product metadata (HTTP {r.status_code}): {r.text}")
    name = r.json().get("Name")
    if not name:
        raise RuntimeError("Product metadata did not contain Name.")
    return name


def download_product(product_id, token, output_path):
    url = DOWNLOAD_URL.format(product_id=product_id)
    print("Downloading:")
    print(url)
    print(f"Output: {output_path}")

    with requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(60, 600),
    ) as r:
        if not r.ok:
            raise RuntimeError(f"CDSE download failed (HTTP {r.status_code}): {r.text[:1000]}")

        total = r.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r{downloaded / 1024**2:.1f} MB / {total / 1024**2:.1f} MB "
                        f"({downloaded / total * 100:.1f}%)",
                        end="", flush=True,
                    )
                else:
                    print(f"\r{downloaded / 1024**2:.1f} MB", end="", flush=True)
    print()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download a Sentinel-1 product ZIP from CDSE.")
    parser.add_argument("product_id", help="CDSE product UUID")
    parser.add_argument("--output-dir", default=os.environ.get("CDSE_OUTPUT_DIR", "data"))
    args = parser.parse_args()

    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        print("ERROR: CDSE_USERNAME and CDSE_PASSWORD must be set.", file=sys.stderr)
        sys.exit(1)

    try:
        print("Authenticating with CDSE...")
        token = get_access_token(username, password)
        print("CDSE authentication successful")

        product_name = get_product_name(args.product_id, token)
        zip_name = product_name[:-5] + ".zip" if product_name.endswith(".SAFE") else product_name + ".zip"
        output_path = Path(args.output_dir) / zip_name

        if output_path.exists():
            print(f"Already exists, skipping: {output_path}")
            return

        download_product(args.product_id, token, output_path)

    except requests.RequestException as exc:
        print(f"ERROR: HTTP request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
