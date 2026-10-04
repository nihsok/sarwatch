import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _find_all_local(root, name):
    """XML namespaceを無視して指定タグを探す"""
    return [
        elem for elem in root.iter()
        if elem.tag.split("}")[-1] == name
    ]


class Sentinel1GeoLocator:
    def __init__(self, annotation_xml):
        self.annotation_xml = annotation_xml

        tree = ET.parse(annotation_xml)
        root = tree.getroot()

        points = _find_all_local(root, "geolocationGridPoint")

        if not points:
            raise RuntimeError(
                "geolocationGridPoint が見つかりません"
            )

        lines = []
        pixels = []
        lats = []
        lons = []

        for point in points:
            values = {}

            for child in point:
                name = child.tag.split("}")[-1]
                if name in {"line", "pixel", "latitude", "longitude"}:
                  values[name] = float(child.text)

            lines.append(values["line"])
            pixels.append(values["pixel"])
            lats.append(values["latitude"])
            lons.append(values["longitude"])

        lines = np.asarray(lines)
        pixels = np.asarray(pixels)
        lats = np.asarray(lats)
        lons = np.asarray(lons)

        # geolocation gridのline/pixel座標
        self.lines = np.unique(lines)
        self.pixels = np.unique(pixels)

        print("Number of geolocation points:", len(points))
        print("Number of lines:", len(self.lines))
        print("Number of pixels:", len(self.pixels))

        expected = len(self.lines) * len(self.pixels)

        if expected != len(points):
            raise RuntimeError(
                f"geolocation grid is not rectangular: "
                f"{len(points)} points != "
                f"{len(self.lines)} x {len(self.pixels)}"
            )

        # line × pixel の2次元配列にする
        lat_grid = np.empty((len(self.lines), len(self.pixels)))
        lon_grid = np.empty((len(self.lines), len(self.pixels)))

        for line, pixel, lat, lon in zip(
            lines, pixels, lats, lons
        ):
            i = np.where(self.lines == line)[0][0]
            j = np.where(self.pixels == pixel)[0][0]

            lat_grid[i, j] = lat
            lon_grid[i, j] = lon

        self.lat_grid = lat_grid
        self.lon_grid = lon_grid

        # pixel(row, col) -> latitude/longitude
        self.lat_interp = RegularGridInterpolator(
            (self.lines, self.pixels),
            self.lat_grid,
            bounds_error=False,
            fill_value=np.nan,
        )

        self.lon_interp = RegularGridInterpolator(
            (self.lines, self.pixels),
            self.lon_grid,
            bounds_error=False,
            fill_value=np.nan,
        )

    def pixel_to_latlon(self, row, col):
        """
        Sentinel-1画像のpixel座標
        row = 画像の行方向
        col = 画像の列方向

        を緯度・経度に変換
        """

        points = np.column_stack([
            np.asarray(row).ravel(),
            np.asarray(col).ravel(),
        ])

        lat = self.lat_interp(points)
        lon = self.lon_interp(points)

        shape = np.broadcast(
            np.asarray(row),
            np.asarray(col)
        ).shape

        lat = lat.reshape(shape)
        lon = lon.reshape(shape)

        return lat, lon