"""Create a UK solar irradiation heatmap from an ERA5 NetCDF file.

Example:
    python main.py --input era5_monthly_ssrd.nc --output uk_irradiation.png

The script:
1. Opens the ERA5 NetCDF file with xarray.
2. Converts longitude from 0..360 to -180..180 when required.
3. Crops the raster to the UK region.
4. Converts SSRD from J/m2 to kWh/m2 by default.
5. Downloads Natural Earth boundaries through GeoPandas if they are not cached.
6. Draws and saves a UK irradiation map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr


UK_BOUNDS = {
    "west": -8.6,
    "east": 2.0,
    "south": 49.0,
    "north": 61.0,
}

# GeoPandas can read this ZIP directly. It will only be fetched if no local
# Natural Earth shapefile is supplied with --boundaries.
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_countries.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a UK irradiation heatmap from ERA5 monthly SSRD data."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the ERA5 NetCDF (.nc) file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("uk_irradiation_map.png"),
        help="Output image path (default: uk_irradiation_map.png).",
    )
    parser.add_argument(
        "--variable",
        default="ssrd",
        help="NetCDF variable containing SSRD (default: ssrd).",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time position to plot when the file contains multiple periods (default: 0).",
    )
    parser.add_argument(
        "--unit",
        choices=("j", "wh", "kwh"),
        default="kwh",
        help="Display unit: j, wh, or kwh per square metre (default: kwh).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title. If omitted, a title is generated from the time coordinate.",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=None,
        help="Optional local Natural Earth country shapefile (.shp).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the map after saving it.",
    )
    return parser.parse_args()


def standardise_coordinate_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename common latitude/longitude alternatives to ERA5-style names."""
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    return ds.rename(rename) if rename else ds


def prepare_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = standardise_coordinate_names(ds)

    missing = {"latitude", "longitude"} - set(ds.coords)
    if missing:
        raise ValueError(
            "Missing required coordinate(s): " + ", ".join(sorted(missing))
        )

    # ERA5 commonly stores longitude as 0..360. This expression is also safe
    # when longitudes are already in the -180..180 range.
    ds = ds.assign_coords(
        longitude=((ds.longitude + 180) % 360) - 180
    ).sortby("longitude")

    # Make latitude descending so slice(61, 49) works consistently.
    if ds.latitude.size > 1 and float(ds.latitude[0]) < float(ds.latitude[-1]):
        ds = ds.sortby("latitude", ascending=False)

    return ds


def select_time(data: xr.DataArray, time_index: int) -> tuple[xr.DataArray, object | None]:
    """Select one period, supporting common ERA5 time-coordinate names."""
    for time_name in ("valid_time", "time"):
        if time_name in data.dims:
            if not 0 <= time_index < data.sizes[time_name]:
                raise IndexError(
                    f"--time-index {time_index} is outside the available range "
                    f"0..{data.sizes[time_name] - 1}."
                )
            selected = data.isel({time_name: time_index})
            return selected, selected.coords.get(time_name)
    return data.squeeze(drop=True), None


def convert_units(data: xr.DataArray, unit: str) -> tuple[xr.DataArray, str]:
    """Convert joules per square metre to the requested display unit."""
    if unit == "j":
        return data, "Solar irradiation (J/m²)"
    if unit == "wh":
        return data / 3_600.0, "Solar irradiation (Wh/m²)"
    return data / 3_600_000.0, "Solar irradiation (kWh/m²)"


def load_uk_boundary(shapefile: Path | None) -> gpd.GeoDataFrame:
    source = str(shapefile) if shapefile else NATURAL_EARTH_URL
    world = gpd.read_file(source)

    if "ADMIN" not in world.columns:
        raise ValueError("Boundary data does not contain an 'ADMIN' column.")

    uk_shape = world.loc[world["ADMIN"] == "United Kingdom"]
    if uk_shape.empty:
        raise ValueError("United Kingdom geometry was not found in the boundary data.")

    # The ERA5 coordinates use geographic longitude/latitude (EPSG:4326).
    if uk_shape.crs is not None and uk_shape.crs.to_epsg() != 4326:
        uk_shape = uk_shape.to_crs(epsg=4326)

    return uk_shape


def make_title(time_coord: object | None) -> str:
    if time_coord is None:
        return "UK Solar Irradiation"

    try:
        value = time_coord.values
        return f"{str(value)[:7]} UK Solar Irradiation"
    except Exception:
        return "UK Solar Irradiation"


def create_map(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with xr.open_dataset(args.input) as opened:
        ds = prepare_dataset(opened)

        if args.variable not in ds.data_vars:
            available = ", ".join(ds.data_vars)
            raise KeyError(
                f"Variable '{args.variable}' was not found. Available variables: {available}"
            )

        uk = ds.sel(
            latitude=slice(UK_BOUNDS["north"], UK_BOUNDS["south"]),
            longitude=slice(UK_BOUNDS["west"], UK_BOUNDS["east"]),
        )

        if uk.latitude.size == 0 or uk.longitude.size == 0:
            raise ValueError("The UK crop returned no cells. Check the dataset coordinates.")

        irradiation, time_coord = select_time(uk[args.variable], args.time_index)
        irradiation, colourbar_label = convert_units(irradiation, args.unit)
        irradiation = irradiation.load()

    if set(irradiation.dims) != {"latitude", "longitude"}:
        raise ValueError(
            "After selecting a time period, the variable must have only latitude "
            f"and longitude dimensions. Found: {irradiation.dims}"
        )

    uk_shape = load_uk_boundary(args.boundaries)

    fig, ax = plt.subplots(figsize=(8, 10))
    image = irradiation.plot.imshow(
        ax=ax,
        cmap="YlOrRd",
        interpolation="bilinear",
        add_colorbar=False,
        x="longitude",
        y="latitude",
    )

    colourbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.025)
    colourbar.set_label(colourbar_label)

    uk_shape.boundary.plot(ax=ax, color="black", linewidth=0.8)

    ax.set_xlim(UK_BOUNDS["west"], UK_BOUNDS["east"])
    ax.set_ylim(UK_BOUNDS["south"], UK_BOUNDS["north"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(args.title or make_title(time_coord), fontsize=16, pad=15)

    output_path = Path.cwd() / Path(args.output).name
    print (f"Saving map to: {output_path}")

    fig.savefig(
        "c:\\era\\test1.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved map to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    create_map(args)


if __name__ == "__main__":
    main()
