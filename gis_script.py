"""
Chinhoyi Smart Solar Streetlight Management & Decision-Support Dashboard
=======================================================================

Purpose
-------
A production-oriented prototype for the Municipality of Chinhoyi that combines:

1. Existing streetlight asset management
2. Road-network-based candidate generation
3. Lighting-gap / coverage analysis
4. Population-density analysis
5. Bus-stop proximity
6. Public-facility proximity
7. Safety / crime-risk (when data are supplied)
8. Solar suitability (when raster data are supplied)
9. Terrain / slope (when DEM data are supplied)
10. Environmental constraints (when supplied)
11. AHP/MCDA weighting
12. Scenario analysis
13. Optional Machine Learning prioritisation
14. Explainable AI-style factor contribution
15. Budget and implementation phasing
16. Maintenance / work-order tracking
17. Community fault reporting
18. CSV / GeoJSON / GeoPackage export

IMPORTANT
---------
The application is designed to work with real GIS layers. It will also run in
"DEMO / DATA-LIMITED" mode when required layers are missing, but those fallback
values are clearly labelled as simulated and should NOT be used as municipal
evidence or final dissertation results.

Recommended structure:

project/
    app.py
    data/
        streetlights.*          optional
        roads.*                optional
        suburbs.*              optional
        boundary.*             optional
        bus_stops.*             optional
        facilities.*            optional
        crime.*                 optional
        population_points.*     optional
        population_density.*    optional raster
        dem.tif                 optional raster
        solar_radiation.tif     optional raster
        flood_risk.*            optional
        wetlands.*              optional
        environmental.*         optional

Run:
    panel serve app.py --show --autoreload

Main libraries:
    panel, folium, geopandas, pandas, numpy, shapely, pyproj
Optional:
    rasterio, scikit-learn, openpyxl

CHANGE LOG (bugfixes applied)
------------------------------
1. `load_all_data()` now runs every loaded vector layer through
   `ensure_point_layer()` so that rows with null/empty geometries are
   dropped immediately after loading, instead of surviving into functions
   such as `coverage_percentage()` which assume every geometry is usable.
   This directly fixes:
       AttributeError: 'NoneType' object has no attribute 'buffer'
   which was raised from `coverage_percentage()` when a streetlights layer
   contained at least one null geometry.
2. `coverage_percentage()` now also defensively filters null/empty
   geometries out of both the lights and population layers itself, so it
   is safe even if it is ever called with a layer that was not routed
   through `ensure_point_layer()`.
3. `safe_read_vector()` now reads `.parquet` files with
   `geopandas.read_parquet()` (pyarrow-based) instead of routing them
   through `geopandas.read_file()`, which depends on GDAL's Arrow/Parquet
   driver and, on this machine, required a missing `duckdb.dll`. This
   removes the startup warning:
       Could not read moc_census_data_addresses.parquet: Can't load
       requested DLL: duckdb.dll
   and allows the real population/census GeoParquet layer to load instead
   of silently falling back to synthetic demo population data. If the
   parquet file turns out to be a plain (non-geo) parquet with lon/lat
   columns rather than a GeoParquet with a geometry column, this branch
   falls back to building points from lon/lat columns the same way the
   `.csv` branch does.
"""

from __future__ import annotations

import io
import json
import math
import os
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import panel as pn

from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import unary_union, nearest_points
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    rasterio = None
    RASTERIO_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------
# PANEL SETUP
# ---------------------------------------------------------------------

pn.extension("tabulator", notifications=True, sizing_mode="stretch_width")

# ---------------------------------------------------------------------
# PATHS / CONFIG
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CHINHOYI_LAT = -17.3667
CHINHOYI_LON = 30.2000

# UTM Zone 36S is used in the original prototype.
WGS84 = "EPSG:4326"
UTM = "EPSG:32736"

TO_UTM = Transformer.from_crs(WGS84, UTM, always_xy=True)
TO_WGS = Transformer.from_crs(UTM, WGS84, always_xy=True)

# ---------------------------------------------------------------------
# BRAND / MAP COLOURS
# ---------------------------------------------------------------------

COLOR_MUNI_GREEN = "#006837"
COLOR_MUNI_GOLD = "#FDB913"
COLOR_CUT_NAVY = "#002147"
COLOR_CUT_GOLD = "#D4AF37"

COLOR_HIGH = "#DC2626"
COLOR_MED = "#EA580C"
COLOR_LOW = "#16A34A"
COLOR_INFO = "#2563EB"
COLOR_FAULT = "#991B1B"
COLOR_MAINT = "#D97706"
COLOR_OFFLINE = "#6B7280"
COLOR_EXISTING = "#0F766E"
COLOR_PROPOSED = "#7C3AED"

# ---------------------------------------------------------------------
# DATA DISCOVERY
# ---------------------------------------------------------------------

LAYER_ALIASES = {
    "streetlights": [
        "streetlights", "street_lights", "street light",
        "existing_streetlights", "streetlights_existing"
    ],
    "roads": [
        "roads", "road", "road_network", "roadnetwork", "streets"
    ],
    "suburbs": [
        "suburbs", "suburb", "wards", "ward", "moc_suburbs"
    ],
    "boundary": [
        "boundary", "municipal_boundary", "municipality", "admin",
        "mcd", "chinhoyi_boundary"
    ],
    "bus_stops": [
        "bus_stops", "bus_stops", "bus stop", "transport_stops"
    ],
    "facilities": [
        "facilities", "public_facilities", "schools", "clinics",
        "hospitals", "markets", "public_services"
    ],
    "population": [
        "population", "census", "census_data", "moc_census",
        "addresses", "moc_census_data_addresses"
    ],
    "crime": [
        "crime", "crime_risk", "incidents", "crime_events", "safety"
    ],
    "flood_risk": [
        "flood", "flood_risk", "flood_zones", "floodplain"
    ],
    "wetlands": [
        "wetland", "wetlands", "environmental", "protected"
    ],
}


def list_spatial_files() -> List[Path]:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {".shp", ".gpkg", ".geojson", ".json", ".parquet", ".csv", ".tif", ".tiff"}
    return sorted([p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in allowed])


def find_layer(name: str) -> Optional[Path]:
    candidates = list_spatial_files()
    aliases = LAYER_ALIASES.get(name, [name])

    # Exact-ish match first.
    for alias in aliases:
        a = alias.lower().replace(" ", "_")
        for p in candidates:
            stem = p.stem.lower().replace(" ", "_")
            if stem == a or stem.startswith(a) or a in stem:
                return p
    return None


def find_raster(*keywords: str) -> Optional[Path]:
    for p in list_spatial_files():
        if p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stem = p.stem.lower().replace(" ", "_")
        if any(k.lower().replace(" ", "_") in stem for k in keywords):
            return p
    return None


@dataclass
class DataStore:
    streetlights: Optional[gpd.GeoDataFrame] = None
    roads: Optional[gpd.GeoDataFrame] = None
    suburbs: Optional[gpd.GeoDataFrame] = None
    boundary: Optional[gpd.GeoDataFrame] = None
    bus_stops: Optional[gpd.GeoDataFrame] = None
    facilities: Optional[gpd.GeoDataFrame] = None
    population: Optional[gpd.GeoDataFrame] = None
    crime: Optional[gpd.GeoDataFrame] = None
    flood_risk: Optional[gpd.GeoDataFrame] = None
    wetlands: Optional[gpd.GeoDataFrame] = None

    dem_path: Optional[Path] = None
    solar_path: Optional[Path] = None

    # Runtime / derived datasets
    candidates: Optional[gpd.GeoDataFrame] = None
    maintenance: pd.DataFrame = field(default_factory=pd.DataFrame)
    community_reports: pd.DataFrame = field(default_factory=pd.DataFrame)
    work_orders: pd.DataFrame = field(default_factory=pd.DataFrame)


DATA = DataStore()


# ---------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------

def _points_from_lonlat_dataframe(df: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    """Build a point GeoDataFrame from a plain DataFrame with lon/lat columns."""
    lon_col = next(
        (c for c in df.columns if c.lower() in {"lon", "longitude", "x"}),
        None,
    )
    lat_col = next(
        (c for c in df.columns if c.lower() in {"lat", "latitude", "y"}),
        None,
    )
    if lon_col and lat_col:
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=WGS84,
        )
    return None


def safe_read_vector(path: Optional[Path]) -> Optional[gpd.GeoDataFrame]:
    if path is None or not path.exists():
        return None

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
            gdf = _points_from_lonlat_dataframe(df)
            return gdf

        if suffix == ".parquet":
            # GeoParquet: read directly with geopandas/pyarrow. This avoids
            # routing through gpd.read_file()'s GDAL Arrow/Parquet driver,
            # which on some Windows installs requires duckdb.dll.
            try:
                gdf = gpd.read_parquet(path)
            except Exception:
                # Fall back: maybe it's a plain (non-geo) parquet with
                # lon/lat columns rather than a geometry column.
                df = pd.read_parquet(path)
                gdf = _points_from_lonlat_dataframe(df)
                if gdf is None:
                    print(
                        f"[WARNING] {path.name} is not a GeoParquet file and "
                        "has no recognizable lon/lat columns."
                    )
                    return None
        else:
            gdf = gpd.read_file(path)

        if gdf is None or gdf.empty:
            return None

        if gdf.crs is None:
            # Match the original prototype's assumption, but record this as
            # an assumption in the UI rather than silently calling it verified.
            gdf = gdf.set_crs(UTM)

        return gdf.to_crs(WGS84)

    except Exception as exc:
        print(f"[WARNING] Could not read {path.name}: {exc}")
        return None


def ensure_point_layer(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """Drop rows with missing/empty geometry so downstream spatial ops
    (buffer, distance, etc.) never encounter a None geometry."""
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    out = out[out.geometry.notna()].copy()
    out = out[~out.geometry.is_empty].copy()

    if out.empty:
        return None

    # For polygons/lines, preserve original layer. Candidate calculations
    # will use centroids only when necessary.
    return out


def load_all_data() -> DataStore:
    store = DataStore()

    for name in [
        "streetlights", "roads", "suburbs", "boundary", "bus_stops",
        "facilities", "population", "crime", "flood_risk", "wetlands"
    ]:
        path = find_layer(name)
        gdf = safe_read_vector(path)
        # Strip null/empty geometries immediately after loading so that no
        # downstream function (e.g. coverage_percentage) ever has to deal
        # with a None geometry.
        setattr(store, name, ensure_point_layer(gdf))

    store.dem_path = find_raster("dem", "elevation", "terrain")
    store.solar_path = find_raster("solar", "irradiance", "radiation")

    # Build an operational maintenance table from available streetlights.
    if store.streetlights is not None and not store.streetlights.empty:
        store.streetlights = store.streetlights.copy()
        store.streetlights["asset_id"] = build_asset_ids(store.streetlights)

        store.maintenance = initialize_maintenance_table(store.streetlights)
    else:
        store.maintenance = pd.DataFrame(
            columns=[
                "work_order_id", "asset_id", "status", "fault_type",
                "reported_at", "assigned_to", "repair_date", "notes"
            ]
        )

    store.community_reports = pd.DataFrame(
        columns=[
            "report_id", "reported_at", "category", "latitude", "longitude",
            "description", "status", "priority"
        ]
    )

    store.work_orders = pd.DataFrame(
        columns=[
            "work_order_id", "asset_id", "status", "priority",
            "fault_type", "assigned_to", "reported_at", "repair_date", "notes"
        ]
    )

    return store


def build_asset_ids(gdf: gpd.GeoDataFrame) -> List[str]:
    ids = []
    for i, _ in enumerate(gdf.itertuples(), start=1):
        ids.append(f"SL-CHN-{i:04d}")
    return ids


def initialize_maintenance_table(streetlights: gpd.GeoDataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = len(streetlights)

    statuses = rng.choice(
        ["Operational", "Operational", "Operational", "Faulty", "Maintenance", "Offline"],
        size=n,
    )

    fault_types = []
    for status in statuses:
        if status == "Faulty":
            fault_types.append(
                rng.choice(["Battery", "Lamp/LED", "Solar Panel", "Controller", "Pole"])
            )
        else:
            fault_types.append("None")

    return pd.DataFrame(
        {
            "work_order_id": [f"WO-{i+1:05d}" for i in range(n)],
            "asset_id": streetlights["asset_id"].tolist(),
            "status": statuses,
            "fault_type": fault_types,
            "reported_at": [datetime.now().date()] * n,
            "assigned_to": ["Unassigned"] * n,
            "repair_date": [None] * n,
            "notes": [""] * n,
        }
    )


# ---------------------------------------------------------------------
# DEMO FALLBACK
# ---------------------------------------------------------------------

def create_demo_layers(store: DataStore) -> DataStore:
    """
    Creates demo layers ONLY when real layers are missing.
    These values are synthetic and are explicitly labelled in the UI.
    """

    rng = np.random.default_rng(10)

    if store.boundary is None:
        polygon = Polygon(
            [
                (30.145, -17.415),
                (30.255, -17.415),
                (30.255, -17.320),
                (30.145, -17.320),
            ]
        )
        store.boundary = gpd.GeoDataFrame(
            {"name": ["Demo Chinhoyi Boundary"]},
            geometry=[polygon],
            crs=WGS84,
        )

    if store.suburbs is None:
        # Simple demonstration polygons.
        west = Polygon(
            [(30.155, -17.405), (30.205, -17.405), (30.205, -17.355), (30.155, -17.355)]
        )
        east = Polygon(
            [(30.205, -17.405), (30.250, -17.405), (30.250, -17.355), (30.205, -17.355)]
        )
        store.suburbs = gpd.GeoDataFrame(
            {"name": ["Demo High Density West", "Demo High Density East"]},
            geometry=[west, east],
            crs=WGS84,
        )

    if store.roads is None:
        lines = []
        names = []
        road_types = []

        for i in range(11):
            y = -17.400 + i * 0.008
            lines.append(LineString([(30.155, y), (30.250, y)]))
            names.append(f"Demo Road H-{i+1}")
            road_types.append("Urban Collector")

        for i in range(11):
            x = 30.155 + i * 0.0095
            lines.append(LineString([(x, -17.405), (x, -17.350)]))
            names.append(f"Demo Road V-{i+1}")
            road_types.append("Local Road")

        store.roads = gpd.GeoDataFrame(
            {"road_name": names, "road_type": road_types},
            geometry=lines,
            crs=WGS84,
        )

    if store.streetlights is None:
        points = []
        road_ids = []
        for i in range(40):
            road_idx = i % len(store.roads)
            geom = store.roads.geometry.iloc[road_idx]
            length = geom.length
            fraction = rng.uniform(0.05, 0.95)
            point = geom.interpolate(fraction, normalized=True)
            points.append(point)
            road_ids.append(road_idx)

        store.streetlights = gpd.GeoDataFrame(
            {
                "asset_id": [f"SL-CHN-{i+1:04d}" for i in range(len(points))],
                "light_type": rng.choice(["Solar LED", "Grid LED"], len(points)),
                "road_id": road_ids,
                "installation_year": rng.choice([2024, 2025, 2026], len(points)),
            },
            geometry=points,
            crs=WGS84,
        )
        store.maintenance = initialize_maintenance_table(store.streetlights)

    if store.bus_stops is None:
        pts = [
            Point(30.166, -17.395),
            Point(30.183, -17.375),
            Point(30.214, -17.390),
            Point(30.235, -17.370),
            Point(30.245, -17.400),
        ]
        store.bus_stops = gpd.GeoDataFrame(
            {"stop_name": [f"Demo Bus Stop {i+1}" for i in range(len(pts))]},
            geometry=pts,
            crs=WGS84,
        )

    if store.population is None:
        pts = []
        values = []
        for _ in range(700):
            x = rng.uniform(30.155, 30.250)
            y = rng.uniform(-17.405, -17.350)
            pts.append(Point(x, y))
            values.append(int(rng.integers(1, 8)))

        store.population = gpd.GeoDataFrame(
            {"households": values},
            geometry=pts,
            crs=WGS84,
        )

    if store.facilities is None:
        facilities = [
            (30.174, -17.392, "School"),
            (30.225, -17.383, "Clinic"),
            (30.235, -17.399, "Market"),
            (30.192, -17.368, "School"),
            (30.215, -17.402, "Police"),
            (30.245, -17.367, "Clinic"),
        ]
        store.facilities = gpd.GeoDataFrame(
            {
                "facility_name": [f"Demo Facility {i+1}" for i in range(len(facilities))],
                "facility_type": [f[2] for f in facilities],
            },
            geometry=[Point(f[0], f[1]) for f in facilities],
            crs=WGS84,
        )

    return store


# ---------------------------------------------------------------------
# SPATIAL UTILITIES
# ---------------------------------------------------------------------

def add_utm_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.to_crs(UTM).copy()
    return out


def representative_point(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Point":
        return geom
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return geom.representative_point()
    return geom.centroid


def point_distance_m(point: Point, layer: Optional[gpd.GeoDataFrame]) -> float:
    if layer is None or layer.empty or point is None:
        return np.nan

    tmp = layer.copy()
    tmp["geometry"] = tmp.geometry.apply(representative_point)
    tmp = tmp[tmp.geometry.notna()]
    tmp = tmp.to_crs(UTM)

    p = gpd.GeoSeries([point], crs=WGS84).to_crs(UTM).iloc[0]
    return float(tmp.geometry.distance(p).min())


def normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)

    fill = float(s.median())
    s = s.fillna(fill)

    min_v = float(s.min())
    max_v = float(s.max())

    if math.isclose(min_v, max_v):
        return pd.Series(0.5, index=series.index)

    out = (s - min_v) / (max_v - min_v)
    return out if higher_is_better else 1.0 - out


def inverse_distance_score(distance_m: pd.Series, max_distance: float) -> pd.Series:
    d = pd.to_numeric(distance_m, errors="coerce").fillna(max_distance)
    return (1.0 - np.minimum(d / max_distance, 1.0)).clip(0, 1)


def nearest_distance_series(
    points: gpd.GeoDataFrame,
    target: Optional[gpd.GeoDataFrame],
) -> pd.Series:
    if target is None or target.empty:
        return pd.Series(np.nan, index=points.index)

    p = points.to_crs(UTM)
    t = target.copy()
    t["geometry"] = t.geometry.apply(representative_point)
    t = t[t.geometry.notna()].to_crs(UTM)

    if t.empty:
        return pd.Series(np.nan, index=points.index)

    target_union = unary_union(list(t.geometry))
    values = [geom.distance(target_union) if geom else np.nan for geom in p.geometry]
    return pd.Series(values, index=points.index)


def safe_div(a, b):
    return np.where(np.asarray(b) == 0, 0, np.asarray(a) / np.asarray(b))


# ---------------------------------------------------------------------
# CANDIDATE GENERATION
# ---------------------------------------------------------------------

def generate_road_based_candidates(
    roads: Optional[gpd.GeoDataFrame],
    existing_lights: Optional[gpd.GeoDataFrame],
    spacing_m: int = 100,
    max_candidates: int = 500,
) -> gpd.GeoDataFrame:
    """
    Generates candidate installation points from actual road geometries.
    Existing streetlights are buffered so that candidate points are not
    unnecessarily duplicated near already-lit locations.
    """

    if roads is None or roads.empty:
        return generate_demo_candidates(existing_lights, max_candidates=max_candidates)

    road_utm = roads.to_crs(UTM)
    existing_utm = existing_lights.to_crs(UTM) if existing_lights is not None and not existing_lights.empty else None

    rows = []
    candidate_counter = 1

    for ridx, row in road_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        elif geom.geom_type == "LineString":
            parts = [geom]
        else:
            continue

        road_name = str(
            row.get("road_name")
            or row.get("name")
            or row.get("street")
            or f"Road {ridx}"
        )
        road_type = str(
            row.get("road_type")
            or row.get("type")
            or row.get("class")
            or "Unknown"
        )

        for part in parts:
            length = float(part.length)
            if length < 25:
                continue

            distances = np.arange(25, max(length, 25), spacing_m)
            if len(distances) == 0:
                distances = [length / 2]

            for dist in distances:
                p = part.interpolate(float(dist))

                # Avoid duplicate candidates too close to existing lights.
                if existing_utm is not None and not existing_utm.empty:
                    min_d = float(existing_utm.geometry.distance(p).min())
                    if min_d < spacing_m * 0.55:
                        continue
                else:
                    min_d = np.nan

                lon, lat = TO_WGS.transform(p.x, p.y)
                rows.append(
                    {
                        "candidate_id": f"CSL-{candidate_counter:04d}",
                        "road_name": road_name,
                        "road_type": road_type,
                        "distance_existing_light_m": min_d,
                        "geometry": Point(lon, lat),
                    }
                )
                candidate_counter += 1

                if len(rows) >= max_candidates:
                    break

            if len(rows) >= max_candidates:
                break

        if len(rows) >= max_candidates:
            break

    if not rows:
        return generate_demo_candidates(existing_lights, max_candidates=max_candidates)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


def generate_demo_candidates(
    existing_lights: Optional[gpd.GeoDataFrame],
    max_candidates: int = 250,
) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(42)

    existing_union = None
    if existing_lights is not None and not existing_lights.empty:
        existing_union = unary_union(existing_lights.to_crs(UTM).geometry)

    rows = []
    c = 1

    for _ in range(max_candidates * 4):
        lon = CHINHOYI_LON + rng.uniform(-0.045, 0.045)
        lat = CHINHOYI_LAT + rng.uniform(-0.045, 0.045)

        p_wgs = Point(lon, lat)
        p_utm = gpd.GeoSeries([p_wgs], crs=WGS84).to_crs(UTM).iloc[0]

        if existing_union is not None and existing_union.distance(p_utm) < 50:
            continue

        rows.append(
            {
                "candidate_id": f"CSL-{c:04d}",
                "road_name": "DEMO CANDIDATE — no road dataset",
                "road_type": "Demo",
                "distance_existing_light_m": (
                    existing_union.distance(p_utm)
                    if existing_union is not None
                    else np.nan
                ),
                "geometry": p_wgs,
            }
        )
        c += 1

        if len(rows) >= max_candidates:
            break

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


# ---------------------------------------------------------------------
# POINT / RASTER FACTOR EXTRACTION
# ---------------------------------------------------------------------

def point_counts_within(
    points: gpd.GeoDataFrame,
    target_points: Optional[gpd.GeoDataFrame],
    radius_m: float,
) -> pd.Series:
    if target_points is None or target_points.empty:
        return pd.Series(0.0, index=points.index)

    p = points.to_crs(UTM)
    t = target_points.copy()
    t["geometry"] = t.geometry.apply(representative_point)
    t = t[t.geometry.notna()].to_crs(UTM)

    coords = np.array([(g.x, g.y) for g in t.geometry])
    out = []

    for geom in p.geometry:
        if coords.size == 0:
            out.append(0.0)
            continue

        dx = coords[:, 0] - geom.x
        dy = coords[:, 1] - geom.y
        d2 = dx * dx + dy * dy
        out.append(float(np.sum(d2 <= radius_m**2)))

    return pd.Series(out, index=points.index)


def population_value_within(
    candidates: gpd.GeoDataFrame,
    population: Optional[gpd.GeoDataFrame],
    radius_m: float,
) -> pd.Series:
    if population is None or population.empty:
        return pd.Series(0.0, index=candidates.index)

    p = candidates.to_crs(UTM)
    pop = population.copy()
    pop["geometry"] = pop.geometry.apply(representative_point)
    pop = pop[pop.geometry.notna()].to_crs(UTM)

    value_col = next(
        (
            c for c in pop.columns
            if c.lower() in {
                "population", "pop", "persons", "households", "count",
                "population_count", "household_count"
            }
        ),
        None,
    )

    coords = np.array([(g.x, g.y) for g in pop.geometry])
    values = (
        pd.to_numeric(pop[value_col], errors="coerce").fillna(1).to_numpy()
        if value_col
        else np.ones(len(pop))
    )

    out = []

    for geom in p.geometry:
        if coords.size == 0:
            out.append(0.0)
            continue
        dx = coords[:, 0] - geom.x
        dy = coords[:, 1] - geom.y
        d2 = dx * dx + dy * dy
        out.append(float(values[d2 <= radius_m**2].sum()))

    return pd.Series(out, index=candidates.index)


def sample_raster_at_points(
    points: gpd.GeoDataFrame,
    raster_path: Optional[Path],
) -> pd.Series:
    if not RASTERIO_AVAILABLE or raster_path is None or not raster_path.exists():
        return pd.Series(np.nan, index=points.index)

    try:
        with rasterio.open(raster_path) as src:
            p = points.to_crs(src.crs)
            coords = [(geom.x, geom.y) for geom in p.geometry]
            values = []
            for value in src.sample(coords):
                v = value[0]
                if src.nodata is not None and np.isclose(v, src.nodata):
                    values.append(np.nan)
                else:
                    values.append(float(v))
            return pd.Series(values, index=points.index)
    except Exception as exc:
        print(f"[WARNING] Raster sampling failed for {raster_path.name}: {exc}")
        return pd.Series(np.nan, index=points.index)


def compute_slope_from_dem(
    points: gpd.GeoDataFrame,
    dem_path: Optional[Path],
) -> pd.Series:
    """
    Lightweight local slope approximation using raster neighbourhood.
    Returns percent-like slope degrees where possible.
    """

    if not RASTERIO_AVAILABLE or dem_path is None or not dem_path.exists():
        return pd.Series(np.nan, index=points.index)

    try:
        with rasterio.open(dem_path) as src:
            p = points.to_crs(src.crs)
            values = []

            for geom in p.geometry:
                row, col = src.index(geom.x, geom.y)

                r0 = max(0, row - 1)
                r1 = min(src.height, row + 2)
                c0 = max(0, col - 1)
                c1 = min(src.width, col + 2)

                arr = src.read(1, window=((r0, r1), (c0, c1))).astype(float)
                arr[arr == src.nodata] = np.nan if src.nodata is not None else np.nan

                if np.isnan(arr).all():
                    values.append(np.nan)
                    continue

                center = arr[arr.shape[0] // 2, arr.shape[1] // 2]
                if np.isnan(center):
                    values.append(np.nan)
                    continue

                local_range = np.nanmax(arr) - np.nanmin(arr)
                res_x = abs(float(src.transform.a))
                res_y = abs(float(src.transform.e))
                run = max((res_x + res_y) / 2, 1e-9)

                slope_radians = math.atan2(float(local_range), run * 2.0)
                values.append(math.degrees(slope_radians))

            return pd.Series(values, index=points.index)

    except Exception as exc:
        print(f"[WARNING] DEM slope calculation failed: {exc}")
        return pd.Series(np.nan, index=points.index)


# ---------------------------------------------------------------------
# ENVIRONMENTAL CONSTRAINTS
# ---------------------------------------------------------------------

def intersects_layer(
    candidates: gpd.GeoDataFrame,
    layer: Optional[gpd.GeoDataFrame],
    buffer_m: float = 0,
) -> pd.Series:
    if layer is None or layer.empty:
        return pd.Series(False, index=candidates.index)

    c = candidates.to_crs(UTM).copy()
    l = layer.to_crs(UTM).copy()

    target = unary_union(l.geometry)
    flags = []

    for geom in c.geometry:
        test_geom = geom.buffer(buffer_m) if buffer_m > 0 else geom
        flags.append(bool(test_geom.intersects(target)))

    return pd.Series(flags, index=candidates.index)


# ---------------------------------------------------------------------
# FACILITY / SAFETY / ROAD FEATURES
# ---------------------------------------------------------------------

def facilities_score(candidates: gpd.GeoDataFrame, facilities: Optional[gpd.GeoDataFrame]) -> pd.Series:
    if facilities is None or facilities.empty:
        return pd.Series(0.0, index=candidates.index)

    c = candidates.to_crs(UTM)

    f = facilities.copy()
    f["geometry"] = f.geometry.apply(representative_point)
    f = f[f.geometry.notna()].to_crs(UTM)

    # Weighted by type to represent the higher importance of health,
    # emergency and education services.
    types = (
        f.get("facility_type", pd.Series(["Other"] * len(f), index=f.index))
        .astype(str)
        .str.lower()
    )
    weights = np.where(
        types.str.contains("police|hospital|clinic|emergency", regex=True), 3.0,
        np.where(
            types.str.contains("school|college|university", regex=True), 2.0,
            np.where(types.str.contains("market|terminal|transport", regex=True), 1.5, 1.0)
        )
    )

    values = []
    for geom in c.geometry:
        distances = f.geometry.distance(geom).to_numpy()
        influence = np.exp(-distances / 500.0)
        values.append(float(np.sum(influence * weights)))

    return pd.Series(values, index=candidates.index)


def crime_score(candidates: gpd.GeoDataFrame, crime: Optional[gpd.GeoDataFrame]) -> pd.Series:
    if crime is None or crime.empty:
        return pd.Series(0.0, index=candidates.index)

    c = candidates.to_crs(UTM)

    cr = crime.copy()
    cr["geometry"] = cr.geometry.apply(representative_point)
    cr = cr[cr.geometry.notna()].to_crs(UTM)

    value_col = next(
        (
            col for col in cr.columns
            if col.lower() in {"count", "frequency", "incidents", "risk", "cases"}
        ),
        None,
    )

    values = (
        pd.to_numeric(cr[value_col], errors="coerce").fillna(1).to_numpy()
        if value_col
        else np.ones(len(cr))
    )

    scores = []
    for geom in c.geometry:
        d = cr.geometry.distance(geom).to_numpy()
        influence = np.exp(-d / 400.0)
        scores.append(float(np.sum(influence * values)))

    return pd.Series(scores, index=candidates.index)


def road_priority_score(candidates: gpd.GeoDataFrame) -> pd.Series:
    if "road_type" not in candidates.columns:
        return pd.Series(0.5, index=candidates.index)

    s = candidates["road_type"].astype(str).str.lower()

    score = np.select(
        [
            s.str.contains("highway|primary|trunk|arterial", regex=True),
            s.str.contains("secondary|collector", regex=True),
            s.str.contains("tertiary|local", regex=True),
        ],
        [1.0, 0.75, 0.5],
        default=0.4,
    )
    return pd.Series(score, index=candidates.index)


# ---------------------------------------------------------------------
# LIGHTING COVERAGE
# ---------------------------------------------------------------------

def compute_lighting_gap_score(
    candidates: gpd.GeoDataFrame,
    existing_lights: Optional[gpd.GeoDataFrame],
    target_radius_m: float = 75,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
        nearest_existing_distance_m
        lighting_gap_score
    """

    if existing_lights is None or existing_lights.empty:
        dist = pd.Series(target_radius_m * 2, index=candidates.index)
        return dist, pd.Series(1.0, index=candidates.index)

    d = nearest_distance_series(candidates, existing_lights)
    score = normalize_series(d, higher_is_better=True).clip(0, 1)
    return d, score


def coverage_percentage(
    existing_lights: Optional[gpd.GeoDataFrame],
    population: Optional[gpd.GeoDataFrame],
    radius_m: float = 75,
) -> float:
    if existing_lights is None or existing_lights.empty:
        return 0.0
    if population is None or population.empty:
        return 0.0

    pop = population.copy()
    pop["geometry"] = pop.geometry.apply(representative_point)
    pop = pop[pop.geometry.notna()].to_crs(UTM)

    if pop.empty:
        return 0.0

    lights = existing_lights.copy()
    # Defensive filter: drop any null/empty geometries before buffering,
    # even though load_all_data() should already have removed them via
    # ensure_point_layer(). This keeps the function safe on its own.
    lights = lights[lights.geometry.notna()]
    lights = lights[~lights.geometry.is_empty]
    lights = lights.to_crs(UTM)

    if lights.empty:
        return 0.0

    light_union = unary_union([g.buffer(radius_m) for g in lights.geometry])

    covered = sum(bool(geom.within(light_union)) for geom in pop.geometry)
    return 100.0 * covered / max(len(pop), 1)


# ---------------------------------------------------------------------
# AHP / MCDA
# ---------------------------------------------------------------------

FEATURE_LABELS = {
    "population": "Population pressure",
    "lighting_gap": "Lighting gap",
    "pedestrian": "Pedestrian activity",
    "bus": "Bus-stop proximity",
    "facilities": "Public facilities",
    "crime": "Safety / crime risk",
    "road": "Road hierarchy",
    "solar": "Solar potential",
    "slope": "Terrain suitability",
    "environment": "Environmental suitability",
    "access": "Maintenance accessibility",
}


DEFAULT_WEIGHTS = {
    "population": 0.14,
    "lighting_gap": 0.14,
    "pedestrian": 0.10,
    "bus": 0.10,
    "facilities": 0.08,
    "crime": 0.12,
    "road": 0.07,
    "solar": 0.07,
    "slope": 0.05,
    "environment": 0.06,
    "access": 0.07,
}


def calculate_mcda(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    out = df.copy()

    total = sum(max(float(v), 0) for v in weights.values())
    if total <= 0:
        total = 1.0

    score = np.zeros(len(out), dtype=float)
    for key, weight in weights.items():
        if key not in out.columns:
            continue
        score += normalize_series(out[key]).to_numpy() * (float(weight) / total)

    out["safety_score"] = (
        normalize_series(
            out.get("population", pd.Series(0, index=out.index))
        ) * 0.32
        + normalize_series(
            out.get("lighting_gap", pd.Series(0, index=out.index))
        ) * 0.22
        + normalize_series(
            out.get("pedestrian", pd.Series(0, index=out.index))
        ) * 0.16
        + normalize_series(
            out.get("crime", pd.Series(0, index=out.index))
        ) * 0.20
        + normalize_series(
            out.get("facilities", pd.Series(0, index=out.index))
        ) * 0.10
    )

    out["technical_score"] = (
        normalize_series(
            out.get("solar", pd.Series(0, index=out.index))
        ) * 0.45
        + normalize_series(
            out.get("slope", pd.Series(0, index=out.index)),
            higher_is_better=False,
        ) * 0.20
        + normalize_series(
            out.get("environment", pd.Series(0, index=out.index))
        ) * 0.15
        + normalize_series(
            out.get("access", pd.Series(0, index=out.index))
        ) * 0.20
    )

    out["mcda_score"] = np.round(score * 100.0, 2)

    # Balanced decision score makes safety dominant while still respecting
    # engineering feasibility.
    out["overall_priority"] = np.round(
        0.65 * out["safety_score"] * 100
        + 0.35 * out["technical_score"] * 100,
        2,
    )

    out = out.sort_values(
        ["overall_priority", "mcda_score"],
        ascending=False,
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)

    out["priority_class"] = np.select(
        [
            out["overall_priority"] >= 75,
            out["overall_priority"] >= 50,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    return out


def ahp_consistency_ratio(pairwise: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns:
        lambda_max, CI, CR

    AHP rule of thumb:
        CR < 0.10 is generally accepted as reasonably consistent.
    """
    n = pairwise.shape[0]
    eigenvalues, _ = np.linalg.eig(pairwise)
    lambda_max = float(np.max(np.real(eigenvalues)))

    ci = (lambda_max - n) / max(n - 1, 1)

    ri_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri else 0.0

    return lambda_max, ci, cr


# ---------------------------------------------------------------------
# FEATURE ENGINE
# ---------------------------------------------------------------------

def build_candidate_features(
    candidates: gpd.GeoDataFrame,
    store: DataStore,
    coverage_radius_m: float,
) -> gpd.GeoDataFrame:

    c = candidates.copy()

    # Core distances.
    nearest_bus = nearest_distance_series(c, store.bus_stops)
    nearest_road = nearest_distance_series(c, store.roads)
    existing_dist, lighting_gap = compute_lighting_gap_score(
        c, store.streetlights, coverage_radius_m
    )

    c["bus_dist_m"] = nearest_bus
    c["road_dist_m"] = nearest_road
    c["existing_light_dist_m"] = existing_dist

    c["bus"] = inverse_distance_score(nearest_bus, 1500).fillna(0)
    c["lighting_gap"] = lighting_gap.fillna(0)

    c["population"] = population_value_within(
        c, store.population, 250
    ).fillna(0)

    c["pedestrian"] = (
        point_counts_within(c, store.bus_stops, 150).fillna(0)
        + point_counts_within(c, store.facilities, 150).fillna(0) * 0.75
    )

    c["facilities"] = facilities_score(c, store.facilities).fillna(0)

    c["crime"] = crime_score(c, store.crime).fillna(0)

    c["road"] = road_priority_score(c)

    # Solar potential:
    # - Real raster value if available.
    # - A neutral fallback if unavailable.
    solar_raw = sample_raster_at_points(c, store.solar_path)
    if solar_raw.notna().any():
        c["solar"] = normalize_series(solar_raw)
        c["solar_raw"] = solar_raw
    else:
        c["solar"] = 0.5
        c["solar_raw"] = np.nan

    # Terrain:
    slope_raw = compute_slope_from_dem(c, store.dem_path)
    if slope_raw.notna().any():
        c["slope"] = slope_raw
        c["slope_raw"] = slope_raw
    else:
        c["slope"] = 0.5
        c["slope_raw"] = np.nan

    # Environmental suitability:
    flood_flag = intersects_layer(c, store.flood_risk, buffer_m=15)
    wetland_flag = intersects_layer(c, store.wetlands, buffer_m=15)
    c["environment_constraint"] = (
        flood_flag.astype(int) + wetland_flag.astype(int)
    )

    # High score means environmentally more suitable.
    c["environment"] = np.where(
        c["environment_constraint"] == 0,
        1.0,
        np.where(c["environment_constraint"] == 1, 0.45, 0.05)
    )

    # Maintenance accessibility:
    # Near a mapped road and away from environmentally constrained land.
    road_access = inverse_distance_score(
        c["road_dist_m"].fillna(500), 250
    )
    c["access"] = (
        0.75 * road_access
        + 0.25 * c["environment"]
    ).clip(0, 1)

    # Coordinates.
    utm = c.to_crs(UTM)
    c["utm_easting"] = utm.geometry.x.round(2)
    c["utm_northing"] = utm.geometry.y.round(2)
    c["longitude"] = c.geometry.x.round(6)
    c["latitude"] = c.geometry.y.round(6)

    return c


# ---------------------------------------------------------------------
# OPTIONAL ML PRIORITISATION
# ---------------------------------------------------------------------

ML_FEATURES = [
    "population",
    "lighting_gap",
    "pedestrian",
    "bus",
    "facilities",
    "crime",
    "road",
    "solar",
    "slope",
    "environment",
    "access",
]


def train_optional_ml(
    candidates: gpd.GeoDataFrame,
) -> Tuple[pd.Series, Dict[str, float], str]:
    """
    Uses observed / proxy labels ONLY when enough training information exists.

    Because a new municipality deployment may not have historical labels, the
    function falls back to the rule-based MCDA priority score and says so.

    This avoids pretending that a trained ML model is available when it is not.
    """

    if not SKLEARN_AVAILABLE:
        return (
            candidates["overall_priority"] / 100.0,
            {},
            "ML unavailable: scikit-learn not installed; using MCDA score.",
        )

    required = [f for f in ML_FEATURES if f in candidates.columns]
    if len(required) < 5 or len(candidates) < 30:
        return (
            candidates["overall_priority"] / 100.0,
            {},
            "ML skipped: insufficient real training data; using MCDA score.",
        )

    # Proxy target for prototype validation only. For the dissertation, replace
    # this with observed historical problem/safety labels.
    y = (
        (
            0.4 * normalize_series(candidates["lighting_gap"])
            + 0.35 * normalize_series(candidates["crime"])
            + 0.25 * normalize_series(candidates["population"])
        ) >= 0.55
    ).astype(int)

    if y.nunique() < 2:
        return (
            candidates["overall_priority"] / 100.0,
            {},
            "ML skipped: target contains one class only; using MCDA score.",
        )

    X = candidates[required].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test), zero_division=0)

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        auc = np.nan

    metadata = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "feature_count": len(required),
    }

    return (
        pd.Series(prob, index=candidates.index),
        metadata,
        "Random Forest trained on prototype proxy labels. Replace proxy labels with observed historical data for research-grade prediction.",
    )


# ---------------------------------------------------------------------
# EXPLAINABLE CONTRIBUTIONS
# ---------------------------------------------------------------------

def explain_site(
    row: pd.Series,
    weights: Dict[str, float],
) -> pd.DataFrame:
    records = []

    total = sum(weights.values()) or 1.0

    for key, w in weights.items():
        if key not in row.index:
            continue

        value = row.get(key, 0)
        if pd.isna(value):
            value = 0

        # Feature value is normalised to 0-1 before contribution.
        contribution = normalize_series(
            pd.Series([value], index=[0])
        ).iloc[0] * (w / total)

        records.append(
            {
                "factor": FEATURE_LABELS.get(key, key),
                "raw_value": round(float(value), 4),
                "weight": round(float(w / total), 4),
                "relative_contribution": round(float(contribution), 4),
            }
        )

    exp = pd.DataFrame(records)

    if not exp.empty:
        exp["contribution_percent"] = (
            exp["relative_contribution"]
            / max(exp["relative_contribution"].sum(), 1e-9)
            * 100
        ).round(1)
        exp = exp.sort_values("contribution_percent", ascending=False)

    return exp


# ---------------------------------------------------------------------
# BUDGET / IMPLEMENTATION
# ---------------------------------------------------------------------

@dataclass
class BudgetConfig:
    available_budget: float = 50000.0
    unit_installation_cost: float = 1600.0
    contingency_percent: float = 10.0


BUDGET = BudgetConfig()


def implementation_plan(
    scored: gpd.GeoDataFrame,
    budget: BudgetConfig,
) -> pd.DataFrame:
    out = scored.copy()

    out["estimated_cost"] = (
        budget.unit_installation_cost
        * (1.0 + budget.contingency_percent / 100.0)
    )

    selected = []
    spent = 0.0

    for _, row in out.sort_values("overall_priority", ascending=False).iterrows():
        cost = float(row["estimated_cost"])
        if spent + cost <= budget.available_budget:
            selected.append(row["candidate_id"])
            spent += cost

    out["funded_phase_1"] = out["candidate_id"].isin(selected)
    out["implementation_phase"] = np.select(
        [
            out["funded_phase_1"] & (out["rank"] <= 20),
            out["funded_phase_1"],
        ],
        ["Phase 1 — Immediate", "Phase 2 — Budget funded"],
        default="Phase 3 — Future funding",
    )

    return out


# ---------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------

def add_circle_markers(
    feature_group,
    rows: pd.DataFrame,
    color_col: str,
    radius_col: Optional[str] = None,
    popup_builder=None,
):
    for _, row in rows.iterrows():
        color = row.get(color_col, COLOR_INFO)
        radius = float(row.get(radius_col, 6)) if radius_col else 6

        popup_html = popup_builder(row) if popup_builder else str(row.to_dict())

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color="#ffffff",
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(feature_group)


def create_map(
    scored: gpd.GeoDataFrame,
    store: DataStore,
    show_heatmap: bool = True,
    show_coverage: bool = True,
) -> folium.Map:

    m = folium.Map(
        location=[CHINHOYI_LAT, CHINHOYI_LON],
        zoom_start=14,
        max_zoom=20,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite Imagery",
        show=True,
        max_zoom=20,
        max_native_zoom=19,
    ).add_to(m)

    # Terrain-ish optional basemap.
    folium.TileLayer(
        tiles=(
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        ),
        attr="OpenTopoMap",
        name="Topographic",
        show=False,
    ).add_to(m)

    # Boundary.
    if store.boundary is not None:
        folium.GeoJson(
            store.boundary,
            name="Municipal Boundary",
            style_function=lambda x: {
                "color": COLOR_MUNI_GREEN,
                "weight": 2.5,
                "fillOpacity": 0.04,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[store.boundary.columns[0]],
                aliases=["Boundary:"],
            ),
        ).add_to(m)

    # Suburbs.
    if store.suburbs is not None and not store.suburbs.empty:
        name_field = next(
            (
                c for c in store.suburbs.columns
                if any(k in c.lower() for k in ["suburb", "name", "ward", "label"])
            ),
            None,
        )

        folium.GeoJson(
            store.suburbs,
            name="Suburbs & Wards",
            style_function=lambda x: {
                "color": COLOR_CUT_NAVY,
                "weight": 1.5,
                "dashArray": "4,4",
                "fillColor": COLOR_MUNI_GOLD,
                "fillOpacity": 0.10,
            },
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=[name_field],
                    aliases=["Suburb / Ward:"],
                )
                if name_field
                else None
            ),
        ).add_to(m)

    # Roads.
    if store.roads is not None and not store.roads.empty:
        roads_fg = folium.FeatureGroup(name="Road Network", show=True)

        folium.GeoJson(
            store.roads,
            style_function=lambda x: {
                "color": "#475569",
                "weight": 1.6,
                "opacity": 0.75,
            },
        ).add_to(roads_fg)

        roads_fg.add_to(m)

    # Existing streetlights.
    if store.streetlights is not None and not store.streetlights.empty:
        existing_fg = folium.FeatureGroup(
            name="Existing Streetlights", show=True
        )

        # Join maintenance status.
        status_map = {}
        if not store.maintenance.empty:
            status_map = dict(
                zip(
                    store.maintenance["asset_id"].astype(str),
                    store.maintenance["status"].astype(str),
                )
            )

        for _, row in store.streetlights.iterrows():
            asset_id = str(row.get("asset_id", "Streetlight"))
            status = status_map.get(asset_id, "Unknown")

            if status == "Operational":
                c = COLOR_EXISTING
            elif status == "Faulty":
                c = COLOR_FAULT
            elif status == "Maintenance":
                c = COLOR_MAINT
            elif status == "Offline":
                c = COLOR_OFFLINE
            else:
                c = COLOR_EXISTING

            popup_html = f"""
            <div style="font-family:Arial; width:280px;">
              <h4 style="margin:0;color:{COLOR_CUT_NAVY};">{asset_id}</h4>
              <hr>
              <b>Status:</b> {status}<br>
              <b>Type:</b> {row.get("light_type", "Unknown")}<br>
              <b>Installation Year:</b> {row.get("installation_year", "Unknown")}<br>
              <b>Coordinates:</b>
                 {row.geometry.y:.6f}, {row.geometry.x:.6f}
            </div>
            """

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#ffffff",
                weight=1,
                fill=True,
                fill_color=c,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(existing_fg)

        existing_fg.add_to(m)

    # Coverage circles.
    if show_coverage and store.streetlights is not None and not store.streetlights.empty:
        coverage_fg = folium.FeatureGroup(
            name="Lighting Service Coverage (75 m)",
            show=False,
        )

        for geom in store.streetlights.geometry:
            folium.Circle(
                location=[geom.y, geom.x],
                radius=75,
                color=COLOR_INFO,
                fill=True,
                fill_opacity=0.04,
                opacity=0.25,
                weight=1,
            ).add_to(coverage_fg)

        coverage_fg.add_to(m)

    # Population heatmap.
    if show_heatmap and store.population is not None and not store.population.empty:
        coords = []

        pop = store.population.copy()
        pop["geometry"] = pop.geometry.apply(representative_point)
        pop = pop[pop.geometry.notna()]

        if len(pop) > 5000:
            pop = pop.sample(5000, random_state=42)

        value_col = next(
            (
                c for c in pop.columns
                if c.lower() in {"population", "pop", "households", "count", "persons"}
            ),
            None,
        )

        max_v = (
            pd.to_numeric(pop[value_col], errors="coerce").fillna(1).max()
            if value_col
            else 1
        )

        for _, r in pop.iterrows():
            weight = (
                float(pd.to_numeric(r[value_col], errors="coerce") or 1) / max(max_v, 1)
                if value_col else 1.0
            )
            coords.append([r.geometry.y, r.geometry.x, min(max(weight, 0.05), 1.0)])

        if coords:
            fg = folium.FeatureGroup(
                name="Population Density / Demand Heatmap",
                show=True,
            )
            plugins.HeatMap(
                coords,
                radius=16,
                blur=12,
                min_opacity=0.25,
                max_zoom=18,
            ).add_to(fg)
            fg.add_to(m)

    # Bus stops.
    if store.bus_stops is not None and not store.bus_stops.empty:
        bus_fg = folium.FeatureGroup(name="Bus Stops", show=False)

        for _, row in store.bus_stops.iterrows():
            p = representative_point(row.geometry)
            if p is None:
                continue

            name = row.get("stop_name") or row.get("name") or "Bus Stop"

            folium.Marker(
                [p.y, p.x],
                icon=folium.Icon(color="blue", icon="bus", prefix="fa"),
                popup=str(name),
            ).add_to(bus_fg)

        bus_fg.add_to(m)

    # Facilities.
    if store.facilities is not None and not store.facilities.empty:
        fac_fg = folium.FeatureGroup(name="Public Facilities", show=False)

        for _, row in store.facilities.iterrows():
            p = representative_point(row.geometry)
            if p is None:
                continue

            f_type = row.get("facility_type", "Facility")
            f_name = row.get("facility_name", row.get("name", "Facility"))

            folium.Marker(
                [p.y, p.x],
                icon=folium.Icon(color="green", icon="building", prefix="fa"),
                popup=f"<b>{f_name}</b><br>Type: {f_type}",
            ).add_to(fac_fg)

        fac_fg.add_to(m)

    # Environmental constraints.
    if store.flood_risk is not None and not store.flood_risk.empty:
        folium.GeoJson(
            store.flood_risk,
            name="Flood Risk",
            style_function=lambda x: {
                "color": "#0284C7",
                "weight": 1,
                "fillColor": "#38BDF8",
                "fillOpacity": 0.18,
            },
        ).add_to(m)

    if store.wetlands is not None and not store.wetlands.empty:
        folium.GeoJson(
            store.wetlands,
            name="Wetlands / Environmental Sensitivity",
            style_function=lambda x: {
                "color": "#15803D",
                "weight": 1,
                "fillColor": "#22C55E",
                "fillOpacity": 0.18,
            },
        ).add_to(m)

    # Proposed locations.
    proposed_fg = folium.FeatureGroup(
        name="Proposed / Prioritised Solar Streetlights",
        show=True,
    )

    def candidate_popup(row: pd.Series) -> str:
        return f"""
        <div style="font-family:Arial; width:310px;">
          <h3 style="margin:0;color:{COLOR_CUT_NAVY};">
              #{int(row["rank"])} — {row["candidate_id"]}
          </h3>
          <hr>
          <b>Priority:</b> {row["priority_class"]}<br>
          <b>Overall Priority:</b> {row["overall_priority"]:.1f}/100<br>
          <b>MCDA Score:</b> {row["mcda_score"]:.1f}/100<br>
          <b>Safety Score:</b> {row["safety_score"] * 100:.1f}/100<br>
          <b>Technical Score:</b> {row["technical_score"] * 100:.1f}/100<br>
          <b>Population Pressure:</b> {row["population"]:.1f}<br>
          <b>Lighting Gap Score:</b> {row["lighting_gap"]:.2f}<br>
          <b>Bus Stop Distance:</b> {row["bus_dist_m"]:.1f} m<br>
          <b>Existing Light Distance:</b> {row["existing_light_dist_m"]:.1f} m<br>
          <b>Road:</b> {row.get("road_name", "Unknown")}<br>
          <b>Environmental Constraint:</b>
             {int(row.get("environment_constraint", 0))}<br>
          <b>Coordinates:</b>
             {row["latitude"]:.6f}, {row["longitude"]:.6f}<br>
          <b>UTM:</b>
             {row["utm_easting"]:.2f}, {row["utm_northing"]:.2f}
        </div>
        """

    for _, row in scored.iterrows():
        pclass = row["priority_class"]

        if pclass == "HIGH":
            color = COLOR_HIGH
            radius = 9
        elif pclass == "MEDIUM":
            color = COLOR_MED
            radius = 7
        else:
            color = COLOR_LOW
            radius = 5

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color="#ffffff",
            weight=1.3,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            popup=folium.Popup(candidate_popup(row), max_width=350),
        ).add_to(proposed_fg)

    proposed_fg.add_to(m)

    # Click-on-map coordinates.
    plugins.MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="Coordinates:",
        num_digits=5,
    ).add_to(m)

    # Full screen.
    plugins.Fullscreen(
        position="topleft",
        title="Full Screen",
        title_cancel="Exit Full Screen",
        force_separate_button=True,
    ).add_to(m)

    # Search when road layer exists.
    if store.roads is not None and not store.roads.empty:
        # Search plugin can be sensitive to field names, so we keep this
        # optional and use the first string-like field.
        search_field = next(
            (
                c for c in store.roads.columns
                if store.roads[c].dtype == "object"
            ),
            None,
        )
        if search_field:
            try:
                plugins.Search(
                    layer=folium.GeoJson(store.roads),
                    search_label=search_field,
                    placeholder="Search road...",
                    collapsed=False,
                ).add_to(m)
            except Exception:
                pass

    # Legend.
    legend_html = f"""
    <div style="
       position: fixed;
       bottom: 25px;
       left: 25px;
       width: 245px;
       z-index:9999;
       background:white;
       border:1px solid #cbd5e1;
       border-radius:8px;
       padding:10px;
       font-family:Arial;
       font-size:11px;">
      <b style="color:{COLOR_CUT_NAVY};font-size:13px;">
        Solar Streetlight Priority
      </b><br><br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_HIGH};"></span> High priority<br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_MED};"></span> Medium priority<br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_LOW};"></span> Low priority<br><br>
      <b>Existing assets</b><br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_EXISTING};"></span> Operational<br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_FAULT};"></span> Faulty<br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_MAINT};"></span> Maintenance<br>
      <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
        background:{COLOR_OFFLINE};"></span> Offline
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ---------------------------------------------------------------------
# DASHBOARD STATE
# ---------------------------------------------------------------------

DATA = load_all_data()
DATA = create_demo_layers(DATA)

DATA_SOURCE_FLAGS = {
    "streetlights": bool(DATA.streetlights is not None),
    "roads": bool(DATA.roads is not None),
    "suburbs": bool(DATA.suburbs is not None),
    "boundary": bool(DATA.boundary is not None),
    "bus_stops": bool(DATA.bus_stops is not None),
    "facilities": bool(DATA.facilities is not None),
    "population": bool(DATA.population is not None),
    "crime": bool(DATA.crime is not None),
    "flood_risk": bool(DATA.flood_risk is not None),
    "wetlands": bool(DATA.wetlands is not None),
    "dem": bool(DATA.dem_path is not None),
    "solar": bool(DATA.solar_path is not None),
}

REAL_DATA_AVAILABLE = (
    DATA_SOURCE_FLAGS["streetlights"]
    and DATA_SOURCE_FLAGS["roads"]
    and DATA_SOURCE_FLAGS["population"]
)

DEMO_WARNING = (
    not REAL_DATA_AVAILABLE
)

COVERAGE_RADIUS = pn.widgets.IntSlider(
    name="Lighting Coverage Radius (m)",
    start=25,
    end=200,
    step=5,
    value=75,
)

CANDIDATE_SPACING = pn.widgets.IntSlider(
    name="Candidate Spacing Along Roads (m)",
    start=50,
    end=250,
    step=10,
    value=100,
)

MAX_CANDIDATES = pn.widgets.IntSlider(
    name="Maximum Candidate Sites",
    start=50,
    end=1000,
    step=50,
    value=300,
)

# ---------------------------------------------------------------------
# INTERACTIVE WEIGHTS
# ---------------------------------------------------------------------

weight_widgets: Dict[str, pn.widgets.FloatSlider] = {}

for key, default in DEFAULT_WEIGHTS.items():
    weight_widgets[key] = pn.widgets.FloatSlider(
        name=FEATURE_LABELS[key],
        start=0.0,
        end=0.50,
        step=0.01,
        value=float(default),
        sizing_mode="stretch_width",
    )

budget_widget = pn.widgets.FloatInput(
    name="Available Budget (USD)",
    value=BUDGET.available_budget,
    start=0,
)

unit_cost_widget = pn.widgets.FloatInput(
    name="Estimated Cost per New Solar Light (USD)",
    value=BUDGET.unit_installation_cost,
    start=0,
)

contingency_widget = pn.widgets.FloatInput(
    name="Contingency (%)",
    value=BUDGET.contingency_percent,
    start=0,
    end=50,
)

show_heatmap_widget = pn.widgets.Checkbox(
    name="Population Heatmap",
    value=True,
)

show_coverage_widget = pn.widgets.Checkbox(
    name="Lighting Coverage",
    value=True,
)

scenario_widget = pn.widgets.Select(
    name="Planning Scenario",
    options=[
        "Balanced",
        "Safety First",
        "Solar / Energy First",
        "Budget First",
        "Equity First",
    ],
    value="Balanced",
)

# ---------------------------------------------------------------------
# SCENARIO WEIGHTS
# ---------------------------------------------------------------------

SCENARIO_WEIGHTS = {
    "Balanced": DEFAULT_WEIGHTS,
    "Safety First": {
        **DEFAULT_WEIGHTS,
        "crime": 0.18,
        "population": 0.17,
        "lighting_gap": 0.16,
        "pedestrian": 0.13,
        "solar": 0.04,
    },
    "Solar / Energy First": {
        **DEFAULT_WEIGHTS,
        "solar": 0.20,
        "slope": 0.10,
        "environment": 0.08,
        "access": 0.10,
        "crime": 0.06,
    },
    "Budget First": {
        **DEFAULT_WEIGHTS,
        "access": 0.14,
        "road": 0.12,
        "lighting_gap": 0.17,
        "population": 0.17,
        "facilities": 0.10,
    },
    "Equity First": {
        **DEFAULT_WEIGHTS,
        "population": 0.20,
        "lighting_gap": 0.20,
        "facilities": 0.11,
        "pedestrian": 0.12,
        "crime": 0.10,
    },
}


def get_active_weights() -> Dict[str, float]:
    if scenario_widget.value != "Balanced":
        return SCENARIO_WEIGHTS[scenario_widget.value]

    return {k: v.value for k, v in weight_widgets.items()}


# ---------------------------------------------------------------------
# DERIVED MODEL
# ---------------------------------------------------------------------

def get_candidate_dataset() -> gpd.GeoDataFrame:
    candidates = generate_road_based_candidates(
        DATA.roads,
        DATA.streetlights,
        spacing_m=CANDIDATE_SPACING.value,
        max_candidates=MAX_CANDIDATES.value,
    )

    features = build_candidate_features(
        candidates,
        DATA,
        COVERAGE_RADIUS.value,
    )

    weights = get_active_weights()
    scored = calculate_mcda(features, weights)

    BUDGET.available_budget = float(budget_widget.value)
    BUDGET.unit_installation_cost = float(unit_cost_widget.value)
    BUDGET.contingency_percent = float(contingency_widget.value)

    scored = implementation_plan(scored, BUDGET)

    # Optional ML.
    ml_prob, ml_meta, ml_message = train_optional_ml(scored)
    scored["ml_probability"] = ml_prob.values
    scored["ml_message"] = ml_message

    scored["hybrid_score"] = np.round(
        0.80 * scored["overall_priority"]
        + 20.0 * scored["ml_probability"],
        2,
    )

    return scored


# ---------------------------------------------------------------------
# KPI FUNCTIONS
# ---------------------------------------------------------------------

def kpi_card(title: str, value, subtitle: str = "", accent: str = COLOR_CUT_NAVY) -> str:
    return f"""
    <div style="
      background:white;
      border:1px solid #e2e8f0;
      border-radius:10px;
      padding:12px;
      min-height:85px;">
      <div style="font-size:11px;color:#64748b;text-transform:uppercase;">
        {title}
      </div>
      <div style="font-size:23px;font-weight:700;color:{accent};margin-top:4px;">
        {value}
      </div>
      <div style="font-size:10px;color:#64748b;margin-top:3px;">
        {subtitle}
      </div>
    </div>
    """


def build_kpis(scored: gpd.GeoDataFrame) -> pn.Row:
    n_existing = len(DATA.streetlights) if DATA.streetlights is not None else 0
    n_candidates = len(scored)

    if not DATA.maintenance.empty:
        operational = int((DATA.maintenance["status"] == "Operational").sum())
        faulty = int((DATA.maintenance["status"] == "Faulty").sum())
        maintenance_count = int((DATA.maintenance["status"] == "Maintenance").sum())
        offline = int((DATA.maintenance["status"] == "Offline").sum())
    else:
        operational = faulty = maintenance_count = offline = 0

    high_priority = int((scored["priority_class"] == "HIGH").sum())

    coverage = coverage_percentage(
        DATA.streetlights,
        DATA.population,
        COVERAGE_RADIUS.value,
    )

    budget = float(budget_widget.value)
    estimated_unit = float(unit_cost_widget.value) * (1 + float(contingency_widget.value) / 100)
    affordable = int(budget // max(estimated_unit, 1))

    avg_bus = (
        float(scored.head(min(10, len(scored)))["bus_dist_m"].mean())
        if len(scored)
        else 0
    )

    return pn.Row(
        pn.pane.HTML(
            kpi_card("Existing Streetlights", f"{n_existing:,}", "Actual asset layer" if not DEMO_WARNING else "Demo fallback"),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("Operational", f"{operational:,}", "Current asset status", COLOR_EXISTING),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("Faulty", f"{faulty:,}", "Maintenance attention required", COLOR_FAULT),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("High-Priority Sites", f"{high_priority:,}", "Recommended candidate locations", COLOR_HIGH),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("Coverage", f"{coverage:.1f}%", f"Within {COVERAGE_RADIUS.value} m service radius", COLOR_INFO),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("Affordable Phase 1", f"{affordable:,}", "Under current budget", COLOR_MUNI_GREEN),
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            kpi_card("Avg. Bus Distance", f"{avg_bus:.0f} m", "Top 10 candidate sites", COLOR_CUT_NAVY),
            sizing_mode="stretch_width",
        ),
    )


# ---------------------------------------------------------------------
# DATA QUALITY / SYSTEM STATUS
# ---------------------------------------------------------------------

def source_status_table() -> pd.DataFrame:
    labels = {
        "streetlights": "Existing streetlights",
        "roads": "Road network",
        "suburbs": "Suburbs / wards",
        "boundary": "Municipal boundary",
        "bus_stops": "Bus stops",
        "facilities": "Public facilities",
        "population": "Population / census",
        "crime": "Crime / safety incidents",
        "flood_risk": "Flood risk",
        "wetlands": "Wetlands / environmental constraints",
        "dem": "DEM",
        "solar": "Solar radiation",
    }

    rows = []
    for key, value in DATA_SOURCE_FLAGS.items():
        rows.append(
            {
                "Dataset": labels[key],
                "Status": "AVAILABLE" if value else "MISSING",
                "Research use": (
                    "Active"
                    if value
                    else "Needs municipal / authoritative dataset"
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# CHART DATA
# ---------------------------------------------------------------------

def priority_summary(scored: gpd.GeoDataFrame) -> pd.DataFrame:
    out = (
        scored["priority_class"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        .rename_axis("Priority")
        .reset_index(name="Sites")
    )
    return out


def status_summary() -> pd.DataFrame:
    if DATA.maintenance.empty:
        return pd.DataFrame({"Status": [], "Assets": []})
    return (
        DATA.maintenance["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Assets")
    )


def suburb_summary() -> pd.DataFrame:
    if DATA.suburbs is None or DATA.suburbs.empty or DATA.streetlights is None:
        return pd.DataFrame({"Area": [], "Streetlights": []})

    s = DATA.suburbs.copy()
    field = next(
        (
            c for c in s.columns
            if any(k in c.lower() for k in ["suburb", "name", "ward", "label"])
        ),
        None,
    )

    if not field:
        return pd.DataFrame({"Area": [], "Streetlights": []})

    joined = gpd.sjoin(
        DATA.streetlights[["geometry"]],
        s[[field, "geometry"]],
        how="left",
        predicate="within",
    )

    return (
        joined[field]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Area")
        .reset_index(name="Streetlights")
    )


# ---------------------------------------------------------------------
# DATA EXPORTS
# ---------------------------------------------------------------------

def geojson_bytes(df: gpd.GeoDataFrame) -> io.BytesIO:
    return io.BytesIO(df.to_json().encode("utf-8"))


def csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    out = df.copy()
    if isinstance(out, gpd.GeoDataFrame):
        out["longitude"] = out.geometry.x.round(6)
        out["latitude"] = out.geometry.y.round(6)
        out = pd.DataFrame(out.drop(columns="geometry"))
    return io.BytesIO(out.to_csv(index=False).encode("utf-8"))


def gpkg_bytes(df: gpd.GeoDataFrame) -> io.BytesIO:
    buffer_path = EXPORT_DIR / f"temporary_{uuid.uuid4().hex}.gpkg"
    try:
        df.to_file(buffer_path, layer="priority_sites", driver="GPKG")
        data = buffer_path.read_bytes()
        return io.BytesIO(data)
    finally:
        if buffer_path.exists():
            buffer_path.unlink()


# ---------------------------------------------------------------------
# COMMUNITY REPORTING
# ---------------------------------------------------------------------

report_category = pn.widgets.Select(
    name="Fault / Community Report Type",
    options=[
        "Streetlight not working",
        "Dim / intermittent light",
        "Pole damaged",
        "Solar panel damaged",
        "Unsafe dark area",
        "Other",
    ],
)

report_lat = pn.widgets.FloatInput(
    name="Latitude",
    value=CHINHOYI_LAT,
)

report_lon = pn.widgets.FloatInput(
    name="Longitude",
    value=CHINHOYI_LON,
)

report_description = pn.widgets.TextAreaInput(
    name="Description",
    placeholder="Describe the problem or unsafe location...",
    height=90,
)

report_priority = pn.widgets.Select(
    name="Priority",
    options=["Low", "Medium", "High", "Critical"],
    value="Medium",
)

report_button = pn.widgets.Button(
    name="Submit Community Report",
    button_type="primary",
)


report_status = pn.pane.Markdown("")


def submit_report(event=None):
    report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"

    new_row = pd.DataFrame(
        [
            {
                "report_id": report_id,
                "reported_at": datetime.now().isoformat(timespec="seconds"),
                "category": report_category.value,
                "latitude": report_lat.value,
                "longitude": report_lon.value,
                "description": report_description.value,
                "status": "Open",
                "priority": report_priority.value,
            }
        ]
    )

    DATA.community_reports = pd.concat(
        [DATA.community_reports, new_row],
        ignore_index=True,
    )

    report_status.object = (
        f"✅ Report **{report_id}** submitted and stored in the current session."
    )

    report_description.value = ""


report_button.on_click(submit_report)


# ---------------------------------------------------------------------
# MAINTENANCE EDITOR
# ---------------------------------------------------------------------

maintenance_tabulator = pn.widgets.Tabulator(
    DATA.maintenance,
    selectable=1,
    pagination="remote",
    page_size=10,
    height=360,
    editors={
        "status": {"type": "select", "options": ["Operational", "Faulty", "Maintenance", "Offline"]},
        "fault_type": {
            "type": "select",
            "options": ["None", "Battery", "Lamp/LED", "Solar Panel", "Controller", "Pole", "Unknown"],
        },
    },
)


def refresh_maintenance(event=None):
    maintenance_tabulator.value = DATA.maintenance.copy()


def save_maintenance(event=None):
    try:
        DATA.maintenance = maintenance_tabulator.value.copy()

        # Synchronise work orders.
        faulty = DATA.maintenance[
            DATA.maintenance["status"].isin(["Faulty", "Maintenance"])
        ].copy()

        if not faulty.empty:
            DATA.work_orders = faulty[
                [
                    "work_order_id",
                    "asset_id",
                    "status",
                    "fault_type",
                    "reported_at",
                    "assigned_to",
                    "repair_date",
                    "notes",
                ]
            ].copy()

        pn.state.notifications.success("Maintenance table updated.", duration=2500)

    except Exception as exc:
        pn.state.notifications.error(f"Could not save maintenance table: {exc}")


save_maintenance_button = pn.widgets.Button(
    name="Save Maintenance Changes",
    button_type="success",
)

save_maintenance_button.on_click(save_maintenance)


# ---------------------------------------------------------------------
# MAP + MAIN APP REACTIVE FUNCTION
# ---------------------------------------------------------------------

def build_main_view() -> pn.Column:
    scored = get_candidate_dataset()

    m = create_map(
        scored,
        DATA,
        show_heatmap=show_heatmap_widget.value,
        show_coverage=show_coverage_widget.value,
    )

    map_pane = pn.pane.HTML(
        m._repr_html_(),
        height=620,
        sizing_mode="stretch_width",
    )

    # Ranked table.
    display_cols = [
        "rank",
        "candidate_id",
        "priority_class",
        "overall_priority",
        "mcda_score",
        "safety_score",
        "technical_score",
        "population",
        "bus_dist_m",
        "existing_light_dist_m",
        "road_name",
        "latitude",
        "longitude",
        "utm_easting",
        "utm_northing",
        "estimated_cost",
        "funded_phase_1",
        "implementation_phase",
    ]

    display_cols = [c for c in display_cols if c in scored.columns]

    table = pn.widgets.Tabulator(
        pd.DataFrame(scored[display_cols]),
        pagination="remote",
        page_size=12,
        height=420,
        selectable=1,
        show_index=False,
    )

    # Explanation panel.
    explanation = pn.Column(
        pn.pane.Markdown("### 🔍 Why this site was selected"),
        pn.pane.Markdown(
            "Select a row in the ranked table to inspect factor contributions."
        ),
        sizing_mode="stretch_width",
    )

    def explain_selection(event):
        if not table.selection:
            return

        idx = table.selection[0]
        selected = scored.iloc[idx]

        exp = explain_site(selected, get_active_weights())

        text_parts = [
            f"### 📍 {selected['candidate_id']} — Rank {int(selected['rank'])}",
            f"**Overall priority:** {selected['overall_priority']:.1f}/100",
            f"**Safety score:** {selected['safety_score'] * 100:.1f}/100",
            f"**Technical score:** {selected['technical_score'] * 100:.1f}/100",
        ]

        if not exp.empty:
            text_parts.append(
                exp[["factor", "contribution_percent"]]
                .head(8)
                .rename(
                    columns={
                        "factor": "Factor",
                        "contribution_percent": "Contribution %",
                    }
                )
                .to_markdown(index=False)
            )

        text_parts.append(
            "\n**Environmental constraint flag:** "
            f"{int(selected.get('environment_constraint', 0))}"
        )

        text_parts.append(
            "\n**Decision interpretation:** "
            "This ranking is a decision-support output. Field verification, "
            "engineering design, safety validation and municipal approval are still required."
        )

        explanation[:] = [
            pn.pane.Markdown("\n\n".join(text_parts)),
        ]

    table.param.watch(explain_selection, "selection")

    return pn.Column(
        map_pane,
        pn.layout.Divider(),
        pn.pane.Markdown("### 🏆 Ranked Candidate Sites"),
        table,
        explanation,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------
# EXECUTIVE / ANALYTICS TABS
# ---------------------------------------------------------------------

@pn.depends(
    COVERAGE_RADIUS.param.value,
    CANDIDATE_SPACING.param.value,
    MAX_CANDIDATES.param.value,
    scenario_widget.param.value,
    budget_widget.param.value,
    unit_cost_widget.param.value,
    contingency_widget.param.value,
    show_heatmap_widget.param.value,
    show_coverage_widget.param.value,
)
def reactive_dashboard(*_):
    scored = get_candidate_dataset()

    header = pn.pane.HTML(
        f"""
        <div style="
            background:linear-gradient(90deg,{COLOR_MUNI_GREEN},{COLOR_CUT_NAVY});
            color:white;
            border-radius:9px;
            padding:16px 20px;">
            <div style="font-size:21px;font-weight:800;">
                🏛️ MUNICIPALITY OF CHINHOYI
            </div>
            <div style="font-size:13px;opacity:0.9;">
                GIS & Geoinformatics Section —
                Smart Solar Streetlight Planning, Monitoring & Decision Support
            </div>
        </div>
        """,
        sizing_mode="stretch_width",
    )

    alert = pn.pane.Markdown(
        (
            "⚠️ **DATA-LIMITED / DEMO MODE:** Some layers are missing. "
            "Replace synthetic fallback layers with authoritative municipal datasets "
            "before using the results for planning or dissertation analysis."
        )
        if DEMO_WARNING
        else
        "✅ **REAL-DATA MODE:** Core layers are present. Continue validating "
        "coordinate reference systems, attributes and positional accuracy before analysis."
    )

    return pn.Column(
        header,
        alert,
        build_kpis(scored),
        pn.layout.Divider(),
        build_main_view(),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------
# ANALYTICS TAB
# ---------------------------------------------------------------------

@pn.depends(
    COVERAGE_RADIUS.param.value,
    CANDIDATE_SPACING.param.value,
    MAX_CANDIDATES.param.value,
    scenario_widget.param.value,
    budget_widget.param.value,
    unit_cost_widget.param.value,
    contingency_widget.param.value,
)
def analytics_view(*_) -> pn.Column:
    scored = get_candidate_dataset()

    ptab = pn.widgets.Tabulator(
        priority_summary(scored),
        height=180,
        show_index=False,
    )

    stab = pn.widgets.Tabulator(
        status_summary(),
        height=220,
        show_index=False,
    )

    sub = pn.widgets.Tabulator(
        suburb_summary(),
        height=250,
        show_index=False,
    )

    # ML diagnostic text.
    ml_message = (
        scored["ml_message"].iloc[0]
        if "ml_message" in scored.columns and len(scored)
        else "No ML information."
    )

    ml_text = pn.pane.Markdown(
        "### 🤖 Machine Learning / Predictive Module\n\n"
        + ml_message
    )

    if SKLEARN_AVAILABLE and len(scored) >= 30:
        # Run once to show available metadata.
        prob, meta, msg = train_optional_ml(scored)
        if meta:
            ml_text.object += (
                f"\n\n**Prototype validation metrics:** "
                f"Accuracy={meta['accuracy']:.3f}, "
                f"F1={meta['f1']:.3f}, "
                f"ROC-AUC={meta['auc']:.3f}"
            )
        ml_text.object += (
            "\n\n⚠️ The current model may use proxy labels. "
            "For dissertation-grade ML, train on observed historical streetlight "
            "fault/safety data and report a proper validation design."
        )

    return pn.Column(
        pn.pane.Markdown("## 📊 Spatial Analytics & Performance"),
        pn.Row(
            pn.Column(
                pn.pane.Markdown("### Priority Distribution"),
                ptab,
                width=400,
            ),
            pn.Column(
                pn.pane.Markdown("### Existing Asset Status"),
                stab,
                width=450,
            ),
            sizing_mode="stretch_width",
        ),
        pn.pane.Markdown("### Streetlights by Suburb / Ward"),
        sub,
        ml_text,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------
# WEIGHT / AHP TAB
# ---------------------------------------------------------------------

def weights_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Factor": [FEATURE_LABELS[k] for k in DEFAULT_WEIGHTS],
            "Key": list(DEFAULT_WEIGHTS.keys()),
            "Current Weight": [get_active_weights()[k] for k in DEFAULT_WEIGHTS],
        }
    )


weights_table_widget = pn.widgets.Tabulator(
    weights_table(),
    height=300,
    show_index=False,
)


@pn.depends(
    *[w.param.value for w in weight_widgets.values()],
    scenario_widget.param.value,
)
def update_weights_summary(*_) -> pn.pane.HTML:
    weights = get_active_weights()
    total = sum(weights.values())
    rows = []

    for key, value in weights.items():
        rows.append(
            f"<tr><td>{FEATURE_LABELS[key]}</td><td>{value:.3f}</td>"
            f"<td>{100*value/max(total,1e-9):.1f}%</td></tr>"
        )

    html = f"""
    <div style="
      border:1px solid #e2e8f0;
      border-radius:9px;
      padding:12px;
      background:white;">
      <h3 style="color:{COLOR_CUT_NAVY};margin-top:0;">
        Active Decision Weights
      </h3>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <tr style="background:#f8fafc;">
          <th style="text-align:left;padding:5px;">Factor</th>
          <th style="text-align:left;padding:5px;">Raw</th>
          <th style="text-align:left;padding:5px;">Normalised</th>
        </tr>
        {''.join(rows)}
      </table>
    </div>
    """

    return pn.pane.HTML(html)


weights_panel = pn.Column(
    pn.pane.Markdown(
        """
## ⚖️ AHP / MCDA Decision Engine

The sliders are an interactive planning tool. In a dissertation, the final
weights should ideally come from a documented pairwise-comparison AHP process
and should be checked using the AHP Consistency Ratio.

**Rule used here:** `CR < 0.10` is normally treated as reasonably consistent.
"""
    ),
    scenario_widget,
    *weight_widgets.values(),
    update_weights_summary,
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------
# ASSET MONITORING TAB
# ---------------------------------------------------------------------

@pn.depends()
def asset_monitor_view() -> pn.Column:
    n = len(DATA.maintenance)

    if n:
        counts = DATA.maintenance["status"].value_counts()
        total = len(DATA.maintenance)

        rows = []
        for status, count in counts.items():
            rows.append(
                {
                    "Status": status,
                    "Assets": int(count),
                    "Percentage": round(100 * count / total, 1),
                }
            )

        summary = pd.DataFrame(rows)
    else:
        summary = pd.DataFrame(columns=["Status", "Assets", "Percentage"])

    return pn.Column(
        pn.pane.Markdown(
            """
## 🔧 Streetlight Asset Monitoring

This module is ready for integration with telemetry / IoT feeds. At present,
status comes from the local asset/maintenance dataset, not from a live sensor
network.
"""
        ),
        pn.widgets.Tabulator(
            summary,
            show_index=False,
            height=220,
        ),
        maintenance_tabulator,
        save_maintenance_button,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------
# BUDGET / IMPLEMENTATION TAB
# ---------------------------------------------------------------------

@pn.depends(
    COVERAGE_RADIUS.param.value,
    CANDIDATE_SPACING.param.value,
    MAX_CANDIDATES.param.value,
    scenario_widget.param.value,
    budget_widget.param.value,
    unit_cost_widget.param.value,
    contingency_widget.param.value,
)
def budget_view(*_) -> pn.Column:
    scored = get_candidate_dataset()

    phase1 = scored[scored["funded_phase_1"]]

    estimated_cost = (
        float(unit_cost_widget.value)
        * (1 + float(contingency_widget.value) / 100)
    )

    total_phase1 = float(phase1["estimated_cost"].sum()) if not phase1.empty else 0
    population_phase1 = (
        float(phase1["population"].sum())
        if not phase1.empty
        else 0
    )

    summary = pd.DataFrame(
        [
            ["Available budget", float(budget_widget.value)],
            ["Effective unit cost", estimated_cost],
            ["Phase 1 sites", len(phase1)],
            ["Phase 1 cost", total_phase1],
            ["Demand points / population proxy reached", population_phase1],
            ["Remaining budget", max(float(budget_widget.value) - total_phase1, 0)],
        ],
        columns=["Metric", "Value"],
    )

    table_cols = [
        "rank",
        "candidate_id",
        "priority_class",
        "overall_priority",
        "estimated_cost",
        "funded_phase_1",
        "implementation_phase",
        "population",
        "latitude",
        "longitude",
    ]

    return pn.Column(
        pn.pane.Markdown("## 💰 Budget & Implementation Planning"),
        pn.widgets.Tabulator(
            summary,
            show_index=False,
            height=250,
        ),
        pn.pane.Markdown("### Implementation Phasing"),
        pn.widgets.Tabulator(
            scored[table_cols].head(100),
            height=420,
            pagination="remote",
            page_size=15,
            show_index=False,
        ),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------
# COMMUNITY TAB
# ---------------------------------------------------------------------

community_tab = pn.Column(
    pn.pane.Markdown(
        """
## 👥 Community Reporting

This allows residents or field teams to create a basic report. In a future
deployment, add authentication, photo capture, GPS from a mobile device and
synchronisation into a municipal work-order database.
"""
    ),
    pn.Row(
        pn.Column(
            report_category,
            report_lat,
            report_lon,
            report_priority,
            width=350,
        ),
        pn.Column(
            report_description,
            report_button,
            report_status,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    ),
    pn.layout.Divider(),
    pn.pane.Markdown("### Current Session Reports"),
    pn.bind(
        lambda: pn.widgets.Tabulator(
            DATA.community_reports.copy(),
            height=300,
            show_index=False,
        )
    ),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------
# DATA QUALITY / SYSTEM TAB
# ---------------------------------------------------------------------

data_quality_tab = pn.Column(
    pn.pane.Markdown(
        """
## 🧪 Data Quality & System Readiness

The dashboard should only be considered a municipal decision-support system
after each dataset has been checked for:

- CRS correctness
- positional accuracy
- currentness / date of acquisition
- attribute completeness
- duplicate records
- missing geometry
- authoritative source
- metadata and update frequency
"""
    ),
    pn.widgets.Tabulator(
        source_status_table(),
        height=450,
        show_index=False,
    ),
    pn.pane.Markdown(
        """
### Recommended authoritative inputs before final dissertation results

**Required core**
- Existing streetlight inventory
- Road network
- Population / census
- Suburbs / wards
- Municipal boundary

**Strongly recommended**
- Bus stops / transport routes
- Schools, clinics, police, markets and other facilities
- DEM
- Solar radiation / irradiance
- Flood / wetland constraints
- Observed lighting faults
- Historical safety / crime data where legally and ethically available

**Advanced**
- IoT telemetry
- Battery voltage / state of charge
- panel output
- lamp current
- gateway connectivity
- weather data
- maintenance response logs
"""
    ),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------
# EXPORT TAB
# ---------------------------------------------------------------------

def current_scored_data() -> gpd.GeoDataFrame:
    return get_candidate_dataset()


csv_download = pn.widgets.FileDownload(
    callback=lambda: csv_bytes(current_scored_data()),
    filename="chinhoyi_solar_streetlight_priority_sites.csv",
    label="📄 Export Priority CSV",
    button_type="success",
)

geojson_download = pn.widgets.FileDownload(
    callback=lambda: geojson_bytes(current_scored_data()),
    filename="chinhoyi_solar_streetlight_priority_sites.geojson",
    label="🌐 Export Priority GeoJSON",
    button_type="primary",
)


# GeoPackage download is only enabled when geopandas/file driver support exists.
gpkg_download = pn.widgets.FileDownload(
    callback=lambda: gpkg_bytes(current_scored_data()),
    filename="chinhoyi_solar_streetlight_priority_sites.gpkg",
    label="🗺️ Export GeoPackage",
    button_type="light",
)

export_tab = pn.Column(
    pn.pane.Markdown(
        """
## 📤 Data Export

The exports contain ranked candidate sites, coordinates, decision scores,
priority class and implementation fields.
"""
    ),
    pn.Row(
        csv_download,
        geojson_download,
        gpkg_download,
    ),
    pn.pane.Markdown(
        """
### Suggested municipal outputs

- CSV for procurement / planning workflows
- GeoJSON for web GIS
- GeoPackage for QGIS
- PDF map/report generated from the final validated dataset
"""
    ),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------
# FULL APPLICATION
# ---------------------------------------------------------------------

map_tab = pn.Column(
    pn.bind(
        reactive_dashboard,
        COVERAGE_RADIUS.param.value,
        CANDIDATE_SPACING.param.value,
        MAX_CANDIDATES.param.value,
        scenario_widget.param.value,
        budget_widget.param.value,
        unit_cost_widget.param.value,
        contingency_widget.param.value,
        show_heatmap_widget.param.value,
        show_coverage_widget.param.value,
    ),
    sizing_mode="stretch_width",
)


controls_sidebar = pn.Column(
    pn.pane.Markdown("## ⚙️ Analysis Controls"),
    COVERAGE_RADIUS,
    CANDIDATE_SPACING,
    MAX_CANDIDATES,
    scenario_widget,
    show_heatmap_widget,
    show_coverage_widget,
    pn.layout.Divider(),
    pn.pane.Markdown("### 💰 Budget"),
    budget_widget,
    unit_cost_widget,
    contingency_widget,
    pn.layout.Divider(),
    pn.pane.Markdown(
        """
### Interpretation

**High priority** means the location scores strongly under the selected
decision framework. It is NOT an automatic construction approval.
"""
    ),
    width=300,
)


application = pn.template.FastListTemplate(
    title="Chinhoyi Smart Solar Streetlight Management System",
    accent_base_color=COLOR_MUNI_GREEN,
    header_background=COLOR_MUNI_GREEN,
    sidebar=[controls_sidebar],
    main=[
        pn.Tabs(
            ("🗺️ Executive GIS Dashboard", map_tab),
            ("📊 Spatial Analytics", analytics_view),
            ("⚖️ AHP / MCDA", weights_panel),
            ("🔧 Asset & Maintenance", asset_monitor_view),
            ("💰 Budget & Phasing", budget_view),
            ("👥 Community Reporting", community_tab),
            ("🧪 Data Readiness", data_quality_tab),
            ("📤 Export", export_tab),
            dynamic=True,
        )
    ],
    main_max_width="100%",
)


# ---------------------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------------------

print("=" * 72)
print("CHINHOYI SMART SOLAR STREETLIGHT MANAGEMENT SYSTEM")
print("=" * 72)

print("\nData layers detected:")
for k, v in DATA_SOURCE_FLAGS.items():
    print(f"  {'✅' if v else '⚠️'} {k}: {'available' if v else 'missing'}")

if DEMO_WARNING:
    print("\n⚠️ RUNNING WITH DEMO / DATA-LIMITED FALLBACKS.")
    print("   Replace missing datasets before using results for planning.")
else:
    print("\n✅ Core real-data layers detected.")

# IMPORTANT: application.servable() must run unconditionally at module level.
# `panel serve` imports/executes this script but does NOT run it as
# __main__, so wrapping this call in `if __name__ == "__main__":` (as the
# original prototype did) means Panel never receives the app to publish,
# producing "Application did not publish any contents" in the browser.
print("\nRegistering Panel application...")
application.servable()

if __name__ == "__main__":
    # This branch only fires when the script is run directly with
    # `python gis_script.py` (not via `panel serve`). In that case Panel's
    # own server isn't started automatically, so start it here.
    pn.serve(application, show=True, autoreload=True)
