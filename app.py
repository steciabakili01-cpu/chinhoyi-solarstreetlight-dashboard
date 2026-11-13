"""
Chinhoyi Smart Solar Streetlight Management & Decision-Support Dashboard
=======================================================================
Role-Based Municipal Web GIS System — Municipality of Chinhoyi, Zimbabwe

Roles
-----
ADMIN / MUNICIPAL STAFF / COMMUNITY

Run
---
python GIS_SCRIPT.py
    (or)  panel serve GIS_SCRIPT.py --port $PORT --address 0.0.0.0

Streetlight data
----------------
Put the streetlight layer in data/ (e.g. streetlights.gpkg / .geojson / .shp /
.parquet, or a .csv with latitude + longitude columns). Optional columns the
dashboard will use when present: a unique ID (asset_id, streetlight_id, id...),
status (status / operational_status / condition ...), suburb, light_type,
install_date, fault_description, last_inspection, last_maintenance,
maintenance_notes, service_status. Anything absent is shown as unavailable.

Objective 2 — AHP / MCDA selection of NEW solar streetlights
-------------------------------------------------------------
Staff pages "AHP Criteria Weights" and "MCDA Site Selection". The weights of the
11 siting criteria are derived with the Analytic Hierarchy Process (pairwise
comparison matrix -> principal eigenvector, consistency ratio CR < 0.10), and
are then used in a weighted linear combination to score and rank candidate
sites along the road network. The recommended new lights are chosen down the
ranking subject to a flood/wetland exclusion, a minimum spacing between new
lights, and the budget. To make YOUR expert judgements the starting point for
every session, save the pairwise matrix as data/ahp_pairwise.csv (use the
"Download matrix" button on the AHP page to get the template).
Environment variables: MIN_NEW_LIGHT_SPACING_M (default 60).

Default development logins
--------------------------
admin / Admin@123          (override with ADMIN_DEFAULT_PASSWORD)
staff / Staff@123          (override with STAFF_DEFAULT_PASSWORD)
resident / Resident@123    (override with COMMUNITY_DEFAULT_PASSWORD)
Replace these before any real municipal deployment.

Brand asset
-----------
The interface carries ONE logo: the Municipality of Chinhoyi. Place the approved
artwork in
    assets/municipality_logo.png   (or .svg / .jpg)
or point to it with MUNICIPAL_LOGO_PATH. A white background around a PNG/JPG logo is
removed automatically (needs Pillow; without Pillow the image is shown as is). When no
real file is found an original, clearly generic placeholder badge is shown — never a
copy of an official crest.

Optional dependencies
----------------------
rasterio, scikit-learn, Pillow and the anthropic SDK are optional. Their absence never
crashes the application — the affected features report that they are unavailable.
The interactive map needs Bokeh, which is installed with Panel.

BUILD HISTORY
-------------
Earlier builds (condensed): one independent AppContext/session per browser session;
"real data detected" flags captured BEFORE demo layers are generated; community users only
ever see their own reports; hardened empty-input paths; passwords/budget/costing in
environment variables; structured logging; permission and scoring caches; CSV/GeoJSON/
GeoPackage export; separate staff and community home pages; a role-scoped assistant; one
streetlight registry (STREETLIGHT_REGISTRY) feeding every KPI, with statuses read from a
real column when one exists (random statuses only as a clearly-labelled SIMULATED fallback);
Objective 2 — AHP weighting + MCDA selection of new solar lights, with sensitivity analysis;
clickable KPI cards opening the Streetlight Explorer (filterable list, WGS84 + UTM 36S
coordinates, CSV / GeoJSON download).

UPDATE 4 (this build)
---------------------
1.  MAP RESTORED, WITHOUT FOLIUM. An interactive Bokeh map (Panel's own charting library)
    now sits on the Streetlight Explorer page and on every role's home page. Click a
    streetlight to see its details; click a row in the list to highlight and zoom to it. The
    streetlight points, roads and municipal boundary are drawn by Bokeh itself, so the map is
    never blank; the street basemap tiles load from the internet when available.
    Residents see only public fields (ID, status, suburb, light type). A resident who clicks
    a light can report a problem with THAT light in one step.
2.  USER SESSIONS SAVED FOR EVERY ROLE. A sign-in creates a server-side session (stored hashed
    in SQLite) and a browser cookie, so a page refresh or a dropped connection returns the
    person to where they were, without signing in again. Each user's working state (page,
    list filter and search, weighting method, budget and spacing settings, AHP judgements,
    selected streetlight) is saved automatically and restored at the next sign-in, also on
    another browser. Sessions slide-expire after SESSION_TTL_HOURS (default 12) and are revoked
    on logout, when an account is disabled or its password reset, or by an administrator
    (Audit & Security page).
3.  PROFESSIONAL REDESIGN, NO DUPLICATION. Chinhoyi University of Technology branding and
    the logo's white box are removed. The Municipality logo appears once (top bar) on a
    transparent background. The duplicated title bar, sidebar brand panel, "Signed in" panel,
    repeated role badges, header assistant shortcut and double footer are gone: one top bar,
    one sidebar of navigation, one footer.
4.  REPORT A PROBLEM ON THE RESIDENT HOME PAGE. The report form is embedded in the resident
    home page (and kept as its own page). Reports can name the streetlight concerned
    (community_reports.asset_id, added by an automatic database migration); picking a light
    on the map pre-fills the form.

UPDATE 5
--------
1.  Municipality logo moved to the CENTRE of the top bar (user on the left, actions on the right).
2.  Basemaps: "Satellite hybrid (Google)" (imagery + roads and place labels, the default) and
    "OpenStreetMap", switchable on every map. Neither needs an API key. The Google tile endpoint is
    unofficial and its terms restrict use outside Google's own products, and OSM's public tile
    servers are for light use — move to a licensed or self-hosted basemap before a public rollout.
3.  Informational banners (blue / brown / green boxes: data-mode, AHP-method, inventory
    reconciliation, attribute coverage, prioritisation notes, export and assistant notes, the two
    status cards on the staff dashboard) are moved off the working pages to a new admin page,
    "System Notices", and the log. Only compact coloured pills in the top bar (DEMO DATA,
    SIMULATED STATUS, AHP PROVISIONAL) remain, so a reader is never misled about the data.
4.  Top bar colours are set inline (vivid green with a gold edge, gold role pill, gold
    notifications and red sign-out buttons) so they render even when page CSS does not reach them.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import secrets
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import panel as pn
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chinhoyi_gis")

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    rasterio = None
    RASTERIO_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (reserved for future predictive-maintenance work)
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
    LLM_AVAILABLE = bool(ANTHROPIC_KEY)
except Exception:
    anthropic = None
    ANTHROPIC_KEY = None
    LLM_AVAILABLE = False

try:
    from bokeh.models import ColumnDataSource, HoverTool, Range1d, TapTool, WMTSTileSource
    from bokeh.plotting import figure as bokeh_figure
    BOKEH_AVAILABLE = True
except Exception as _bokeh_exc:  # pragma: no cover
    BOKEH_AVAILABLE = False
    logger.warning("Bokeh could not be imported (%s); the interactive map is disabled.", _bokeh_exc)

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _get_anthropic_client():
    if not LLM_AVAILABLE:
        return None
    try:
        return anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    except Exception as exc:
        logger.warning("Anthropic client could not be created, falling back to template assistant: %s", exc)
        return None


ANTHROPIC_CLIENT = _get_anthropic_client()

# =====================================================================
# APPLICATION CONFIGURATION
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"
ASSETS_DIR = BASE_DIR / "assets"
DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "app_data.db")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CHINHOYI_LAT = -17.3667
CHINHOYI_LON = 30.2000
WGS84 = "EPSG:4326"
UTM = "EPSG:32736"
TO_WGS = Transformer.from_crs(UTM, WGS84, always_xy=True)
TO_UTM = Transformer.from_crs(WGS84, UTM, always_xy=True)

# Configurable planning defaults — tunable per deployment without a code change.
DEFAULT_BUDGET_USD = float(os.environ.get("DEFAULT_BUDGET_USD", "50000"))
DEFAULT_UNIT_COST_USD = float(os.environ.get("DEFAULT_UNIT_COST_USD", "1600"))
DEFAULT_CONTINGENCY_PCT = float(os.environ.get("DEFAULT_CONTINGENCY_PCT", "10"))
DEFAULT_COVERAGE_RADIUS_M = int(float(os.environ.get("DEFAULT_COVERAGE_RADIUS_M", "75")))
DEFAULT_MAX_CANDIDATES = int(float(os.environ.get("DEFAULT_MAX_CANDIDATES", "250")))
CANDIDATE_SPACING_M = float(os.environ.get("CANDIDATE_SPACING_M", "100"))

# Session handling
SESSION_COOKIE = "moc_session"
SESSION_TTL_HOURS = max(float(os.environ.get("SESSION_TTL_HOURS", "12")), 0.1)

# Palette: civic green + gold, navy for text and accents. NOT verified official brand colours —
# swap in the Municipality's guideline values if available.
COLOR_GREEN = "#0B6B3A"
COLOR_GREEN_DARK = "#064E2C"
COLOR_GREEN_SOFT = "#EAF7EF"
COLOR_GOLD = "#F4B41A"
COLOR_NAVY = "#1B3A6B"
COLOR_NAVY_DARK = "#0B1F3D"
COLOR_SKY = "#1976D2"
# "EB Garamond" (the open Google Fonts revival) is loaded explicitly and listed first, because
# the proprietary "Garamond" is not installed on most machines.
FONT_SERIF = "'EB Garamond', Garamond, 'Cormorant Garamond', Georgia, 'Times New Roman', serif"
GOOGLE_FONT_URL = "https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600&display=swap"
COLOR_RED = "#C62828"
COLOR_ORANGE = "#EF8F00"
COLOR_AMBER = "#D99A00"
COLOR_TEAL = "#0F766E"
COLOR_GRAY = "#64748B"
COLOR_BG = "#F4F7F9"
COLOR_BORDER = "#DFE7EE"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
RASTER_LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_NOT_MUNICIPAL_HINTS = ("university", "cut", "technology", "gis_logo", "project")


def _resolve_municipal_logo() -> Path:
    """Find the Municipality's logo robustly across common asset filenames / formats. Files that
    look like another institution's logo are never picked up by the fuzzy search."""
    env_value = os.environ.get("MUNICIPAL_LOGO_PATH")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate
    preferred = ["municipality_logo.png", "municipality_logo.svg", "municipality_logo.jpg", "municipality_logo.jpeg",
                 "chinhoyi_logo.png", "chinhoyi_logo.jpg", "chinhoyi_logo.jpeg"]
    for name in preferred:
        candidate = ASSETS_DIR / name
        if candidate.exists() and candidate.is_file():
            return candidate
    if ASSETS_DIR.exists():
        scored = []
        for candidate in ASSETS_DIR.iterdir():
            if not (candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            stem = candidate.stem.lower().replace("-", "_").replace(" ", "_")
            if any(h in stem for h in _NOT_MUNICIPAL_HINTS):
                continue
            score = 10 if any(k in stem for k in ("municipality", "chinhoyi", "moc", "coc")) else 0
            if score:
                scored.append((score + (3 if "logo" in stem else 0), candidate.name.lower(), candidate))
        if scored:
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return scored[0][2]
    return ASSETS_DIR / preferred[0]


MUNICIPAL_LOGO = _resolve_municipal_logo()

# Original, generic placeholder badge — NOT a reproduction of the real crest. It renders only
# when no real logo file is found.
MUNICIPALITY_PLACEHOLDER_SVG = """
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Municipality of Chinhoyi placeholder badge">
  <defs><linearGradient id="mocBadgeGrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0E7C47"/><stop offset="1" stop-color="#064E2C"/>
  </linearGradient></defs>
  <circle cx="50" cy="50" r="47" fill="url(#mocBadgeGrad)" stroke="#F4B41A" stroke-width="3"/>
  <g stroke="#F4B41A" stroke-width="2.2" stroke-linecap="round">
    <circle cx="50" cy="23" r="6.5" fill="#F4B41A" stroke="none"/>
    <line x1="50" y1="7" x2="50" y2="13"/>
    <line x1="34" y1="23" x2="40" y2="23"/>
    <line x1="60" y1="23" x2="66" y2="23"/>
    <line x1="38.5" y1="11.5" x2="42.5" y2="15.5"/>
    <line x1="61.5" y1="11.5" x2="57.5" y2="15.5"/>
  </g>
  <polygon points="24,47 50,30 76,47" fill="#FFFFFF" opacity="0.94"/>
  <rect x="27" y="46" width="46" height="32" fill="#FFFFFF" opacity="0.94"/>
  <rect x="34" y="53" width="6.5" height="9" fill="#0E7C47"/>
  <rect x="46.7" y="53" width="6.5" height="9" fill="#0E7C47"/>
  <rect x="59.5" y="53" width="6.5" height="9" fill="#0E7C47"/>
  <rect x="41" y="67" width="18" height="11" fill="#0E7C47"/>
  <text x="50" y="93" font-family="'EB Garamond',Garamond,Georgia,serif" font-size="15" font-weight="800"
        fill="#FFFFFF" text-anchor="middle" letter-spacing="2">MOC</text>
</svg>
""".strip()

# =====================================================================
# PANEL PROFESSIONAL THEME + RESPONSIVE CSS
# =====================================================================

pn.extension(
    "tabulator",
    "chat",
    notifications=True,
    sizing_mode="stretch_width",
    loading_indicator=True,
    # css_files inserts a real <link rel="stylesheet"> into the document head, which is the
    # reliable way to pull in a web font regardless of what is installed on the visitor's machine.
    css_files=[GOOGLE_FONT_URL],
    raw_css=[
        f"""
        :root {{
            --moc-green: {COLOR_GREEN};
            --moc-green-dark: {COLOR_GREEN_DARK};
            --moc-gold: {COLOR_GOLD};
            --moc-navy: {COLOR_NAVY};
            --moc-navy-dark: {COLOR_NAVY_DARK};
            --moc-bg: {COLOR_BG};
            --moc-border: {COLOR_BORDER};
            --moc-gray: {COLOR_GRAY};
            --shadow-sm: 0 2px 10px rgba(16,42,67,.06);
            --shadow-md: 0 8px 25px rgba(16,42,67,.10);
        }}

        html, body, .bk-root {{
            background: var(--moc-bg) !important;
            color: #23384D !important;
            font-family: {FONT_SERIF} !important;
        }}
        /* Form controls do not inherit font-family from the page in any browser. */
        .bk-btn, .bk-input, .bk-select, .bk-textarea, .tabulator, .tabulator-cell, .tabulator-col-title {{
            font-family: {FONT_SERIF} !important;
        }}
        .bk-btn {{ border-radius: 9px !important; font-weight: 700 !important; }}
        .bk-input, .bk-select, .bk-textarea {{ border-radius: 9px !important; }}

        /* ---- single top bar: user | centred logo + name | actions ---- */
        .bk-panel-models-layout-Row.moc-topbar {{
            background: linear-gradient(100deg, #054A28 0%, #0B8A4A 50%, #054A28 100%);
            border-bottom: 4px solid var(--moc-gold);
            padding: 10px 20px; align-items: center; gap: 12px; flex-wrap: nowrap !important;
            box-shadow: 0 4px 18px rgba(5,74,40,.30); color: #fff;
        }}
        .bk-panel-models-layout-Row.moc-topbar > * {{ flex: 1 1 0 !important; min-width: 0 !important; }}
        .bk-panel-models-layout-Row.moc-topbar > .moc-topbar-center {{ flex: 0 0 auto !important; }}
        .moc-topbar-right {{ justify-content: flex-end; gap: 8px; }}
        .moc-topbar-right > * {{ flex: 0 0 auto !important; }}
        .moc-topbar-right > :first-child {{ flex: 1 1 auto !important; }}
        .moc-brand {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; text-align: center; }}
        .moc-logo {{ height: 56px; width: auto; max-width: 160px; object-fit: contain; display: block; background: transparent; }}
        .moc-logo-placeholder {{ display: block; height: 56px; width: 56px; }}
        .moc-logo-placeholder svg {{ width: 100%; height: 100%; display: block; }}
        .moc-logo-lg {{ height: 74px; max-width: 200px; margin: 0 auto; }}
        .moc-logo-placeholder.moc-logo-lg {{ height: 74px; width: 74px; }}
        .moc-topbar-title {{ font-size: 20px; font-weight: 800; letter-spacing: .5px; color: #fff; line-height: 1.15; }}
        .moc-topbar-subtitle {{ font-size: 11.5px; color: #FFE08A; font-weight: 600; margin-top: 1px; }}

        /* ---- shell: sidebar + main ---- */
        .bk-panel-models-layout-Row.moc-shell {{ align-items: flex-start; gap: 14px; padding: 12px 14px; flex-wrap: wrap; }}
        .bk-panel-models-layout-Row.moc-shell > .moc-sidebar {{ flex: 0 0 250px !important; max-width: 100%; }}
        .bk-panel-models-layout-Row.moc-shell > .moc-main {{ flex: 1 1 560px !important; min-width: 0 !important; }}
        .moc-side-head {{ background: #fff; border: 1px solid var(--moc-border); border-radius: 12px; padding: 11px 13px; box-shadow: var(--shadow-sm); margin-bottom: 8px; }}
        .moc-side-title {{ font-size: 13px; font-weight: 800; color: var(--moc-navy); }}
        .moc-side-sub {{ font-size: 10.5px; color: var(--moc-gray); line-height: 1.45; margin-top: 3px; }}
        .moc-side-note {{ font-size: 10px; color: #8190A0; line-height: 1.5; padding: 8px 4px; }}

        .moc-page-head {{
            background:white; border:1px solid var(--moc-border); border-left:4px solid var(--moc-green);
            border-radius:12px; padding:15px 18px; box-shadow:var(--shadow-sm); margin-bottom:12px;
        }}
        .moc-page-title {{ font-size:18px; font-weight:800; color:var(--moc-navy); margin:0; }}
        .moc-page-subtitle {{ font-size:12px; color:var(--moc-gray); margin-top:4px; line-height:1.5; }}

        .moc-card {{
            background:#fff; border:1px solid var(--moc-border); border-radius:12px;
            padding:15px 16px; box-shadow:var(--shadow-sm); height:100%; box-sizing:border-box;
        }}
        .moc-card-title {{ font-size:14px; font-weight:800; color:var(--moc-navy); margin-bottom:5px; }}
        .moc-card-subtitle {{ font-size:11.5px; color:var(--moc-gray); line-height:1.5; }}
        .moc-section-title {{
            color:var(--moc-navy); font-size:14px; font-weight:800; letter-spacing:.25px;
            display:inline-block; border-bottom:3px solid var(--moc-gold); padding-bottom:5px; margin:6px 0 9px;
        }}
        .moc-kpi {{
            background:#fff; border:1px solid var(--moc-border); border-radius:12px; padding:15px 15px 13px;
            min-height:112px; box-shadow:var(--shadow-sm); box-sizing:border-box; border-top:4px solid var(--accent,#0B6B3A);
        }}
        .moc-kpi-label {{ color:#768598; text-transform:uppercase; letter-spacing:.65px; font-size:10px; font-weight:800; }}
        .moc-kpi-value {{ color:var(--accent,#102A43); font-size:27px; line-height:1.05; font-weight:800; margin-top:5px; }}
        .moc-kpi-sub {{ color:#78889A; font-size:10.5px; margin-top:5px; }}

        .moc-kpi-click {{ cursor:pointer; user-select:none; transition:transform .12s ease, box-shadow .12s ease; }}
        .moc-kpi-click:hover {{ transform:translateY(-2px); box-shadow:var(--shadow-md); }}
        .moc-kpi-click:focus-visible {{ outline:3px solid {COLOR_GOLD}; outline-offset:2px; }}
        .moc-kpi-selected {{ background:#FFFBEA; box-shadow:0 0 0 2px {COLOR_GOLD}, var(--shadow-sm); }}
        .moc-kpi-hint {{ margin-top:7px; font-size:10.5px; font-weight:800; letter-spacing:.3px; color:var(--accent,{COLOR_GREEN}); }}
        .moc-kpi-green {{ --accent: {COLOR_GREEN}; }}
        .moc-kpi-teal {{ --accent: {COLOR_TEAL}; }}
        .moc-kpi-red {{ --accent: {COLOR_RED}; }}
        .moc-kpi-orange {{ --accent: {COLOR_ORANGE}; }}
        .moc-kpi-navy {{ --accent: {COLOR_NAVY}; }}

        /* ---- map ---- */
        .moc-legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:11.5px; color:#3A4B5C; padding:2px 2px 8px; }}
        .moc-dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; vertical-align:-1px; border:1.5px solid #fff; box-shadow:0 0 0 1px rgba(0,0,0,.18); }}
        .moc-detail {{ background:#fff; border:1px solid var(--moc-border); border-left:4px solid var(--moc-gold); border-radius:12px; padding:12px 15px; box-shadow:var(--shadow-sm); }}
        .moc-detail-title {{ font-size:15px; font-weight:800; color:var(--moc-navy); display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
        .moc-detail table {{ border-collapse:collapse; margin-top:8px; font-size:12px; width:100%; }}
        .moc-detail td {{ padding:3px 8px 3px 0; vertical-align:top; border-bottom:1px solid #EEF3F6; }}
        .moc-detail td:first-child {{ color:#768598; font-weight:700; white-space:nowrap; width:1%; }}

        /* ---- login ---- */
        .moc-login-shell {{ width:100%; max-width:1060px; margin:22px auto 8px; }}
        .moc-login-card {{ background:#fff; border:1px solid var(--moc-border); border-radius:18px; overflow:hidden; box-shadow:0 18px 50px rgba(16,42,67,.14); }}
        .moc-login-hero {{
            min-height:470px; padding:34px 36px;
            background:linear-gradient(135deg,var(--moc-green-dark) 0%,var(--moc-green) 100%);
            border-bottom:3px solid var(--moc-gold); color:#fff;
            display:flex; flex-direction:column; justify-content:center; box-sizing:border-box;
        }}
        .moc-login-form {{ padding:34px 32px; min-height:470px; display:flex; flex-direction:column; justify-content:center; box-sizing:border-box; }}
        .moc-login-row {{ align-items:stretch !important; gap:0 !important; }}
        .moc-login-title {{ font-size:26px; font-weight:800; margin:18px 0 6px; }}
        .moc-login-sub {{ font-size:12.5px; opacity:.9; line-height:1.6; }}
        .moc-pill {{ display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 10px; font-size:10px; font-weight:800; }}
        .pill-green {{ background:#E8F7ED; color:#12633A; }}
        .pill-blue {{ background:#E8F0FF; color:#244AA7; }}
        .pill-gold {{ background:#FFF7D9; color:#8E6700; }}
        .pill-red {{ background:#FDECEC; color:#9D1F1F; }}
        .pill-onbar {{ background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.35); }}

        .moc-alert {{ border-radius:10px; padding:10px 12px; font-size:12px; border:1px solid; }}
        .moc-alert-info {{ background:#EFF6FF; border-color:#BFDBFE; color:#1D4ED8; }}
        .moc-alert-ok {{ background:#ECFDF3; border-color:#A7F3D0; color:#166534; }}
        .moc-alert-warn {{ background:#FFF7E6; border-color:#F6D487; color:#8A5A00; }}
        .moc-alert-danger {{ background:#FEF2F2; border-color:#FCA5A5; color:#991B1B; }}

        .moc-footer {{ text-align:center; color:#8493A4; font-size:10px; padding:12px 4px 14px; }}
        .tabulator {{ border:1px solid var(--moc-border) !important; border-radius:12px !important; overflow:auto !important; }}
        .tabulator .tabulator-header {{ font-weight:800 !important; }}
        .bk-panel-models-layout-Row {{ flex-wrap:wrap !important; row-gap:12px !important; }}
        .bk-panel-models-layout-Row > * {{ min-width:0 !important; flex:1 1 220px !important; }}
        .bk-panel-models-layout-Row.moc-topbar {{ flex-wrap:nowrap !important; }}

        @media (max-width: 900px) {{
            .moc-login-shell {{ margin:10px auto; padding:0 10px; }}
            .moc-login-hero {{ min-height:320px; padding:28px 24px; }}
            .moc-login-form {{ min-height:0; padding:28px 24px; }}
            .bk-panel-models-layout-Row.moc-topbar {{ flex-wrap:wrap !important; padding:10px 12px; }}
            .moc-topbar-title {{ font-size:16px; }}
            .moc-user-meta {{ text-align:left; }}
            .bk-panel-models-layout-Row.moc-shell > .moc-sidebar {{ flex:1 1 100% !important; }}
        }}
        @media (max-width: 620px) {{
            .bk-panel-models-layout-Row > * {{ flex:1 1 100% !important; }}
            .moc-logo {{ height:40px; }}
            .moc-page-head {{ padding:12px; }}
            .moc-card, .moc-kpi {{ border-radius:10px; }}
            .moc-login-title {{ font-size:21px; }}
        }}
        """
    ],
)

# =====================================================================
# BRAND HELPERS
# =====================================================================

def esc(value) -> str:
    import html
    return html.escape(str(value))


@lru_cache(maxsize=1)
def municipal_logo_uri() -> Optional[str]:
    """data: URI of the Municipality logo, or None when no real file exists. For PNG / JPG / WebP
    files the white background that surrounds the artwork is made transparent (flood-filled from
    the four corners, so white details INSIDE the crest are kept). Needs Pillow; without it the
    file is used unchanged."""
    path = MUNICIPAL_LOGO
    if not path.exists() or not path.is_file():
        return None
    try:
        suffix = path.suffix.lower()
        if suffix in RASTER_LOGO_EXTENSIONS:
            try:
                from PIL import Image, ImageDraw
                img = Image.open(path).convert("RGBA")
                w, h = img.size
                for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
                    px = img.getpixel(corner)
                    if px[3] > 0 and min(px[:3]) >= 232:
                        ImageDraw.floodfill(img, corner, (255, 255, 255, 0), thresh=42)
                buf = io.BytesIO()
                img.save(buf, "PNG")
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except ImportError:
                logger.info("Pillow is not installed; the Municipality logo is shown without background removal.")
            except Exception as exc:
                logger.warning("Logo background removal failed, using the original file: %s", exc)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".svg": "image/svg+xml"}.get(suffix, "application/octet-stream")
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:
        logger.warning("Could not encode the Municipality logo %s: %s", path, exc)
        return None


def logo_html(css_class: str = "") -> str:
    """The Municipality logo on a transparent background (no box, no border)."""
    src = municipal_logo_uri()
    if src:
        return f'<img class="moc-logo {css_class}" src="{src}" alt="Municipality of Chinhoyi" />'
    return (f'<span class="moc-logo moc-logo-placeholder {css_class}" role="img" '
            f'aria-label="Municipality of Chinhoyi placeholder logo">{MUNICIPALITY_PLACEHOLDER_SVG}</span>')


def role_badge(role: str, on_dark: bool = False) -> str:
    values = {
        "ADMIN": ("ADMIN", "pill-red"),
        "MUNICIPAL_STAFF": ("MUNICIPAL STAFF", "pill-blue"),
        "COMMUNITY": ("COMMUNITY", "pill-green"),
    }
    label, css = values.get(role, (role or "GUEST", "pill-gold"))
    if on_dark:
        css = "pill-onbar"
    return f"<span class='moc-pill {css}'>{label}</span>"

# =====================================================================
# GIS DATA ENGINE
# =====================================================================

LAYER_ALIASES = {
    "streetlights": ["streetlights", "street_lights", "existing_streetlights"],
    "roads": ["roads", "road_network", "streets"],
    "suburbs": ["suburbs", "wards"],
    "boundary": ["boundary", "municipal_boundary"],
    "bus_stops": ["bus_stops", "transport_stops"],
    "facilities": ["facilities", "public_facilities"],
    "population": ["population", "census", "households"],
    "crime": ["crime", "incidents"],
    "flood_risk": ["flood", "flood_risk"],
    "wetlands": ["wetland", "wetlands"],
}
PREFERRED_EXTENSION_ORDER = [".parquet", ".gpkg", ".geojson", ".json", ".shp", ".csv"]


def list_spatial_files() -> List[Path]:
    allowed = {".shp", ".gpkg", ".geojson", ".json", ".parquet", ".csv", ".tif", ".tiff"}
    try:
        return sorted([p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in allowed])
    except Exception as exc:
        logger.warning("Could not scan data directory %s: %s", DATA_DIR, exc)
        return []


def find_layer(name: str) -> Optional[Path]:
    matches: List[Path] = []
    aliases = LAYER_ALIASES.get(name, [name])
    for alias in aliases:
        key = alias.lower().replace(" ", "_")
        for path in list_spatial_files():
            stem = path.stem.lower().replace(" ", "_")
            if stem == key or stem.startswith(key) or key in stem:
                if path not in matches:
                    matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: PREFERRED_EXTENSION_ORDER.index(p.suffix.lower()) if p.suffix.lower() in PREFERRED_EXTENSION_ORDER else 99)
    return matches[0]


def find_raster(*keywords: str) -> Optional[Path]:
    for path in list_spatial_files():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stem = path.stem.lower().replace(" ", "_")
        if any(k.lower().replace(" ", "_") in stem for k in keywords):
            return path
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
    maintenance: pd.DataFrame = field(default_factory=pd.DataFrame)
    demo_layers: Set[str] = field(default_factory=set)
    # Where each streetlight's status really came from: "source" (a real column), "simulated"
    # (random demo values) or "unknown".
    status_provenance: Dict[str, Any] = field(default_factory=lambda: {"kind": "unknown", "column": None, "recognised": 0, "unrecognised_values": {}})
    asset_id_source: str = "generated"
    # EVERY streetlight record from the source file, including rows with missing geometry (those
    # are excluded from `streetlights`, which feeds the spatial analysis, but must still be counted).
    streetlights_all: Optional[gpd.GeoDataFrame] = None


def _points_from_lonlat(df: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    lon = next((c for c in df.columns if str(c).lower() in {"lon", "longitude", "x"}), None)
    lat = next((c for c in df.columns if str(c).lower() in {"lat", "latitude", "y"}), None)
    if lon and lat:
        return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon], df[lat]), crs=WGS84)
    return None


def safe_read_vector(path: Optional[Path]) -> Optional[gpd.GeoDataFrame]:
    if path is None or not path.exists():
        return None
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _points_from_lonlat(pd.read_csv(path))
        if suffix == ".parquet":
            try:
                gdf = gpd.read_parquet(path)
            except Exception:
                gdf = _points_from_lonlat(pd.read_parquet(path))
            if gdf is None:
                return None
        else:
            gdf = gpd.read_file(path)
        if gdf is None or gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs(UTM)
        return gdf.to_crs(WGS84)
    except Exception as exc:
        logger.warning("Could not read spatial layer %s: %s", path.name, exc)
        return None


def ensure_geometry(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty or "geometry" not in gdf:
        return None
    try:
        out = gdf[gdf.geometry.notna()].copy()
        out = out[~out.geometry.is_empty].copy()
        out = out[out.geometry.is_valid | out.geometry.notna()].copy()
    except Exception as exc:
        logger.warning("Geometry validation failed, dropping layer: %s", exc)
        return None
    return out if not out.empty else None


def representative_point(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        if geom.geom_type == "Point":
            return geom
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            return geom.representative_point()
        return geom.centroid
    except Exception:
        return None


# ---------------------------------------------------------------------
# Streetlight status classification and attribute discovery
# ---------------------------------------------------------------------
# Every status value — from a real column or from the demonstration fallback — is normalised
# through ONE function (classify_status), and every count is derived from the resulting registry.
SIMULATE_ASSET_STATUS = os.environ.get("SIMULATE_ASSET_STATUS", "true").strip().lower() not in {"0", "false", "no", "off"}

STATUS_OPERATIONAL = "OPERATIONAL"
STATUS_FAULT_OFFLINE = "FAULT_OFFLINE"
STATUS_MAINTENANCE = "MAINTENANCE"
STATUS_UNCLASSIFIED = "UNCLASSIFIED"

STATUS_COLUMN_ALIASES = [
    "status", "operational_status", "light_status", "working_status", "functional_status", "condition",
    "state", "lamp_status", "streetlight_status", "operational", "functional", "working",
]
_STATUS_OPERATIONAL_TERMS = {
    "operational", "operating", "working", "functional", "functioning", "on", "active", "ok", "okay", "good",
    "in service", "in use", "normal", "serviceable", "lit", "yes", "y", "true", "1",
}
_STATUS_FAULT_TERMS = {
    "faulty", "fault", "broken", "defective", "damaged", "failed", "failure", "not working", "out of order",
    "dead", "vandalised", "vandalized", "stolen", "missing", "no", "n", "false", "0",
}
_STATUS_OFFLINE_TERMS = {"offline", "off line", "off", "disconnected", "no power", "inactive", "not lit", "unlit", "dark", "down"}
_STATUS_MAINTENANCE_TERMS = {
    "maintenance", "under maintenance", "in maintenance", "maintenance required", "needs maintenance",
    "scheduled maintenance", "repair", "under repair", "needs repair", "pending repair", "service due",
}
_STATUS_NEGATION = re.compile(r"^(not|non)\s*(operational|working|functional|functioning|operative|in service)\b|^in(operative|operational)\b")

ATTRIBUTE_ALIASES: Dict[str, List[str]] = {
    "suburb": ["suburb", "suburb_name", "ward", "ward_name", "neighbourhood", "neighborhood", "township", "area"],
    "light_type": ["light_type", "lamp_type", "fixture_type", "streetlight_type", "type"],
    "install_date": ["install_date", "installation_date", "date_installed", "installed_on", "installed", "commissioned", "commission_date", "year_installed"],
    "fault_description": ["fault_description", "fault_desc", "fault", "fault_type", "issue", "problem", "defect"],
    "last_inspection": ["last_inspection", "last_inspection_date", "inspection_date", "last_inspected", "last_survey"],
    "last_maintenance": ["last_maintenance", "last_maintenance_date", "last_service", "last_serviced", "maintenance_date", "last_repair"],
    "maintenance_notes": ["maintenance_history", "maintenance_notes", "maintenance", "remarks", "comments", "notes"],
    "service_status": ["service_status", "service_state"],
}
ATTRIBUTE_LABELS = {
    "suburb": "Suburb", "light_type": "Light type", "install_date": "Installation date",
    "fault_description": "Fault description", "last_inspection": "Last inspection",
    "last_maintenance": "Last maintenance", "maintenance_notes": "Maintenance history / notes",
    "service_status": "Service status",
}
ASSET_ID_COLUMN_ALIASES = ["asset_id", "streetlight_id", "street_light_id", "light_id", "lightid", "pole_id", "pole_no", "id", "fid", "objectid", "globalid", "name"]


def _norm_text(value) -> str:
    """Case-, whitespace- and punctuation-insensitive form used to compare status words."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return re.sub(r"\s+", " ", re.sub(r"[_\-/.]+", " ", str(value))).strip().lower()


def classify_status(value) -> Tuple[str, str]:
    """Map a raw status value to (status_class, display_label). Matching is on whole normalised
    values, never substrings, so 'Not operational' can never be mistaken for 'operational'.
    Anything unrecognised is returned as UNCLASSIFIED rather than being guessed at."""
    n = _norm_text(value)
    if not n:
        return STATUS_UNCLASSIFIED, "Unknown"
    if _STATUS_NEGATION.match(n):
        return STATUS_FAULT_OFFLINE, "Faulty"
    if n in _STATUS_OFFLINE_TERMS:
        return STATUS_FAULT_OFFLINE, "Offline"
    if n in _STATUS_FAULT_TERMS:
        return STATUS_FAULT_OFFLINE, "Faulty"
    if n in _STATUS_MAINTENANCE_TERMS:
        return STATUS_MAINTENANCE, "Maintenance"
    if n in _STATUS_OPERATIONAL_TERMS:
        return STATUS_OPERATIONAL, "Operational"
    return STATUS_UNCLASSIFIED, "Unknown"


def _find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns if str(c) != "geometry"}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _find_status_column(df: pd.DataFrame) -> Optional[str]:
    """First recognised status-like column that holds at least one value we can classify."""
    lowered = {str(c).strip().lower(): c for c in df.columns if str(c) != "geometry"}
    for alias in STATUS_COLUMN_ALIASES:
        col = lowered.get(alias)
        if col is None:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        if any(classify_status(v)[0] != STATUS_UNCLASSIFIED for v in values.head(2000)):
            return col
    return None


def _resolve_asset_ids(gdf: gpd.GeoDataFrame) -> Tuple[List[str], str]:
    """Keep the municipality's own asset identifiers when the layer has a complete, unique ID
    column; only fall back to generated IDs otherwise."""
    for alias in ASSET_ID_COLUMN_ALIASES:
        col = _find_column(gdf, [alias])
        if col is None:
            continue
        values = gdf[col].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
        if (values != "").all() and values.is_unique:
            return values.tolist(), f"source column '{col}'"
    return [f"SL-CHN-{i:04d}" for i in range(1, len(gdf) + 1)], "generated (no complete, unique ID column in the streetlight layer)"


def initialize_maintenance_table(streetlights: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the per-asset status table AND report where the statuses came from. A real status
    column always wins; only when none exists are statuses simulated (random, seeded) — and that
    fact is carried in the returned provenance so the UI can say so."""
    columns = ["work_order_id", "asset_id", "status", "status_raw", "fault_type", "reported_at", "assigned_to", "repair_date", "notes"]
    prov: Dict[str, Any] = {"kind": "unknown", "column": None, "recognised": 0, "unrecognised_values": {}}
    n = len(streetlights)
    if n == 0:
        return pd.DataFrame(columns=columns), prov
    ids = streetlights["asset_id"].astype(str).tolist()
    status_col = _find_status_column(streetlights)
    if status_col is not None:
        raw = streetlights[status_col]
        classified = [classify_status(v) for v in raw]
        labels = [c[1] for c in classified]
        unclassified_mask = [c[0] == STATUS_UNCLASSIFIED for c in classified]
        fault_col = _find_column(streetlights, ATTRIBUTE_ALIASES["fault_description"])
        fault_vals = []
        for i, (cls, _) in enumerate(classified):
            v = streetlights[fault_col].iloc[i] if fault_col is not None else None
            text = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
            fault_vals.append(text if text and cls in (STATUS_FAULT_OFFLINE, STATUS_MAINTENANCE) else ("Not recorded" if cls in (STATUS_FAULT_OFFLINE, STATUS_MAINTENANCE) else "None"))
        status_raw = [None if (isinstance(v, float) and np.isnan(v)) else (None if v is None else str(v)) for v in raw]
        unrec = pd.Series([r for r, bad in zip(status_raw, unclassified_mask) if bad], dtype="object").fillna("(blank)").astype(str).str.strip().value_counts().head(10).to_dict()
        prov = {"kind": "source", "column": str(status_col), "recognised": int(n - sum(unclassified_mask)), "unrecognised_values": unrec}
    elif SIMULATE_ASSET_STATUS:
        rng = np.random.default_rng(42)
        labels = list(rng.choice(["Operational", "Operational", "Operational", "Faulty", "Maintenance", "Offline"], n))
        fault_vals = [rng.choice(["Battery", "Lamp/LED", "Solar Panel", "Controller", "Pole"]) if s == "Faulty" else "None" for s in labels]
        status_raw = [None] * n
        prov = {"kind": "simulated", "column": None, "recognised": n, "unrecognised_values": {}}
    else:
        labels = ["Unknown"] * n
        fault_vals = ["None"] * n
        status_raw = [None] * n
    return pd.DataFrame({
        "work_order_id": [f"WO-{i:05d}" for i in range(1, n + 1)],
        "asset_id": ids,
        "status": labels,
        "status_raw": status_raw,
        "fault_type": fault_vals,
        "reported_at": [datetime.now().date().isoformat()] * n,
        "assigned_to": ["Unassigned"] * n,
        "repair_date": [None] * n,
        "notes": [""] * n,
    }), prov


def load_all_data() -> DataStore:
    """Load only genuinely-detected datasets. No fallback/demo data here — that keeps this
    function's output usable as the authoritative "real data detected" signal (RAW_DATA_FLAGS)."""
    store = DataStore()
    raw_streetlights: Optional[gpd.GeoDataFrame] = None
    for name in LAYER_ALIASES:
        raw = safe_read_vector(find_layer(name))
        if name == "streetlights":
            raw_streetlights = raw
        setattr(store, name, ensure_geometry(raw))
    store.dem_path = find_raster("dem", "elevation", "terrain")
    store.solar_path = find_raster("solar", "irradiance", "radiation")
    if raw_streetlights is not None and not raw_streetlights.empty:
        full = raw_streetlights.copy()
        ids, id_source = _resolve_asset_ids(full)
        full["asset_id"] = ids
        store.asset_id_source = id_source
        store.streetlights_all = full
        mappable = ensure_geometry(full)
        # Never fall through to demo streetlights when a REAL file exists but none of its rows can
        # be mapped — keep an empty layer instead.
        store.streetlights = mappable if mappable is not None else full.iloc[0:0].copy()
        store.maintenance, store.status_provenance = initialize_maintenance_table(full)
    return store


def create_demo_layers(store: DataStore) -> DataStore:
    """Fill in missing core layers with clearly-labelled demonstration data so the dashboard stays
    usable when no real municipal dataset is present. Every synthesised layer is recorded in
    store.demo_layers so the UI can flag it accurately."""
    rng = np.random.default_rng(12)
    if store.boundary is None:
        store.boundary = gpd.GeoDataFrame(
            {"name": ["Demo Municipal Boundary"]},
            geometry=[Polygon([(30.145, -17.415), (30.255, -17.415), (30.255, -17.320), (30.145, -17.320)])],
            crs=WGS84,
        )
        store.demo_layers.add("boundary")
    if store.suburbs is None:
        store.suburbs = gpd.GeoDataFrame(
            {"name": ["Demo West", "Demo East"]},
            geometry=[
                Polygon([(30.155, -17.405), (30.205, -17.405), (30.205, -17.355), (30.155, -17.355)]),
                Polygon([(30.205, -17.405), (30.250, -17.405), (30.250, -17.355), (30.205, -17.355)]),
            ],
            crs=WGS84,
        )
        store.demo_layers.add("suburbs")
    if store.roads is None:
        lines, names, types = [], [], []
        for i in range(11):
            y = -17.400 + i * 0.008
            lines.append(LineString([(30.155, y), (30.250, y)])); names.append(f"Demo Road H-{i+1}"); types.append("Collector")
        for i in range(11):
            x = 30.155 + i * 0.0095
            lines.append(LineString([(x, -17.405), (x, -17.350)])); names.append(f"Demo Road V-{i+1}"); types.append("Local")
        store.roads = gpd.GeoDataFrame({"road_name": names, "road_type": types}, geometry=lines, crs=WGS84)
        store.demo_layers.add("roads")
    if store.streetlights is None:
        pts = []
        for i in range(60):
            geom = store.roads.geometry.iloc[i % len(store.roads)]
            pts.append(geom.interpolate(rng.uniform(0.04, 0.96), normalized=True))
        store.streetlights = gpd.GeoDataFrame(
            {"asset_id": [f"SL-CHN-{i:04d}" for i in range(1, len(pts) + 1)], "light_type": ["Solar LED"] * len(pts)},
            geometry=pts,
            crs=WGS84,
        )
        store.maintenance, store.status_provenance = initialize_maintenance_table(store.streetlights)
        store.asset_id_source = "generated (demonstration layer)"
        store.demo_layers.add("streetlights")
    if store.bus_stops is None:
        pts = [Point(30.166, -17.395), Point(30.183, -17.375), Point(30.214, -17.390), Point(30.235, -17.370), Point(30.245, -17.400)]
        store.bus_stops = gpd.GeoDataFrame({"stop_name": [f"Bus Stop {i+1}" for i in range(len(pts))]}, geometry=pts, crs=WGS84)
        store.demo_layers.add("bus_stops")
    if store.facilities is None:
        facs = [(30.174, -17.392, "School"), (30.225, -17.383, "Clinic"), (30.235, -17.399, "Market"), (30.192, -17.368, "School"), (30.215, -17.402, "Police"), (30.245, -17.367, "Clinic")]
        store.facilities = gpd.GeoDataFrame(
            {"facility_name": [f"Facility {i+1}" for i in range(len(facs))], "facility_type": [x[2] for x in facs]},
            geometry=[Point(x[0], x[1]) for x in facs], crs=WGS84,
        )
        store.demo_layers.add("facilities")
    if store.population is None:
        pts = [Point(rng.uniform(30.155, 30.250), rng.uniform(-17.405, -17.350)) for _ in range(700)]
        vals = rng.integers(1, 8, len(pts))
        store.population = gpd.GeoDataFrame({"households": vals}, geometry=pts, crs=WGS84)
        store.demo_layers.add("population")
    return store


# --- Data initialisation -------------------------------------------------
# IMPORTANT: capture "real data detected" flags BEFORE any demo/fallback layer is generated, so the
# dashboard can clearly separate real municipal evidence from demonstration data.
_RAW_STORE = load_all_data()
RAW_DATA_FLAGS: Dict[str, bool] = {name: getattr(_RAW_STORE, name) is not None for name in LAYER_ALIASES}
RAW_DATA_FLAGS["dem"] = _RAW_STORE.dem_path is not None
RAW_DATA_FLAGS["solar"] = _RAW_STORE.solar_path is not None

DATA = create_demo_layers(_RAW_STORE)
DEMO_LAYERS: Set[str] = set(DATA.demo_layers)

DATA_FLAGS: Dict[str, bool] = {
    "streetlights": DATA.streetlights is not None,
    "roads": DATA.roads is not None,
    "suburbs": DATA.suburbs is not None,
    "boundary": DATA.boundary is not None,
    "bus_stops": DATA.bus_stops is not None,
    "facilities": DATA.facilities is not None,
    "population": DATA.population is not None,
    "crime": DATA.crime is not None,
    "flood_risk": DATA.flood_risk is not None,
    "wetlands": DATA.wetlands is not None,
    "dem": DATA.dem_path is not None,
    "solar": DATA.solar_path is not None,
}
CORE_REAL_DATA = bool(RAW_DATA_FLAGS.get("streetlights") and RAW_DATA_FLAGS.get("roads") and RAW_DATA_FLAGS.get("population"))

logger.info("Core real data detected: %s | Demo layers in use: %s", CORE_REAL_DATA, sorted(DEMO_LAYERS) or "none")


# =====================================================================
# STREETLIGHT REGISTRY — the single source of truth for every streetlight KPI. Anything that cannot
# be mapped is counted and reported rather than silently dropped.
# =====================================================================

REGISTRY_COLUMNS = [
    "asset_id", "latitude", "longitude", "coord_valid", "status_class", "status_label", "status_raw",
    "suburb", "suburb_source", "light_type", "install_date", "fault_description", "last_inspection",
    "last_maintenance", "maintenance_notes", "service_status", "duplicate_id", "duplicate_coords",
]


def _clean_cell(value) -> Optional[str]:
    """Render a source attribute for display, or None when it is blank."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat() if not pd.isna(value) else None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "nat", "none", "null"} else None


def build_streetlight_registry(store: DataStore) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One row per streetlight record, with validated coordinates, a normalised status class and
    whichever optional attributes genuinely exist in the source layer, plus data-quality metadata."""
    meta: Dict[str, Any] = {
        "status_provenance": dict(store.status_provenance),
        "asset_id_source": store.asset_id_source,
        "available_fields": [], "missing_fields": [], "suburb_source": None,
    }
    gdf = store.streetlights_all if store.streetlights_all is not None else store.streetlights
    if gdf is None or gdf.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS), meta
    pts = gdf.geometry.apply(representative_point)
    lat = pd.to_numeric(pts.apply(lambda p: p.y if p is not None else np.nan), errors="coerce")
    lon = pd.to_numeric(pts.apply(lambda p: p.x if p is not None else np.nan), errors="coerce")
    valid = (
        np.isfinite(lat) & np.isfinite(lon) & lat.between(-90, 90) & lon.between(-180, 180)
        & ~((lat == 0) & (lon == 0))
    )
    reg = pd.DataFrame({
        "asset_id": gdf["asset_id"].astype(str).values,
        "latitude": lat.values, "longitude": lon.values, "coord_valid": valid.values,
    })
    maint = store.maintenance.drop_duplicates("asset_id").set_index("asset_id") if not store.maintenance.empty else pd.DataFrame()
    labels = reg["asset_id"].map(maint["status"]) if "status" in maint.columns else pd.Series("Unknown", index=reg.index)
    reg["status_label"] = labels.fillna("Unknown").values
    reg["status_class"] = [classify_status(v)[0] for v in reg["status_label"]]
    reg["status_raw"] = reg["asset_id"].map(maint["status_raw"]).values if "status_raw" in maint.columns else None
    for key, aliases in ATTRIBUTE_ALIASES.items():
        col = _find_column(gdf, aliases)
        reg[key] = [_clean_cell(v) for v in gdf[col].values] if col is not None else None
        reg[key] = reg[key].astype(object).where(reg[key].notna(), None)
    if store.status_provenance.get("kind") == "simulated" and "fault_type" in maint.columns:
        sim = reg["asset_id"].map(maint["fault_type"])
        if not reg["fault_description"].notna().any():
            reg["fault_description"] = pd.Series(np.where(sim.notna() & (sim != "None"), sim, None), index=reg.index, dtype=object)
    reg["suburb_source"] = np.where(reg["suburb"].notna(), "source data", None)
    # Suburb from a REAL suburbs polygon layer when the streetlight layer itself has none.
    if reg["suburb"].isna().all() and "suburbs" not in DEMO_LAYERS and store.suburbs is not None and not store.suburbs.empty:
        name_col = _find_column(store.suburbs, ["name", "suburb", "suburb_name", "ward", "ward_name", "area"])
        if name_col is not None and valid.any():
            try:
                pts_gdf = gpd.GeoDataFrame({"_i": reg.index[valid.values]}, geometry=gpd.points_from_xy(reg.loc[valid.values, "longitude"], reg.loc[valid.values, "latitude"]), crs=WGS84)
                joined = gpd.sjoin(pts_gdf, store.suburbs[[name_col, "geometry"]].to_crs(WGS84), how="left", predicate="within")
                joined = joined[~joined.index.duplicated(keep="first")]
                names = joined.set_index("_i")[name_col].map(_clean_cell)
                reg.loc[names.index, "suburb"] = names.values
                reg["suburb_source"] = np.where(reg["suburb"].notna(), "spatial join with suburbs layer", None)
                meta["suburb_source"] = "spatial join with the suburbs layer"
            except Exception as exc:
                logger.warning("Suburb spatial join failed: %s", exc)
    elif reg["suburb"].notna().any():
        meta["suburb_source"] = "streetlight layer attribute"
    reg["duplicate_id"] = reg["asset_id"].duplicated(keep=False)
    coord_key = reg["latitude"].round(6).astype(str) + "," + reg["longitude"].round(6).astype(str)
    reg["duplicate_coords"] = reg["coord_valid"] & coord_key.duplicated(keep=False)
    meta["available_fields"] = [k for k in ATTRIBUTE_ALIASES if reg[k].notna().any()]
    meta["missing_fields"] = [k for k in ATTRIBUTE_ALIASES if not reg[k].notna().any()]
    return reg[REGISTRY_COLUMNS].reset_index(drop=True), meta


STREETLIGHT_REGISTRY, STREETLIGHT_META = build_streetlight_registry(DATA)


def add_utm_columns(reg: pd.DataFrame) -> pd.DataFrame:
    """Add UTM zone 36S easting / northing (metres) for every record with valid coordinates; the
    others keep blank values rather than receiving substitute coordinates."""
    reg = reg.copy()
    reg["utm_easting"] = np.nan
    reg["utm_northing"] = np.nan
    if len(reg) and bool(reg["coord_valid"].any()):
        ok = reg["coord_valid"].to_numpy(dtype=bool)
        try:
            east, north = TO_UTM.transform(reg.loc[ok, "longitude"].to_numpy(dtype=float), reg.loc[ok, "latitude"].to_numpy(dtype=float))
            reg.loc[ok, "utm_easting"] = np.round(np.asarray(east, dtype=float), 2)
            reg.loc[ok, "utm_northing"] = np.round(np.asarray(north, dtype=float), 2)
        except Exception as exc:
            logger.warning("Could not compute UTM coordinates for the streetlight registry: %s", exc)
    return reg


STREETLIGHT_REGISTRY = add_utm_columns(STREETLIGHT_REGISTRY)


def streetlight_summary(reg: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Counts used by every streetlight KPI card."""
    reg = STREETLIGHT_REGISTRY if reg is None else reg
    prov = STREETLIGHT_META.get("status_provenance", {})
    total = int(len(reg))
    by_class = reg["status_class"].value_counts() if total else pd.Series(dtype=int)
    mapped = int(reg["coord_valid"].sum()) if total else 0
    return {
        "total": total, "mapped": mapped, "unmapped": total - mapped,
        "operational": int(by_class.get(STATUS_OPERATIONAL, 0)),
        "fault_offline": int(by_class.get(STATUS_FAULT_OFFLINE, 0)),
        "maintenance": int(by_class.get(STATUS_MAINTENANCE, 0)),
        "unclassified": int(by_class.get(STATUS_UNCLASSIFIED, 0)),
        "duplicate_ids": int(reg["duplicate_id"].sum()) if total else 0,
        "duplicate_coords": int(reg["duplicate_coords"].sum()) if total else 0,
        "status_kind": prov.get("kind", "unknown"), "status_column": prov.get("column"),
        "unrecognised_values": prov.get("unrecognised_values", {}),
    }


def status_source_text(summary: Optional[Dict[str, Any]] = None) -> str:
    summary = summary or streetlight_summary()
    kind = summary["status_kind"]
    if kind == "source":
        return f"status from source column '{summary['status_column']}'"
    if kind == "simulated":
        return "SIMULATED status — no status column in the data"
    return "no status information available"


def normalize_series(series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    s = s.fillna(float(s.median()))
    mn, mx = float(s.min()), float(s.max())
    if math.isclose(mn, mx):
        return pd.Series(0.5, index=s.index)
    out = (s - mn) / (mx - mn)
    return out if higher_is_better else 1 - out


def nearest_distance_series(points: gpd.GeoDataFrame, target: Optional[gpd.GeoDataFrame]) -> pd.Series:
    if target is None or target.empty:
        return pd.Series(np.nan, index=points.index)
    try:
        p = points.to_crs(UTM)
        t = target.copy()
        t["geometry"] = t.geometry.apply(representative_point)
        t = t[t.geometry.notna()].to_crs(UTM)
        if t.empty:
            return pd.Series(np.nan, index=points.index)
        union = unary_union(list(t.geometry))
        return pd.Series([g.distance(union) for g in p.geometry], index=points.index)
    except Exception as exc:
        logger.warning("nearest_distance_series failed, defaulting to NaN: %s", exc)
        return pd.Series(np.nan, index=points.index)


def inverse_distance_score(d: pd.Series, max_d: float) -> pd.Series:
    x = pd.to_numeric(d, errors="coerce").fillna(max_d)
    return (1 - np.minimum(x / max_d, 1)).clip(0, 1)


def population_value_within(cands, pop, radius_m: float) -> pd.Series:
    if pop is None or pop.empty or cands.empty:
        return pd.Series(0.0, index=cands.index)
    try:
        p = cands.to_crs(UTM)
        q = pop.copy(); q["geometry"] = q.geometry.apply(representative_point); q = q[q.geometry.notna()].to_crs(UTM)
        value_col = next((c for c in q.columns if str(c).lower() in {"population", "pop", "persons", "households", "count"}), None)
        coords = np.array([(g.x, g.y) for g in q.geometry]) if not q.empty else np.empty((0, 2))
        vals = pd.to_numeric(q[value_col], errors="coerce").fillna(1).to_numpy() if value_col and not q.empty else np.ones(len(q))
        out = []
        for geom in p.geometry:
            if coords.size == 0:
                out.append(0.0); continue
            d2 = (coords[:, 0] - geom.x) ** 2 + (coords[:, 1] - geom.y) ** 2
            out.append(float(vals[d2 <= radius_m**2].sum()))
        return pd.Series(out, index=cands.index)
    except Exception as exc:
        logger.warning("population_value_within failed: %s", exc)
        return pd.Series(0.0, index=cands.index)


def point_counts_within(points, target, radius_m: float) -> pd.Series:
    if target is None or target.empty or points.empty:
        return pd.Series(0.0, index=points.index)
    try:
        p = points.to_crs(UTM); q = target.copy(); q["geometry"] = q.geometry.apply(representative_point); q = q[q.geometry.notna()].to_crs(UTM)
        coords = np.array([(g.x, g.y) for g in q.geometry]) if not q.empty else np.empty((0, 2))
        out = []
        for geom in p.geometry:
            if coords.size == 0:
                out.append(0.0); continue
            d2 = (coords[:, 0] - geom.x) ** 2 + (coords[:, 1] - geom.y) ** 2
            out.append(float((d2 <= radius_m**2).sum()))
        return pd.Series(out, index=points.index)
    except Exception as exc:
        logger.warning("point_counts_within failed: %s", exc)
        return pd.Series(0.0, index=points.index)


def facilities_score(cands, facs) -> pd.Series:
    if facs is None or facs.empty or cands.empty:
        return pd.Series(0.0, index=cands.index)
    try:
        c = cands.to_crs(UTM); f = facs.copy(); f["geometry"] = f.geometry.apply(representative_point); f = f[f.geometry.notna()].to_crs(UTM)
        types = f.get("facility_type", pd.Series(["Other"] * len(f))).astype(str).str.lower()
        weights = np.where(types.str.contains("police|clinic|hospital|emergency"), 3.0, np.where(types.str.contains("school|college|university"), 2.0, np.where(types.str.contains("market|terminal|transport"), 1.5, 1.0)))
        return pd.Series([float(np.sum(np.exp(-f.geometry.distance(g).to_numpy() / 500.0) * weights)) for g in c.geometry], index=cands.index)
    except Exception as exc:
        logger.warning("facilities_score failed: %s", exc)
        return pd.Series(0.0, index=cands.index)


def crime_score(cands, crime) -> pd.Series:
    if crime is None or crime.empty or cands.empty:
        return pd.Series(0.0, index=cands.index)
    try:
        c = cands.to_crs(UTM); cr = crime.copy(); cr["geometry"] = cr.geometry.apply(representative_point); cr = cr[cr.geometry.notna()].to_crs(UTM)
        col = next((x for x in cr.columns if str(x).lower() in {"count", "frequency", "incidents", "risk", "cases"}), None)
        vals = pd.to_numeric(cr[col], errors="coerce").fillna(1).to_numpy() if col else np.ones(len(cr))
        return pd.Series([float(np.sum(np.exp(-cr.geometry.distance(g).to_numpy() / 400.0) * vals)) for g in c.geometry], index=cands.index)
    except Exception as exc:
        logger.warning("crime_score failed: %s", exc)
        return pd.Series(0.0, index=cands.index)


def sample_raster_at_points(points, path) -> pd.Series:
    if not RASTERIO_AVAILABLE or path is None or not path.exists() or points.empty:
        return pd.Series(np.nan, index=points.index)
    try:
        with rasterio.open(path) as src:
            pts = points.to_crs(src.crs)
            vals = []
            for value in src.sample([(g.x, g.y) for g in pts.geometry]):
                v = float(value[0]) if len(value) else np.nan
                vals.append(np.nan if src.nodata is not None and np.isclose(v, src.nodata) else v)
            return pd.Series(vals, index=points.index)
    except Exception as exc:
        logger.warning("Raster sampling failed for %s: %s", path, exc)
        return pd.Series(np.nan, index=points.index)


def environment_score(cands) -> Tuple[pd.Series, pd.Series]:
    if cands.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        flood = pd.Series(False, index=cands.index)
        wetland = pd.Series(False, index=cands.index)
        if DATA.flood_risk is not None and not DATA.flood_risk.empty:
            union = unary_union(list(DATA.flood_risk.to_crs(UTM).geometry))
            flood = pd.Series([g.buffer(15).intersects(union) for g in cands.to_crs(UTM).geometry], index=cands.index)
        if DATA.wetlands is not None and not DATA.wetlands.empty:
            union = unary_union(list(DATA.wetlands.to_crs(UTM).geometry))
            wetland = pd.Series([g.buffer(15).intersects(union) for g in cands.to_crs(UTM).geometry], index=cands.index)
        constraint = flood.astype(int) + wetland.astype(int)
        score = pd.Series(np.where(constraint == 0, 1.0, np.where(constraint == 1, 0.45, 0.05)), index=cands.index)
        return constraint, score
    except Exception as exc:
        logger.warning("environment_score failed, treating all sites as unconstrained: %s", exc)
        return pd.Series(0, index=cands.index), pd.Series(1.0, index=cands.index)


def generate_candidates(existing, roads, spacing_m: float = CANDIDATE_SPACING_M, max_candidates: int = 450) -> gpd.GeoDataFrame:
    rows = []; counter = 1
    try:
        if roads is not None and not roads.empty:
            road_utm = roads.to_crs(UTM)
            existing_utm = existing.to_crs(UTM) if existing is not None and not existing.empty else None
            for ridx, row in road_utm.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom] if geom.geom_type == "LineString" else []
                road_name = str(row.get("road_name") or row.get("name") or f"Road {ridx}")
                road_type = str(row.get("road_type") or row.get("type") or "Unknown")
                for part in parts:
                    if part.length < 30:
                        continue
                    for distance in np.arange(25, max(part.length, 25), spacing_m):
                        point = part.interpolate(float(distance))
                        nearest = np.inf if existing_utm is None or existing_utm.empty else float(existing_utm.geometry.distance(point).min())
                        if nearest < spacing_m * 0.55:
                            continue
                        lon, lat = TO_WGS.transform(point.x, point.y)
                        rows.append({"candidate_id": f"CSL-{counter:04d}", "road_name": road_name, "road_type": road_type, "distance_existing_light_m": nearest, "geometry": Point(lon, lat)})
                        counter += 1
                        if len(rows) >= max_candidates:
                            break
                    if len(rows) >= max_candidates:
                        break
                if len(rows) >= max_candidates:
                    break
    except Exception as exc:
        logger.warning("Road-based candidate generation failed, falling back to a grid: %s", exc)
        rows = []

    if not rows:
        rng = np.random.default_rng(40)
        existing_union = None
        try:
            existing_union = unary_union(existing.to_crs(UTM).geometry) if existing is not None and not existing.empty else None
        except Exception as exc:
            logger.warning("Could not build existing-light union for fallback candidates: %s", exc)
        for _ in range(max_candidates * 5):
            p = Point(CHINHOYI_LON + rng.uniform(-.04, .04), CHINHOYI_LAT + rng.uniform(-.04, .04))
            try:
                pu = gpd.GeoSeries([p], crs=WGS84).to_crs(UTM).iloc[0]
                if existing_union is not None and existing_union.distance(pu) < 50:
                    continue
                dist_val = np.nan if existing_union is None else existing_union.distance(pu)
            except Exception:
                dist_val = np.nan
            rows.append({"candidate_id": f"CSL-{counter:04d}", "road_name": "DEMO — no road dataset", "road_type": "Demo", "distance_existing_light_m": dist_val, "geometry": p})
            counter += 1
            if len(rows) >= max_candidates:
                break
    if not rows:
        return gpd.GeoDataFrame({"candidate_id": [], "road_name": [], "road_type": [], "distance_existing_light_m": []}, geometry=[], crs=WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


DEFAULT_WEIGHTS = {
    "population": .14, "lighting_gap": .16, "pedestrian": .10, "bus": .09,
    "facilities": .08, "crime": .12, "road": .07, "solar": .07,
    "slope": .05, "environment": .06, "access": .06,
}


def _load_scenario_overrides() -> Dict[str, Dict[str, float]]:
    """Optional municipal customisation: if data/scenario_weights.json exists, its scenarios are
    merged on top of the built-in defaults. Never crashes if the file is malformed or absent."""
    path = DATA_DIR / "scenario_weights.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in raw.items() if isinstance(v, dict)}
    except Exception as exc:
        logger.warning("Ignoring invalid scenario_weights.json: %s", exc)
        return {}


# Preset weight sets. These are fixed expert-set weights, NOT derived by AHP; they are kept as
# alternatives for what-if comparison and for the weight-sensitivity analysis.
SCENARIOS = {
    "Balanced": DEFAULT_WEIGHTS,
    "Safety First": {**DEFAULT_WEIGHTS, "crime": .18, "population": .17, "lighting_gap": .18, "pedestrian": .13},
    "Solar / Energy First": {**DEFAULT_WEIGHTS, "solar": .20, "slope": .09, "environment": .08, "access": .08, "crime": .05},
    "Budget First": {**DEFAULT_WEIGHTS, "access": .13, "road": .12, "lighting_gap": .18, "population": .17},
    "Equity First": {**DEFAULT_WEIGHTS, "population": .20, "lighting_gap": .20, "facilities": .11, "pedestrian": .12},
}
for _name, _weights in _load_scenario_overrides().items():
    SCENARIOS[_name] = {**DEFAULT_WEIGHTS, **_weights}

FEATURE_LABELS = {
    "population": "Population pressure", "lighting_gap": "Lighting gap", "pedestrian": "Pedestrian activity",
    "bus": "Bus-stop proximity", "facilities": "Public facilities", "crime": "Safety / crime risk",
    "road": "Road hierarchy", "solar": "Solar potential", "slope": "Terrain suitability",
    "environment": "Environmental suitability", "access": "Maintenance accessibility",
}

# =====================================================================
# OBJECTIVE 2 — AHP CRITERIA WEIGHTING + MCDA SELECTION OF NEW SOLAR LIGHTS
# =====================================================================
# Method (weighted linear combination, WLC):
#   1. Every candidate site (generated along the road network, away from the
#      existing lights) is described by the 11 criteria in FEATURE_LABELS.
#   2. Each criterion is standardised to 0-1 by min-max scaling across the
#      candidates. All criteria are engineered so that HIGHER = MORE SUITABLE.
#   3. Criterion weights come from the Analytic Hierarchy Process (AHP): a
#      Saaty-scale pairwise-comparison matrix, whose principal eigenvector gives
#      the weights. Judgement consistency is checked with
#      CI = (lambda_max - n) / (n - 1) and CR = CI / RI; CR < 0.10 is acceptable.
#   4. Suitability index = sum(weight_k x standardised_k) x 100.
#   5. Sites are ranked. Going down the ranking, a site becomes a RECOMMENDED
#      NEW SOLAR LIGHT unless it is inside a flood/wetland constraint area,
#      closer than the minimum spacing to an already-chosen site, or the budget
#      is exhausted.
#   6. Sensitivity analysis shows how stable the chosen sites are when the
#      weights change.

DEFAULT_MIN_SPACING_M = int(float(os.environ.get("MIN_NEW_LIGHT_SPACING_M", "60")))
AHP_DATA_FILE = DATA_DIR / "ahp_pairwise.csv"
AHP_CR_THRESHOLD = 0.10
AHP_METHOD = "AHP (pairwise judgements)"
WEIGHTING_METHODS = [AHP_METHOD] + list(SCENARIOS)

# Priority classes are RELATIVE: the observed range of the suitability index is
# split into five equal-interval bands (top 20% of the range = VERY HIGH, and so
# on). "High-priority sites" = VERY HIGH + HIGH.
PRIORITY_CLASS_BREAKS = {"VERY HIGH": 0.80, "HIGH": 0.60, "MODERATE": 0.40, "LOW": 0.20}
HIGH_PRIORITY_CLASSES = ("VERY HIGH", "HIGH")

CRITERIA: List[str] = list(FEATURE_LABELS)
GROUP_DEMAND = "Demand & safety"
GROUP_TECHNICAL = "Technical & environmental"
CRITERIA_GROUPS: Dict[str, str] = {
    "population": GROUP_DEMAND, "lighting_gap": GROUP_DEMAND, "pedestrian": GROUP_DEMAND, "bus": GROUP_DEMAND,
    "facilities": GROUP_DEMAND, "crime": GROUP_DEMAND, "road": GROUP_DEMAND,
    "solar": GROUP_TECHNICAL, "slope": GROUP_TECHNICAL, "environment": GROUP_TECHNICAL, "access": GROUP_TECHNICAL,
}
CRITERIA_MEASURE: Dict[str, str] = {
    "population": "People / households within 250 m of the site (population layer)",
    "lighting_gap": "Distance to the nearest existing streetlight — a larger gap means greater need",
    "pedestrian": "Bus stops within 150 m plus 0.75 × public facilities within 150 m",
    "bus": "Proximity to the nearest bus stop (1 at the stop, 0 at 1,500 m or more)",
    "facilities": "Distance-decayed (500 m) weighted score of schools, clinics, police, markets and other facilities",
    "crime": "Distance-decayed (400 m) crime / incident intensity from the crime layer",
    "road": "Road hierarchy (primary 1.0, secondary / collector 0.75, local 0.5, other 0.4)",
    "solar": "Solar irradiance sampled from a raster (neutral 0.5 when no raster is available)",
    "slope": "Terrain suitability — NOT calculated in this version, fixed at a neutral 0.5",
    "environment": "Outside flood-risk and wetland areas (1.0 clear, 0.45 one constraint, 0.05 both)",
    "access": "Maintenance accessibility: proximity to the road network (75%) and environmental suitability (25%)",
}
CRITERION_SOURCES: Dict[str, List[str]] = {
    "population": ["population"], "lighting_gap": ["streetlights"], "pedestrian": ["bus_stops", "facilities"],
    "bus": ["bus_stops"], "facilities": ["facilities"], "crime": ["crime"], "road": ["roads"],
    "solar": ["solar"], "slope": ["dem"], "environment": ["flood_risk", "wetlands"], "access": ["roads"],
}

# Saaty's 1-9 scale and its reciprocals, and Saaty's random consistency index.
SAATY_VALUES: List[float] = [1.0 / v for v in range(9, 1, -1)] + [1.0] + [float(v) for v in range(2, 10)]
SAATY_WORDS = {
    1: "equal importance", 2: "between equal and moderate", 3: "moderate importance",
    4: "between moderate and strong", 5: "strong importance", 6: "between strong and very strong",
    7: "very strong importance", 8: "between very strong and extreme", 9: "extreme importance",
}
SAATY_RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51,
                      12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}


def nearest_saaty(ratio: float) -> float:
    """Snap a positive ratio to the nearest value on Saaty's scale (log distance)."""
    r = min(max(float(ratio), 1.0 / 9.0), 9.0)
    return min(SAATY_VALUES, key=lambda v: abs(math.log(v) - math.log(r)))


def saaty_options() -> Dict[str, float]:
    """Dropdown options: label -> judgement value (how important A is relative to B)."""
    options: Dict[str, float] = {}
    for v in SAATY_VALUES:
        if v > 1.0:
            n = int(round(v))
            options[f"A is {n}× as important as B — {SAATY_WORDS[n]}"] = v
        elif v == 1.0:
            options["A and B are equally important (1)"] = v
        else:
            n = int(round(1.0 / v))
            options[f"B is {n}× as important as A — {SAATY_WORDS[n]}"] = v
    return dict(sorted(options.items(), key=lambda kv: -kv[1]))


def format_saaty(value: float) -> str:
    v = float(value)
    if v >= 1.0:
        return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
    inv = 1.0 / v
    return f"1/{inv:.0f}" if abs(inv - round(inv)) < 1e-6 else f"{v:.2f}"


def default_ahp_matrix() -> np.ndarray:
    """Starting judgements: the weights this dashboard used before (DEFAULT_WEIGHTS)
    expressed as pairwise ratios and rounded to Saaty's scale. They are ILLUSTRATIVE
    placeholders — replace them with your own expert / stakeholder judgements."""
    n = len(CRITERIA)
    w = np.array([DEFAULT_WEIGHTS[k] for k in CRITERIA], dtype=float)
    m = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = nearest_saaty(w[i] / w[j])
            m[j, i] = 1.0 / m[i, j]
    return m


def ahp_analysis(matrix) -> Dict[str, Any]:
    """Principal-eigenvector weights, lambda_max, consistency index and ratio."""
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = max((lam - n) / (n - 1), 0.0) if n > 1 else 0.0
    ri = SAATY_RANDOM_INDEX.get(n, 1.59)
    cr = ci / ri if ri > 0 else 0.0
    gm = np.prod(a, axis=1) ** (1.0 / n)
    return {
        "n": n, "weights": w, "weights_by_key": {k_: float(v) for k_, v in zip(CRITERIA, w)},
        "geometric_mean_weights": gm / gm.sum(), "lambda_max": lam, "ci": float(ci), "ri": float(ri),
        "cr": float(cr), "consistent": bool(cr < AHP_CR_THRESHOLD),
    }


def ahp_inconsistency_hints(matrix, top: int = 3) -> List[Dict[str, Any]]:
    """The judgements that depart most from the ratio implied by the weights
    (a heuristic for deciding which comparisons to revisit first)."""
    a = np.asarray(matrix, dtype=float)
    w = ahp_analysis(a)["weights"]
    rows = []
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[0]):
            implied = w[i] / w[j]
            rows.append((abs(math.log(a[i, j]) - math.log(implied)), i, j, implied))
    rows.sort(reverse=True)
    return [{"a": FEATURE_LABELS[CRITERIA[i]], "b": FEATURE_LABELS[CRITERIA[j]], "current": float(a[i, j]), "suggested": nearest_saaty(imp)}
            for _, i, j, imp in rows[:top]]


def matrix_to_csv(matrix) -> str:
    labels = [FEATURE_LABELS[k] for k in CRITERIA]
    # Decimals rather than "1/3": spreadsheet programs turn fractions into dates.
    return pd.DataFrame(np.round(np.asarray(matrix, dtype=float), 6), index=labels, columns=labels).to_csv(index_label="criterion")


def _norm_label(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _parse_saaty_value(raw) -> float:
    s = str(raw).strip().replace(",", ".")
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def parse_pairwise_csv(raw: bytes) -> Tuple[Optional[np.ndarray], str]:
    """Read a pairwise matrix from CSV (first column = criterion names, header = criterion
    names; names may be the dashboard labels or internal keys; values may be decimals or
    fractions such as 1/3). Only the upper triangle is required; the lower triangle is
    rebuilt as reciprocals. Returns (matrix, message)."""
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, index_col=0, skipinitialspace=True)
    except Exception as exc:
        return None, f"The file could not be read as CSV ({exc})."
    lookup: Dict[str, str] = {}
    for key, label in FEATURE_LABELS.items():
        lookup[_norm_label(key)] = key
        lookup[_norm_label(label)] = key

    def to_keys(names):
        keys = []
        for name in names:
            key = lookup.get(_norm_label(name))
            if key is None:
                return None, str(name)
            keys.append(key)
        return keys, None

    row_keys, bad = to_keys(df.index)
    if row_keys is None:
        return None, f"Row label “{bad}” is not one of the {len(CRITERIA)} criteria."
    col_keys, bad = to_keys(df.columns)
    if col_keys is None:
        return None, f"Column heading “{bad}” is not one of the {len(CRITERIA)} criteria."
    if sorted(row_keys) != sorted(CRITERIA) or sorted(col_keys) != sorted(CRITERIA):
        return None, f"The matrix must contain each of the {len(CRITERIA)} criteria exactly once as a row and once as a column."
    grid = pd.DataFrame(df.values, index=row_keys, columns=col_keys)
    n = len(CRITERIA)
    m = np.ones((n, n))
    try:
        for i, a in enumerate(CRITERIA):
            for j in range(i + 1, n):
                b = CRITERIA[j]
                upper, lower = grid.loc[a, b], grid.loc[b, a]
                if pd.notna(upper) and str(upper).strip():
                    v = _parse_saaty_value(upper)
                elif pd.notna(lower) and str(lower).strip():
                    v = 1.0 / _parse_saaty_value(lower)
                else:
                    return None, f"No judgement was given for “{FEATURE_LABELS[a]}” versus “{FEATURE_LABELS[b]}”."
                if not (1.0 / 9.0 - 1e-6 <= v <= 9.0 + 1e-6):
                    return None, f"The judgement for “{FEATURE_LABELS[a]}” versus “{FEATURE_LABELS[b]}” ({v:.3f}) is outside Saaty's 1/9 to 9 range."
                for s in SAATY_VALUES:           # tolerate 0.333333 for 1/3 after a spreadsheet round trip
                    if abs(v - s) / s < 1e-4:
                        v = s
                        break
                m[i, j], m[j, i] = v, 1.0 / v
    except (ValueError, ZeroDivisionError) as exc:
        return None, f"A judgement value could not be read ({exc})."
    return m, "Matrix read successfully."


def load_initial_ahp_matrix() -> Tuple[np.ndarray, str]:
    """data/ahp_pairwise.csv (if present and valid) is the starting matrix for every
    session; otherwise the illustrative defaults are used."""
    if AHP_DATA_FILE.exists():
        try:
            matrix, message = parse_pairwise_csv(AHP_DATA_FILE.read_bytes())
            if matrix is not None:
                return matrix, f"loaded from data/{AHP_DATA_FILE.name}"
            logger.warning("Ignoring %s: %s", AHP_DATA_FILE.name, message)
        except Exception as exc:
            logger.warning("Could not read %s: %s", AHP_DATA_FILE.name, exc)
    return default_ahp_matrix(), "default illustrative judgements (rounded from the earlier fixed weights) — replace with your own expert judgements"


AHP_INITIAL_MATRIX, AHP_INITIAL_SOURCE = load_initial_ahp_matrix()

SCORE_OUTPUT_COLUMNS = [
    "candidate_id", "road_name", "road_type", "distance_existing_light_m",
    "bus_dist_m", "road_dist_m", "existing_light_dist_m", "bus", "lighting_gap",
    "population", "pedestrian", "facilities", "crime", "road", "solar", "slope",
    "environment_constraint", "environment", "access", "utm_easting", "utm_northing",
    "longitude", "latitude", "mcda_score", "safety_score", "technical_score",
    "overall_priority", "main_driver", "rank", "priority_class", "estimated_cost",
    "selection_status", "funded_phase_1", "new_light_id", "implementation_phase",
]


def build_candidate_features(cands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = cands.copy()
    out["bus_dist_m"] = nearest_distance_series(out, DATA.bus_stops)
    out["road_dist_m"] = nearest_distance_series(out, DATA.roads)
    out["existing_light_dist_m"] = nearest_distance_series(out, DATA.streetlights)
    out["bus"] = inverse_distance_score(out["bus_dist_m"], 1500)
    out["lighting_gap"] = normalize_series(out["existing_light_dist_m"], True).fillna(0)
    out["population"] = population_value_within(out, DATA.population, 250)
    out["pedestrian"] = point_counts_within(out, DATA.bus_stops, 150) + 0.75 * point_counts_within(out, DATA.facilities, 150)
    out["facilities"] = facilities_score(out, DATA.facilities)
    out["crime"] = crime_score(out, DATA.crime)
    rt = out.get("road_type", pd.Series("Unknown", index=out.index)).astype(str).str.lower()
    out["road"] = np.select([rt.str.contains("highway|primary|trunk|arterial"), rt.str.contains("secondary|collector"), rt.str.contains("tertiary|local")], [1.0, .75, .5], .4)
    solar = sample_raster_at_points(out, DATA.solar_path)
    out["solar"] = normalize_series(solar) if solar.notna().any() else pd.Series(.5, index=out.index)
    out["slope"] = pd.Series(.5, index=out.index)
    constraint, env = environment_score(out)
    out["environment_constraint"] = constraint
    out["environment"] = env
    out["access"] = (0.75 * inverse_distance_score(out["road_dist_m"].fillna(500), 250) + 0.25 * out["environment"]).clip(0, 1)
    u = out.to_crs(UTM)
    out["utm_easting"] = u.geometry.x.round(2)
    out["utm_northing"] = u.geometry.y.round(2)
    out["longitude"] = out.geometry.x.round(6)
    out["latitude"] = out.geometry.y.round(6)
    return out


def _safe_number(value, default=0.0):
    """Convert scalar-like values to float without raising on pandas NA or non-numeric objects."""
    try:
        if value is None:
            return float(default)
        f = float(value)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _empty_scored_frame() -> gpd.GeoDataFrame:
    cols = {c: pd.Series(dtype="object") for c in SCORE_OUTPUT_COLUMNS}
    return gpd.GeoDataFrame(cols, geometry=gpd.GeoSeries([], dtype="geometry"), crs=WGS84)


def _group_score(norm: pd.DataFrame, w: np.ndarray, keys: List[str], members: List[str]) -> pd.Series:
    """Weighted score (0-100) over one group of criteria, re-based on that group's own weight total."""
    idx = [i for i, k in enumerate(keys) if k in members]
    total = float(w[idx].sum()) if idx else 0.0
    if total <= 0:
        return pd.Series(50.0, index=norm.index)
    return pd.Series(np.round(norm.iloc[:, idx].to_numpy() @ w[idx] / total * 100, 2), index=norm.index)


def select_new_lights(out: pd.DataFrame, budget: float, min_spacing_m: float, hard_constraints: bool) -> Tuple[np.ndarray, List[str]]:
    """Greedy selection down the ranking (``out`` must already be sorted best-first).
    A site is skipped when it is inside a flood/wetland constraint area (optional), closer
    than ``min_spacing_m`` to an already-selected site, or when the budget is used up."""
    n = len(out)
    xs = pd.to_numeric(out["utm_easting"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(out["utm_northing"], errors="coerce").to_numpy(dtype=float)
    cost = pd.to_numeric(out["estimated_cost"], errors="coerce").fillna(0).to_numpy(dtype=float)
    cons = pd.to_numeric(out["environment_constraint"], errors="coerce").fillna(0).to_numpy(dtype=float)
    chosen = np.zeros(n, dtype=bool)
    status: List[str] = [""] * n
    picked: List[int] = []
    spent = 0.0
    for i in range(n):
        if hard_constraints and cons[i] >= 1:
            status[i] = "Excluded — inside a flood / wetland constraint area"
            continue
        if picked and np.isfinite(xs[i]) and np.isfinite(ys[i]):
            if float(np.hypot(xs[picked] - xs[i], ys[picked] - ys[i]).min()) < min_spacing_m:
                status[i] = f"Excluded — within {min_spacing_m:.0f} m of a higher-ranked site"
                continue
        if cost[i] <= 0 or spent + cost[i] > budget:
            status[i] = "Not funded — budget exhausted"
            continue
        chosen[i] = True
        picked.append(i)
        spent += cost[i]
        status[i] = "Recommended — within budget"
    return chosen, status


def score_features(out: pd.DataFrame, weights: Dict[str, float], budget: float, unit_cost: float, contingency: float,
                   min_spacing_m: float = DEFAULT_MIN_SPACING_M, hard_constraints: bool = True) -> pd.DataFrame:
    """Steps 2-5 of the AHP/MCDA method on a table that already holds the criteria columns."""
    keys = [k for k in CRITERIA if k in out.columns]
    raw = np.array([max(float(weights.get(k, 0.0)), 0.0) for k in keys], dtype=float)
    w = raw / raw.sum() if raw.sum() > 0 else np.full(len(keys), 1.0 / max(len(keys), 1))
    norm = pd.DataFrame({k: normalize_series(out[k]).to_numpy() for k in keys}, index=out.index)
    contrib = norm * w
    out = out.copy()
    out["mcda_score"] = np.round(contrib.sum(axis=1).to_numpy() * 100, 2)
    out["overall_priority"] = out["mcda_score"]
    out["safety_score"] = _group_score(norm, w, keys, [k for k, g in CRITERIA_GROUPS.items() if g == GROUP_DEMAND])
    out["technical_score"] = _group_score(norm, w, keys, [k for k, g in CRITERIA_GROUPS.items() if g == GROUP_TECHNICAL])
    out["main_driver"] = contrib.idxmax(axis=1).map(FEATURE_LABELS)
    out = out.sort_values("overall_priority", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    scores = pd.to_numeric(out["overall_priority"], errors="coerce")
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        out["priority_class"] = "MODERATE"
    else:
        rel = (scores - lo) / (hi - lo)
        out["priority_class"] = np.select(
            [rel >= PRIORITY_CLASS_BREAKS["VERY HIGH"], rel >= PRIORITY_CLASS_BREAKS["HIGH"],
             rel >= PRIORITY_CLASS_BREAKS["MODERATE"], rel >= PRIORITY_CLASS_BREAKS["LOW"]],
            ["VERY HIGH", "HIGH", "MODERATE", "LOW"], default="VERY LOW")

    safe_unit = max(_safe_number(unit_cost, 0.0), 0.0)
    safe_cont = max(_safe_number(contingency, 0.0), 0.0)
    safe_budget = max(_safe_number(budget, 0.0), 0.0)
    out["estimated_cost"] = safe_unit * (1 + safe_cont / 100.0)
    chosen, status = select_new_lights(out, safe_budget, float(min_spacing_m), bool(hard_constraints))
    seq = np.cumsum(chosen)
    out["selection_status"] = status
    out["funded_phase_1"] = chosen
    out["new_light_id"] = [f"NSL-{int(s):03d}" if c else "" for s, c in zip(seq, chosen)]
    out["implementation_phase"] = np.select(
        [chosen & (seq <= 20), chosen, np.array([s.startswith("Not funded") for s in status])],
        ["Phase 1 — Immediate", "Phase 2 — Budget funded", "Phase 3 — Future funding"], default="Not recommended")
    return out


def score_candidates(cands: gpd.GeoDataFrame, weights: Dict[str, float], budget: float, unit_cost: float, contingency: float,
                     min_spacing_m: float = DEFAULT_MIN_SPACING_M, hard_constraints: bool = True) -> gpd.GeoDataFrame:
    if cands is None or cands.empty:
        return _empty_scored_frame()
    return score_features(build_candidate_features(cands), weights, budget, unit_cost, contingency, min_spacing_m, hard_constraints)


RECOMMENDED_COLUMNS = [
    "new_light_id", "rank", "candidate_id", "road_name", "road_type", "latitude", "longitude", "utm_easting", "utm_northing",
    "overall_priority", "priority_class", "safety_score", "technical_score", "main_driver", "existing_light_dist_m",
    "population", "estimated_cost", "implementation_phase",
]


def recommended_sites(scored: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The recommended NEW solar streetlights as a plain table (no geometry), best first."""
    if scored is None or scored.empty or "funded_phase_1" not in scored.columns or not bool(scored["funded_phase_1"].any()):
        return pd.DataFrame(columns=RECOMMENDED_COLUMNS)
    rec = scored[scored["funded_phase_1"].astype(bool)].sort_values("rank")
    cols = [c for c in RECOMMENDED_COLUMNS if c in rec.columns]
    return pd.DataFrame(rec[cols]).round({"overall_priority": 1, "safety_score": 1, "technical_score": 1, "existing_light_dist_m": 0, "population": 0, "estimated_cost": 0}).reset_index(drop=True)


def criteria_data_status(scored: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Which criteria genuinely discriminate between sites. A criterion that is the same everywhere
    (no data, or a placeholder) carries an AHP weight but cannot influence the ranking."""
    out: Dict[str, str] = {}
    for k in CRITERIA:
        if scored is None or scored.empty or k not in scored.columns:
            out[k] = "No candidates"
        elif pd.to_numeric(scored[k], errors="coerce").nunique(dropna=True) <= 1:
            out[k] = "Neutral — no data / no variation"
        elif any(src in DEMO_LAYERS for src in CRITERION_SOURCES.get(k, [])):
            out[k] = "Demo data"
        else:
            out[k] = "Data-driven"
    return out


def priority_diagnostics(scored: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Explain the prioritisation outcome from live data, and separate a genuine result from a
    data/processing problem (no candidates at all)."""
    n = 0 if scored is None else int(len(scored))
    diag: Dict[str, Any] = {"n_candidates": n, "n_high": 0, "top_score": None, "low_score": None, "verdict": "", "findings": [], "constant_criteria": []}
    if n == 0:
        if DATA.roads is None or DATA.roads.empty:
            reason = "no road layer is loaded, so no candidate positions can be generated along roads"
        else:
            reason = f"every position along the roads is within about {CANDIDATE_SPACING_M * 0.55:.0f} m of an existing streetlight, or road segments are shorter than 30 m"
        diag["verdict"] = "DATA / PROCESSING PROBLEM: no candidate sites were produced, so nothing can be ranked. Likely cause: " + reason + "."
        return diag
    scores = pd.to_numeric(scored["overall_priority"], errors="coerce")
    diag["n_high"] = int(scored["priority_class"].isin(HIGH_PRIORITY_CLASSES).sum())
    diag["top_score"], diag["low_score"] = float(scores.max()), float(scores.min())
    status = criteria_data_status(scored)
    diag["constant_criteria"] = [FEATURE_LABELS[k] for k, v in status.items() if v.startswith("Neutral")]
    findings = []
    if DATA.crime is None:
        findings.append("No crime / safety layer was found, so the crime criterion has no variation and its AHP weight has no effect on the ranking.")
    if DATA.solar_path is None or not RASTERIO_AVAILABLE:
        findings.append("Solar potential is not calculated (" + ("no solar raster found" if DATA.solar_path is None else "rasterio is not installed") + "); every site gets the same neutral value.")
    findings.append("Terrain suitability is not calculated in this version (fixed neutral value), so its AHP weight has no effect on the ranking.")
    if DATA.flood_risk is None and DATA.wetlands is None:
        findings.append("No flood or wetland layers were found, so every site is treated as unconstrained.")
    demo_inputs = [name for name in ("roads", "population", "bus_stops", "facilities", "streetlights") if name in DEMO_LAYERS]
    if demo_inputs:
        findings.append("These inputs are demonstration data, not real: " + ", ".join(demo_inputs) + ".")
    diag["findings"] = findings
    diag["verdict"] = (f"{diag['n_high']} of {n} candidate sites are in the top two suitability classes (upper 40% of the observed score range "
                       f"{diag['low_score']:.1f}–{diag['top_score']:.1f} / 100). Classes are relative to this candidate set, not absolute thresholds.")
    return diag


def _norm_matrix(scored: pd.DataFrame, weights: Dict[str, float]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    keys = [k for k in CRITERIA if k in scored.columns]
    x = np.column_stack([normalize_series(scored[k]).to_numpy() for k in keys]) if keys else np.empty((len(scored), 0))
    raw = np.array([max(float(weights.get(k, 0.0)), 0.0) for k in keys], dtype=float)
    w = raw / raw.sum() if raw.sum() > 0 else np.full(len(keys), 1.0 / max(len(keys), 1))
    return keys, x, w


def _top_set(score: np.ndarray, n: int) -> Set[int]:
    return set(np.argsort(-score, kind="stable")[:n].tolist())


def oat_sensitivity(scored: Optional[pd.DataFrame], weights: Dict[str, float], top_n: int, delta: float = 0.25) -> pd.DataFrame:
    """One-at-a-time weight sensitivity: scale each criterion's weight down and up by ``delta``
    (re-normalising the rest) and report how many of the current top-N sites stay in the top-N."""
    if scored is None or scored.empty:
        return pd.DataFrame()
    keys, x, w = _norm_matrix(scored, weights)
    top_n = max(1, min(int(top_n), len(scored)))
    base = _top_set(x @ w, top_n)
    rows = []
    for j, k in enumerate(keys):
        kept = {}
        for tag, factor in (("down", 1 - delta), ("up", 1 + delta)):
            w2 = w.copy()
            w2[j] *= factor
            w2 = w2 / w2.sum()
            kept[tag] = 100.0 * len(base & _top_set(x @ w2, top_n)) / top_n
        rows.append({"Criterion": FEATURE_LABELS[k], "Weight (%)": round(100 * float(w[j]), 1),
                     f"Top-{top_n} kept, weight −{delta:.0%} (%)": round(kept["down"], 1),
                     f"Top-{top_n} kept, weight +{delta:.0%} (%)": round(kept["up"], 1),
                     "Least stable (%)": round(min(kept.values()), 1)})
    return pd.DataFrame(rows).sort_values("Least stable (%)", kind="mergesort").reset_index(drop=True)


def preset_comparison(scored: Optional[pd.DataFrame], weights: Dict[str, float], alternatives: Dict[str, Dict[str, float]], top_n: int) -> pd.DataFrame:
    """Compare the active weight set with alternative weight sets: share of the top-N sites kept and
    the Spearman rank correlation of the full rankings."""
    if scored is None or scored.empty or not alternatives:
        return pd.DataFrame()
    keys, x, w = _norm_matrix(scored, weights)
    top_n = max(1, min(int(top_n), len(scored)))
    base_score = x @ w
    base_top = _top_set(base_score, top_n)
    base_rank = pd.Series(base_score).rank(ascending=False).to_numpy()
    rows = []
    for name, alt in alternatives.items():
        v = np.array([max(float(alt.get(k, 0.0)), 0.0) for k in keys], dtype=float)
        v = v / v.sum() if v.sum() > 0 else np.full(len(keys), 1.0 / max(len(keys), 1))
        s = x @ v
        rank = pd.Series(s).rank(ascending=False).to_numpy()
        rho = float(np.corrcoef(base_rank, rank)[0, 1]) if len(rank) > 2 and base_rank.std() > 0 and rank.std() > 0 else float("nan")
        rows.append({"Alternative weight set": name, f"Top-{top_n} kept (%)": round(100.0 * len(base_top & _top_set(s, top_n)) / top_n, 1),
                     "Rank correlation (Spearman ρ)": round(rho, 3) if np.isfinite(rho) else None})
    return pd.DataFrame(rows)

# =====================================================================
# DATABASE / RBAC / SESSIONS
# =====================================================================

ROLES = ("ADMIN", "MUNICIPAL_STAFF", "COMMUNITY")
ROLE_META = {
    "ADMIN": {"label": "System Administrator", "description": "Governance, security, data, permissions, reporting and system configuration."},
    "MUNICIPAL_STAFF": {"label": "Municipality Staff", "description": "Lighting assets, maintenance, reports, AHP/MCDA site selection and budget planning."},
    "COMMUNITY": {"label": "Community / Public User", "description": "Streetlight map and status, problem reporting and report tracking."},
}
DEPARTMENTS = {
    "PUBLIC_LIGHTING": "Public Lighting & Infrastructure",
    "ENGINEERING": "Engineering & Works",
    "GIS_PLANNING": "GIS & Urban Planning",
    "COMMUNITY_SERVICES": "Community Services",
    "ICT_ADMIN": "ICT & Administration",
}
REPORT_CATEGORIES = ["Light not working", "Damaged pole", "Broken solar panel", "Battery problem", "Controller problem", "Light too dim", "Light flashing", "Pole obstruction", "Other"]
REPORT_STATUSES = ["Submitted", "Received", "Under Review", "Verified", "Assigned", "In Progress", "Resolved", "Closed", "Returned for Clarification"]

PERMISSIONS = {
    "admin.user_management": "User Management", "admin.permission_management": "Role & Permission Management", "admin.data_management": "Data Management", "admin.gis_configuration": "GIS Configuration", "admin.security_audit": "Security & Audit", "admin.system_reports": "System Reports", "admin.maintenance_oversight": "Maintenance Oversight", "admin.settings": "System Settings", "admin.system_notices": "System Notices", "admin.chatbot": "Dashboard Assistant", "admin.explorer": "Streetlight Map",
    "staff.dashboard": "Staff Dashboard", "staff.explorer": "Streetlight Map", "staff.assets": "Streetlight Assets", "staff.gaps": "Lighting Gaps", "staff.prioritisation": "Site Prioritisation", "staff.maintenance": "Maintenance", "staff.work_orders": "Work Orders", "staff.reports_view": "Community Reports", "staff.reports_update": "Update Reports", "staff.ahp": "AHP Criteria Weights", "staff.mcda": "MCDA Site Selection", "staff.what_if": "What-If Planning", "staff.budget": "Budget & Phasing", "staff.analytics": "Analytics", "staff.export": "Export", "staff.chatbot": "Dashboard Assistant",
    "community.dashboard": "Community Dashboard", "community.explorer": "Streetlight Map", "community.submit_report": "Submit Report", "community.my_reports": "My Reports", "community.safety": "Safety Information", "community.help": "Help & Contact", "community.profile": "My Profile", "community.chatbot": "Ask the Assistant", "common.notifications": "Notifications",
}
DEFAULT_ROLE_PERMISSIONS = {
    "ADMIN": set(PERMISSIONS),
    "MUNICIPAL_STAFF": {k for k in PERMISSIONS if k.startswith("staff.")} | {"common.notifications"},
    "COMMUNITY": {k for k in PERMISSIONS if k.startswith("community.")} | {"common.notifications"},
}
# The Dashboard Assistant is available from every role's own navigation, scoped to what each role
# may see — see DashboardAssistant._facts() for the per-role fact boundary.
NAV = {
    "ADMIN": [("home", "Overview"), ("explorer", "Streetlight Map"), ("users", "User Management"), ("permissions", "Permissions"), ("data", "Data Management"), ("gis_config", "GIS Configuration"), ("audit", "Audit & Security"), ("reports", "System Reports"), ("maintenance_ov", "Maintenance Oversight"), ("notices", "System Notices"), ("settings", "Settings"), ("chatbot", "Dashboard Assistant")],
    "MUNICIPAL_STAFF": [("home", "Dashboard"), ("explorer", "Streetlight Map"), ("assets", "Streetlight Assets"), ("gaps", "Lighting Gaps"), ("prioritisation", "Site Prioritisation"), ("maintenance", "Maintenance"), ("reports", "Community Reports"), ("ahp", "AHP Criteria Weights"), ("mcda", "MCDA Site Selection"), ("what_if", "What-If Planning"), ("budget", "Budget & Phasing"), ("analytics", "Analytics"), ("export", "Export"), ("chatbot", "Dashboard Assistant")],
    "COMMUNITY": [("home", "Home"), ("explorer", "Streetlight Map"), ("submit_report", "Report a Problem"), ("my_reports", "My Reports"), ("safety", "Safety Information"), ("help", "Help & Contact"), ("chatbot", "Ask the Assistant"), ("profile", "My Profile")],
}

_PERMISSION_CACHE: Dict[str, Set[str]] = {}


def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return digest, salt


def verify_password(password: str, digest: str, salt: str) -> bool:
    current, _ = hash_password(password, salt)
    return secrets.compare_digest(current, digest)


def _seed_password(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


SEED_ACCOUNTS = [
    ("admin", "admin@chinhoyi.gov.zw", "System Administrator", "ADMIN", "ICT_ADMIN", _seed_password("ADMIN_DEFAULT_PASSWORD", "Admin@123")),
    ("staff", "staff@chinhoyi.gov.zw", "Municipal Staff", "MUNICIPAL_STAFF", "PUBLIC_LIGHTING", _seed_password("STAFF_DEFAULT_PASSWORD", "Staff@123")),
    ("resident", "resident@example.com", "Community Resident", "COMMUNITY", None, _seed_password("COMMUNITY_DEFAULT_PASSWORD", "Resident@123")),
]
USING_DEFAULT_CREDENTIALS = not any(
    os.environ.get(n) for n in ("ADMIN_DEFAULT_PASSWORD", "STAFF_DEFAULT_PASSWORD", "COMMUNITY_DEFAULT_PASSWORD")
)


def init_db():
    with connect_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            full_name TEXT,
            role TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            department TEXT,
            active INTEGER DEFAULT 1,
            created_at TEXT,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS role_permissions (role TEXT NOT NULL, feature TEXT NOT NULL, PRIMARY KEY (role, feature));
        CREATE TABLE IF NOT EXISTS community_reports (
            report_id TEXT PRIMARY KEY, user_id INTEGER, username TEXT, reported_at TEXT, category TEXT,
            latitude REAL, longitude REAL, location_text TEXT, description TEXT, contact TEXT,
            status TEXT DEFAULT 'Submitted', priority TEXT DEFAULT 'Medium', assigned_to TEXT,
            assigned_department TEXT, linked_work_order TEXT, municipal_response TEXT,
            verification_note TEXT, resolution_notes TEXT, verified_at TEXT, resolved_at TEXT,
            closed_at TEXT, updated_at TEXT, asset_id TEXT
        );
        CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, message TEXT, kind TEXT DEFAULT 'info', is_read INTEGER DEFAULT 0, created_at TEXT);
        CREATE TABLE IF NOT EXISTS audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, username TEXT, role TEXT, action TEXT, details TEXT);
        CREATE TABLE IF NOT EXISTS work_orders (
            work_order_id TEXT PRIMARY KEY, asset_id TEXT, report_id TEXT, status TEXT DEFAULT 'Open', priority TEXT,
            fault_type TEXT, assigned_to TEXT, assigned_department TEXT, reported_at TEXT, notes TEXT, updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS user_sessions (
            token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, username TEXT, role TEXT,
            created_at TEXT, last_seen TEXT, expires_at TEXT, revoked INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS user_state (user_id INTEGER PRIMARY KEY, state_json TEXT, updated_at TEXT);
        """)
        # Migration: databases created by earlier builds have no asset_id on community_reports.
        report_cols = {r["name"] for r in conn.execute("PRAGMA table_info(community_reports)").fetchall()}
        if "asset_id" not in report_cols:
            conn.execute("ALTER TABLE community_reports ADD COLUMN asset_id TEXT")
        for role, features in DEFAULT_ROLE_PERMISSIONS.items():
            for feature in features:
                conn.execute("INSERT OR IGNORE INTO role_permissions(role, feature) VALUES (?,?)", (role, feature))
        if conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0:
            for username, email, name, role, department, password in SEED_ACCOUNTS:
                digest, salt = hash_password(password)
                conn.execute(
                    "INSERT INTO users(username,email,full_name,role,password_hash,salt,department,active,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (username, email, name, role, digest, salt, department, 1, datetime.now().isoformat(timespec="seconds")),
                )
        # Housekeeping: forget sessions that ended more than a week ago.
        cutoff = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
        conn.execute("DELETE FROM user_sessions WHERE expires_at < ? OR (revoked=1 AND last_seen < ?)", (cutoff, cutoff))
        conn.commit()
    if os.environ.get("RENDER") and USING_DEFAULT_CREDENTIALS:
        logger.warning(
            "Running with default development credentials on what looks like a Render deployment. "
            "Set ADMIN_DEFAULT_PASSWORD / STAFF_DEFAULT_PASSWORD / COMMUNITY_DEFAULT_PASSWORD before real use."
        )


def invalidate_permission_cache():
    _PERMISSION_CACHE.clear()


def get_role_permissions(role: str) -> Set[str]:
    if role not in _PERMISSION_CACHE:
        with connect_db() as conn:
            rows = conn.execute("SELECT feature FROM role_permissions WHERE role=?", (role,)).fetchall()
        _PERMISSION_CACHE[role] = {r["feature"] for r in rows}
    return _PERMISSION_CACHE[role]


def audit(username, role, action, details=""):
    try:
        with connect_db() as conn:
            conn.execute("INSERT INTO audit_logs(ts,username,role,action,details) VALUES (?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), username or "anonymous", role or "-", action, details))
            conn.commit()
    except Exception as exc:
        logger.warning("Could not write audit log entry (%s): %s", action, exc)


def _user_dict(row, last_login: Optional[str] = None) -> dict:
    return dict(id=row["id"], username=row["username"], email=row["email"], full_name=row["full_name"], role=row["role"],
                department=row["department"], last_login=last_login or row["last_login"])


def authenticate(username: str, password: str) -> Optional[dict]:
    with connect_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=? AND active=1", (username.strip(),)).fetchone()
    if row is None or not verify_password(password, row["password_hash"], row["salt"]):
        return None
    login_time = datetime.now().isoformat(timespec="seconds")
    with connect_db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (login_time, row["id"])); conn.commit()
    return _user_dict(row, login_time)


# ---------------------------------------------------------------------
# Persistent sessions — the browser holds a random token in a cookie; only its SHA-256 hash is
# stored. A session slides forward each time it is used and ends at logout, on expiry, or when it
# is revoked (account disabled, password reset, or an administrator ends it).
# ---------------------------------------------------------------------

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _expiry_iso() -> str:
    return (datetime.now() + timedelta(hours=SESSION_TTL_HOURS)).isoformat(timespec="seconds")


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now().isoformat(timespec="seconds")
    with connect_db() as conn:
        conn.execute("INSERT INTO user_sessions(token_hash,user_id,username,role,created_at,last_seen,expires_at,revoked) VALUES (?,?,?,?,?,?,?,0)",
                     (_token_hash(token), user["id"], user["username"], user["role"], now, now, _expiry_iso()))
        conn.commit()
    return token


def get_session_user(token: Optional[str]) -> Optional[dict]:
    """The user a valid, unexpired, unrevoked session token belongs to (and extend the session)."""
    if not token or not isinstance(token, str) or len(token) > 200:
        return None
    try:
        now = datetime.now().isoformat(timespec="seconds")
        with connect_db() as conn:
            row = conn.execute(
                "SELECT u.* FROM user_sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash=? AND s.revoked=0 AND s.expires_at > ? AND u.active=1", (_token_hash(token), now)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE user_sessions SET last_seen=?, expires_at=? WHERE token_hash=?", (now, _expiry_iso(), _token_hash(token)))
            conn.commit()
        return _user_dict(row)
    except Exception as exc:
        logger.warning("Session lookup failed: %s", exc)
        return None


def touch_session(token: Optional[str]) -> None:
    if not token:
        return
    try:
        with connect_db() as conn:
            conn.execute("UPDATE user_sessions SET last_seen=?, expires_at=? WHERE token_hash=? AND revoked=0",
                         (datetime.now().isoformat(timespec="seconds"), _expiry_iso(), _token_hash(token)))
            conn.commit()
    except Exception as exc:
        logger.debug("Could not extend session: %s", exc)


def revoke_session(token: Optional[str]) -> None:
    if not token:
        return
    try:
        with connect_db() as conn:
            conn.execute("UPDATE user_sessions SET revoked=1, last_seen=? WHERE token_hash=?", (datetime.now().isoformat(timespec="seconds"), _token_hash(token)))
            conn.commit()
    except Exception as exc:
        logger.warning("Could not revoke session: %s", exc)


def revoke_user_sessions(user_id: int) -> int:
    with connect_db() as conn:
        cur = conn.execute("UPDATE user_sessions SET revoked=1, last_seen=? WHERE user_id=? AND revoked=0", (datetime.now().isoformat(timespec="seconds"), int(user_id)))
        conn.commit()
        return int(cur.rowcount or 0)


def list_active_sessions() -> pd.DataFrame:
    now = datetime.now().isoformat(timespec="seconds")
    with connect_db() as conn:
        rows = conn.execute("SELECT user_id, username, role, created_at, last_seen, expires_at FROM user_sessions "
                            "WHERE revoked=0 AND expires_at > ? ORDER BY last_seen DESC", (now,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_user_state(user_id: int, state: Dict[str, Any]) -> None:
    with connect_db() as conn:
        conn.execute("INSERT OR REPLACE INTO user_state(user_id,state_json,updated_at) VALUES (?,?,?)",
                     (int(user_id), json.dumps(state, default=str), datetime.now().isoformat(timespec="seconds")))
        conn.commit()


def load_user_state(user_id: int) -> Dict[str, Any]:
    try:
        with connect_db() as conn:
            row = conn.execute("SELECT state_json FROM user_state WHERE user_id=?", (int(user_id),)).fetchone()
        loaded = json.loads(row["state_json"]) if row and row["state_json"] else {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.warning("Could not load saved user state: %s", exc)
        return {}


def has_permission(role: str, feature: str) -> bool:
    if not feature:
        return False
    return feature in get_role_permissions(role)


def can_route(role: str, route: str) -> bool:
    if route == "home":
        return has_permission(role, {"ADMIN": "admin.system_reports", "MUNICIPAL_STAFF": "staff.dashboard", "COMMUNITY": "community.dashboard"}.get(role, ""))
    if route == "reports":
        # "reports" is reused by two nav items — Admin "System Reports" and Staff "Community
        # Reports" — so it is resolved per role rather than through the flat lookup below.
        feature = "admin.system_reports" if role == "ADMIN" else "staff.reports_view"
        return has_permission(role, feature)
    if route == "chatbot":
        feature = {"ADMIN": "admin.chatbot", "MUNICIPAL_STAFF": "staff.chatbot", "COMMUNITY": "community.chatbot"}.get(role, "")
        return has_permission(role, feature)
    if route == "explorer":
        feature = {"ADMIN": "admin.explorer", "MUNICIPAL_STAFF": "staff.explorer", "COMMUNITY": "community.explorer"}.get(role, "")
        return has_permission(role, feature)
    mapping = {
        "users": "admin.user_management", "permissions": "admin.permission_management", "data": "admin.data_management", "gis_config": "admin.gis_configuration", "audit": "admin.security_audit", "maintenance_ov": "admin.maintenance_oversight", "settings": "admin.settings", "notices": "admin.system_notices",
        "assets": "staff.assets", "gaps": "staff.gaps", "prioritisation": "staff.prioritisation", "maintenance": "staff.maintenance", "ahp": "staff.ahp", "mcda": "staff.mcda", "what_if": "staff.what_if", "budget": "staff.budget", "analytics": "staff.analytics", "export": "staff.export",
        "submit_report": "community.submit_report", "my_reports": "community.my_reports", "safety": "community.safety", "help": "community.help", "profile": "community.profile",
    }
    return has_permission(role, mapping.get(route, ""))


def list_users() -> pd.DataFrame:
    with connect_db() as conn:
        rows = conn.execute("SELECT id,username,email,full_name,role,department,active,created_at,last_login FROM users ORDER BY username").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def create_user(username, email, full_name, role, password, department=None) -> Tuple[bool, str]:
    if role not in ROLES or not username.strip() or not password:
        return False, "Username, role and password are required."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    digest, salt = hash_password(password)
    try:
        with connect_db() as conn:
            conn.execute("INSERT INTO users(username,email,full_name,role,password_hash,salt,department,active,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (username.strip(), email.strip(), full_name.strip(), role, digest, salt, department, 1, datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def set_user_active(user_id: int, active: bool):
    with connect_db() as conn:
        conn.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, int(user_id))); conn.commit()
    if not active:
        revoke_user_sessions(int(user_id))


def reset_password(user_id: int, password: str):
    digest, salt = hash_password(password)
    with connect_db() as conn:
        conn.execute("UPDATE users SET password_hash=?,salt=? WHERE id=?", (digest, salt, int(user_id))); conn.commit()
    revoke_user_sessions(int(user_id))


def add_notification(user_id: int, message: str, kind: str = "info"):
    with connect_db() as conn:
        conn.execute("INSERT INTO notifications(user_id,message,kind,is_read,created_at) VALUES(?,?,?,?,?)", (user_id, message, kind, 0, datetime.now().isoformat(timespec="seconds"))); conn.commit()


def get_notifications(user_id: int, unread_only=False, limit=10) -> pd.DataFrame:
    sql = "SELECT * FROM notifications WHERE user_id=?"
    params = [user_id]
    if unread_only:
        sql += " AND is_read=0"
    sql += " ORDER BY id DESC LIMIT ?"; params.append(limit)
    with connect_db() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def mark_notifications_read(user_id: int):
    with connect_db() as conn:
        conn.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (user_id,)); conn.commit()


def save_report(user, category, latitude, longitude, location_text, description, contact, priority="Medium", asset_id: Optional[str] = None) -> str:
    rid = f"RPT-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
    now = datetime.now().isoformat(timespec="seconds")
    with connect_db() as conn:
        conn.execute("INSERT INTO community_reports(report_id,user_id,username,reported_at,category,latitude,longitude,location_text,description,contact,status,priority,updated_at,asset_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (rid, user["id"], user["username"], now, category, float(latitude), float(longitude), location_text, description, contact, "Submitted", priority, now, (asset_id or None)))
        conn.commit()
    return rid


def get_reports(user_id=None, department=None) -> pd.DataFrame:
    sql = "SELECT * FROM community_reports WHERE 1=1"; params = []
    if user_id is not None:
        sql += " AND user_id=?"; params.append(int(user_id))
    if department:
        sql += " AND assigned_department=?"; params.append(department)
    sql += " ORDER BY reported_at DESC"
    with connect_db() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def update_report(report_id: str, changed_by: str, **fields):
    allowed = {"status", "priority", "assigned_to", "assigned_department", "linked_work_order", "municipal_response", "verification_note", "resolution_notes", "verified_at", "resolved_at", "closed_at"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")
    clauses = ", ".join(f"{k}=?" for k in updates)
    with connect_db() as conn:
        conn.execute(f"UPDATE community_reports SET {clauses} WHERE report_id=?", tuple(updates.values()) + (report_id,)); conn.commit()


def create_work_order(asset_id, report_id, fault_type, priority, assigned_to="Unassigned", department="PUBLIC_LIGHTING", notes="") -> str:
    wid = f"WO-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
    now = datetime.now().isoformat(timespec="seconds")
    with connect_db() as conn:
        conn.execute("INSERT INTO work_orders(work_order_id,asset_id,report_id,status,priority,fault_type,assigned_to,assigned_department,reported_at,notes,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (wid, asset_id, report_id, "Open", priority, fault_type, assigned_to, department, now, notes, now)); conn.commit()
    return wid


def get_work_orders() -> pd.DataFrame:
    with connect_db() as conn:
        rows = conn.execute("SELECT * FROM work_orders ORDER BY updated_at DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def update_work_order(work_order_id, **fields):
    allowed = {"status", "priority", "assigned_to", "assigned_department", "notes"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")
    clauses = ", ".join(f"{k}=?" for k in updates)
    with connect_db() as conn:
        conn.execute(f"UPDATE work_orders SET {clauses} WHERE work_order_id=?", tuple(updates.values()) + (work_order_id,)); conn.commit()


def get_audit_logs(limit=200) -> pd.DataFrame:
    with connect_db() as conn:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

# =====================================================================
# SESSION / APPLICATION CONTEXT
# =====================================================================

@dataclass
class Session:
    user: Optional[dict] = None
    notifications: List[str] = field(default_factory=list)
    token: Optional[str] = None

    @property
    def role(self):
        return self.user.get("role") if self.user else None

    @property
    def authenticated(self):
        return self.user is not None

    def logout(self):
        self.user = None
        self.notifications = []
        self.token = None


def _widget_float(widget, default: float) -> float:
    try:
        v = float(widget.value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


class AppContext:
    """Holds all per-session, per-user mutable state: the logged-in session, the decision widgets,
    this session's AHP pairwise matrix and a small results cache. A fresh AppContext is created
    for every browser session by create_dashboard(). The user's working state is saved to the
    database (save_state) and restored at the next sign-in or after a page refresh."""

    def __init__(self):
        self.session = Session()
        self.route = "home"
        # `scenario` is the weighting method: AHP (default) or one of the fixed preset weight sets.
        self.scenario = pn.widgets.Select(name="Weighting method", options=list(WEIGHTING_METHODS), value=AHP_METHOD)
        self.coverage_radius = pn.widgets.IntSlider(name="Coverage radius (m)", start=50, end=150, step=5, value=DEFAULT_COVERAGE_RADIUS_M)
        self.budget = pn.widgets.FloatInput(name="Available budget (USD)", value=DEFAULT_BUDGET_USD, step=1000, start=0)
        self.unit_cost = pn.widgets.FloatInput(name="Unit installation cost (USD)", value=DEFAULT_UNIT_COST_USD, step=100)
        self.contingency = pn.widgets.FloatInput(name="Contingency (%)", value=DEFAULT_CONTINGENCY_PCT, step=1)
        self.max_candidates = pn.widgets.IntInput(name="Maximum candidate sites", value=DEFAULT_MAX_CANDIDATES, start=25, end=1000, step=25)
        self.min_spacing = pn.widgets.IntSlider(name="Minimum spacing between new lights (m)", start=20, end=200, step=5, value=DEFAULT_MIN_SPACING_M)
        self.hard_constraints = pn.widgets.Checkbox(name="Exclude sites inside flood / wetland constraint areas", value=True)
        self.ahp_matrix: np.ndarray = AHP_INITIAL_MATRIX.copy()
        self.ahp_source: str = AHP_INITIAL_SOURCE
        self._ahp_cache: Optional[Tuple[bytes, Dict[str, Any]]] = None
        self.last_scored: Optional[gpd.GeoDataFrame] = None
        self._cache_key = None
        # KPI card -> Streetlight Explorer state (per browser session)
        self.explorer_filter = "all"
        self.explorer_selected = False    # True once the user has picked a filter this session
        self.explorer_search = ""
        self.map_selected: Optional[str] = None      # streetlight / site last clicked on the map
        self.basemap = "satellite"                    # "satellite" (Google hybrid) or "osm"
        self.report_prefill: Dict[str, Any] = {}      # streetlight picked on the map for a problem report
        self.navigate = lambda route: None  # replaced by create_dashboard() with the real router
        self.bridge = None                  # SessionBridge (browser cookie), set by create_dashboard()
        self._restoring = False
        for widget in (self.scenario, self.coverage_radius, self.budget, self.unit_cost, self.contingency,
                       self.max_candidates, self.min_spacing, self.hard_constraints):
            widget.param.watch(lambda event: self.save_state(), "value")

    # -- saved sessions ----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route, "explorer_filter": self.explorer_filter, "explorer_search": self.explorer_search,
            "explorer_selected": bool(self.explorer_selected), "map_selected": self.map_selected, "basemap": self.basemap,
            "scenario": self.scenario.value, "coverage_radius": int(_widget_float(self.coverage_radius, DEFAULT_COVERAGE_RADIUS_M)),
            "budget": _widget_float(self.budget, DEFAULT_BUDGET_USD), "unit_cost": _widget_float(self.unit_cost, DEFAULT_UNIT_COST_USD),
            "contingency": _widget_float(self.contingency, DEFAULT_CONTINGENCY_PCT),
            "max_candidates": int(_widget_float(self.max_candidates, DEFAULT_MAX_CANDIDATES)),
            "min_spacing": int(_widget_float(self.min_spacing, DEFAULT_MIN_SPACING_M)),
            "hard_constraints": bool(self.hard_constraints.value),
            "ahp_matrix": np.asarray(self.ahp_matrix, dtype=float).tolist(), "ahp_source": self.ahp_source,
        }

    def save_state(self) -> None:
        """Persist this user's working state and extend the session. Never raises."""
        if self._restoring or not self.session.user:
            return
        try:
            save_user_state(self.session.user["id"], self.state_dict())
            touch_session(self.session.token)
        except Exception as exc:
            logger.debug("Could not save user state: %s", exc)

    def apply_state(self, state: Dict[str, Any]) -> None:
        """Restore saved settings. Every value is validated, and anything invalid is ignored."""
        if not isinstance(state, dict) or not state:
            return
        role = self.session.role
        self._restoring = True
        try:
            if state.get("scenario") in WEIGHTING_METHODS:
                self.scenario.value = state["scenario"]
            for widget, key, caster in ((self.coverage_radius, "coverage_radius", int), (self.budget, "budget", float),
                                        (self.unit_cost, "unit_cost", float), (self.contingency, "contingency", float),
                                        (self.max_candidates, "max_candidates", int), (self.min_spacing, "min_spacing", int)):
                try:
                    value = caster(state[key])
                    if np.isfinite(value):
                        widget.value = value
                except Exception:
                    pass
            if isinstance(state.get("hard_constraints"), bool):
                self.hard_constraints.value = state["hard_constraints"]
            try:
                m = np.asarray(state.get("ahp_matrix"), dtype=float)
                n = len(CRITERIA)
                if m.shape == (n, n) and np.all(np.isfinite(m)) and np.all(m > 0) and np.allclose(m * m.T, 1.0, atol=1e-4):
                    self.ahp_matrix = m.copy()
                    self.ahp_source = str(state.get("ahp_source") or "restored from your saved session")[:200]
                    self._ahp_cache = None
            except Exception:
                pass
            if state.get("explorer_filter") in allowed_explorer_filters(role):
                self.explorer_filter = state["explorer_filter"]
            self.explorer_search = str(state.get("explorer_search") or "")[:120]
            self.explorer_selected = bool(state.get("explorer_selected"))
            ms = state.get("map_selected")
            self.map_selected = str(ms)[:80] if ms else None
            if state.get("basemap") in BASEMAPS:
                self.basemap = state["basemap"]
            route = state.get("route")
            if isinstance(route, str) and route in {r for r, _ in NAV.get(role, [])} and can_route(role, route):
                self.route = route
        except Exception as exc:
            logger.warning("Saved state could not be fully restored: %s", exc)
        finally:
            self._restoring = False
            self.invalidate()

    def _sync_cookie(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge.max_age = int(SESSION_TTL_HOURS * 3600)
                self.bridge.token = self.session.token or ""
            except Exception as exc:
                logger.debug("Could not update the session cookie: %s", exc)

    def begin_session(self, user: dict) -> None:
        """A successful sign-in: open a stored session, restore the user's saved state, set the cookie."""
        self.session.user = user
        self.session.notifications = ["Secure session established."]
        self.session.token = create_session(user)
        self.route = "home"
        self.reset_explorer_state()
        self.apply_state(load_user_state(user["id"]))
        self._sync_cookie()

    def resume_session(self, user: dict, token: str) -> None:
        """The browser presented a valid session cookie (page refresh / returning visitor)."""
        self.session.user = user
        self.session.token = token
        self.session.notifications = ["Welcome back — your session was restored."]
        self.apply_state(load_user_state(user["id"]))
        self._sync_cookie()
        audit(user["username"], user["role"], "session_resumed", "")

    def end_session(self) -> None:
        user = self.session.user or {}
        self.save_state()
        revoke_session(self.session.token)
        audit(user.get("username"), self.session.role, "logout", "")
        self.session.logout()
        self._sync_cookie()
        self.invalidate()
        self.reset_explorer_state()
        self.map_selected = None
        self.report_prefill = {}
        self.route = "home"

    # -- KPI card navigation ---------------------------------------------
    def open_explorer(self, filter_key: str) -> None:
        """Called by a KPI card: select the filter, then open the Streetlight Map. Respects the
        role's allowed filters and the route permission, so a card can never open a page the
        account may not use."""
        role = self.session.role
        if filter_key not in allowed_explorer_filters(role):
            filter_key = "all"
        if not can_route(role, "explorer"):
            try:
                pn.state.notifications.warning("Your account does not have permission to open the streetlight map.", duration=4000)
            except Exception:
                logger.info("Explorer permission denied for role=%s", role)
            return
        self.explorer_filter, self.explorer_selected, self.explorer_search = filter_key, True, ""
        audit((self.session.user or {}).get("username"), role, "open_explorer", filter_key)
        self.navigate("explorer")

    def reset_explorer_state(self) -> None:
        self.explorer_filter, self.explorer_selected, self.explorer_search = "all", False, ""

    # -- AHP -------------------------------------------------------------
    def ahp(self) -> Dict[str, Any]:
        key = self.ahp_matrix.tobytes()
        if self._ahp_cache is None or self._ahp_cache[0] != key:
            self._ahp_cache = (key, ahp_analysis(self.ahp_matrix))
        return self._ahp_cache[1]

    def set_ahp_matrix(self, matrix: np.ndarray, source: str) -> None:
        self.ahp_matrix = np.asarray(matrix, dtype=float).copy()
        self.ahp_source = source
        self.invalidate()
        self.save_state()

    def weights(self) -> Dict[str, float]:
        if self.scenario.value == AHP_METHOD:
            return dict(self.ahp()["weights_by_key"])
        return {**DEFAULT_WEIGHTS, **SCENARIOS[self.scenario.value]}

    # -- scoring ---------------------------------------------------------
    def scored(self) -> gpd.GeoDataFrame:
        method = self.scenario.value
        key = (method, int(_widget_float(self.max_candidates, DEFAULT_MAX_CANDIDATES)), _widget_float(self.budget, DEFAULT_BUDGET_USD),
               _widget_float(self.unit_cost, DEFAULT_UNIT_COST_USD), _widget_float(self.contingency, DEFAULT_CONTINGENCY_PCT),
               int(_widget_float(self.min_spacing, DEFAULT_MIN_SPACING_M)), bool(self.hard_constraints.value),
               self.ahp_matrix.tobytes() if method == AHP_METHOD else None)
        if self.last_scored is not None and self._cache_key == key:
            return self.last_scored
        candidates = generate_candidates(DATA.streetlights, DATA.roads, spacing_m=CANDIDATE_SPACING_M, max_candidates=int(key[1]))
        result = score_candidates(candidates, self.weights(), key[2], key[3], key[4], float(key[5]), key[6])
        self.last_scored = result
        self._cache_key = key
        return result

    def invalidate(self):
        self.last_scored = None
        self._cache_key = None

# =====================================================================
# UI HELPERS
# =====================================================================

def page_header(title: str, subtitle: str, role: Optional[str] = None):
    """Page title block. The role badge lives in the top bar only (never repeated per page); the
    `role` argument is accepted for call-site compatibility."""
    return pn.pane.HTML(
        f"<div class='moc-page-head'><div class='moc-page-title'>{esc(title)}</div><div class='moc-page-subtitle'>{esc(subtitle)}</div></div>",
        sizing_mode="stretch_width",
    )


def card(title: str, body: str):
    return pn.pane.HTML(f"<div class='moc-card'><div class='moc-card-title'>{title}</div><div class='moc-card-subtitle'>{body}</div></div>", sizing_mode="stretch_width")


def section_title(text: str):
    return pn.pane.HTML(f"<div class='moc-section-title'>{esc(text)}</div>")


def alert(text: str, kind="info"):
    return pn.pane.HTML(f"<div class='moc-alert moc-alert-{kind}'>{text}</div>", sizing_mode="stretch_width")


def _hidden():
    """A zero-height placeholder. Data-mode / AHP-method notes no longer sit on working pages;
    they live on the admin 'System Notices' page, with compact pills in the top bar."""
    return pn.pane.HTML("", height=0, margin=0, sizing_mode="stretch_width")


def data_mode_alert():
    return _hidden()


def kpi(label: str, value, sub="", accent=COLOR_GREEN):
    return pn.pane.HTML(f"<div class='moc-kpi' style='--accent:{accent};'><div class='moc-kpi-label'>{esc(label)}</div><div class='moc-kpi-value'>{esc(value)}</div><div class='moc-kpi-sub'>{esc(sub)}</div></div>", sizing_mode="stretch_width")


# ---------------------------------------------------------------------
# Clickable KPI cards and the browser-session cookie bridge
# ---------------------------------------------------------------------
_COOKIE_JS = (
    "const secure = (location.protocol === 'https:') ? '; Secure' : '';"
    "if (data.token) { document.cookie = data.cookie + '=' + data.token + '; path=/; max-age=' + data.max_age + '; SameSite=Lax' + secure; }"
    "else { document.cookie = data.cookie + '=; path=/; max-age=0; SameSite=Lax' + secure; }"
)
try:
    import param
    from panel.reactive import ReactiveHTML

    class KPICard(ReactiveHTML):
        """A whole-surface clickable KPI card. Clicking (or pressing Enter / Space while it has keyboard
        focus) increments ``clicks`` and so fires the Python callback registered by kpi_card(). Text
        params use the literal ``{{ }}`` syntax and are escaped with esc() by the caller."""
        label = param.String(default="")
        value = param.String(default="")
        sub = param.String(default="")
        hint = param.String(default="")
        state = param.String(default="moc-kpi-green")
        aria = param.String(default="")
        clicks = param.Integer(default=0)

        _template = """
<div id="card" class="moc-kpi moc-kpi-click ${state}" role="button" tabindex="0" aria-label="${aria}"
     onclick="${script('activate')}" onkeydown="${script('key')}">
  <div class="moc-kpi-label">{{ label }}</div>
  <div class="moc-kpi-value">{{ value }}</div>
  <div class="moc-kpi-sub">{{ sub }}</div>
  <div class="moc-kpi-hint">{{ hint }}</div>
</div>"""
        _scripts = {
            "activate": "data.clicks += 1",
            "key": "if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); data.clicks += 1 }",
        }

    class SessionBridge(ReactiveHTML):
        """Invisible component that writes (or clears) the session cookie in the browser. The token is
        random and only its hash is kept on the server; the cookie lets a refreshed page or a returning
        visitor resume the saved session without signing in again."""
        token = param.String(default="")
        max_age = param.Integer(default=int(SESSION_TTL_HOURS * 3600))
        cookie = param.String(default=SESSION_COOKIE)

        _template = '<div id="bridge" style="display:none"></div>'
        _scripts = {"render": _COOKIE_JS, "token": _COOKIE_JS, "max_age": _COOKIE_JS}
except Exception as _exc:  # very old/new Panel without ReactiveHTML
    KPICard = None
    SessionBridge = None
    logger.warning("ReactiveHTML unavailable (%s); KPI cards fall back to a card plus an 'Open list' button and "
                   "browser-cookie session resume is disabled (sessions are still saved on the server).", _exc)


def kpi_card(label: str, value, sub: str, accent_class: str, on_click, selected: bool = False, hint: str = "Click to open the map and list ›"):
    """A KPI card that opens the Streetlight Map. Falls back to a static card with an explicit
    button if ReactiveHTML is unavailable, so navigation still works."""
    classes = accent_class + (" moc-kpi-selected" if selected else "")
    if KPICard is not None:
        card_ = KPICard(
            label=esc(label), value=esc(value), sub=esc(sub), hint=esc(hint), state=classes,
            aria=esc(f"{label}: {value}. {sub}. Activate to open the map and list for this category."),
            sizing_mode="stretch_width",
        )
        card_.param.watch(lambda event: on_click(), "clicks")
        return card_
    button = pn.widgets.Button(name=hint, button_type="primary", sizing_mode="stretch_width")
    button.on_click(lambda _: on_click())
    accent = {"moc-kpi-green": COLOR_GREEN, "moc-kpi-teal": COLOR_TEAL, "moc-kpi-red": COLOR_RED, "moc-kpi-orange": COLOR_ORANGE, "moc-kpi-navy": COLOR_NAVY}.get(accent_class, COLOR_GREEN)
    return pn.Column(kpi(label, value, sub, accent), button, sizing_mode="stretch_width")


def streetlight_kpi_row(ctx: "AppContext", include_new_sites: bool = True) -> pn.Row:
    """The streetlight KPI cards. Every number comes from STREETLIGHT_REGISTRY (and, for the last card,
    from the AHP/MCDA selection), and each card opens the Streetlight Map pre-filtered to its
    category, so the cards cannot disagree with the lists."""
    role = ctx.session.role
    s = streetlight_summary()
    sel = lambda key: ctx.explorer_selected and ctx.explorer_filter == key
    pct = f"{100 * s['operational'] / s['total']:.0f}% of inventory" if s["total"] else "no records"
    src = {"source": f"status from “{s['status_column']}”", "simulated": "SIMULATED status", "unknown": "status unavailable"}[s["status_kind"]]
    unmapped = f" · {s['unmapped']} without valid coordinates" if s["unmapped"] else ""
    cards = [
        kpi_card("Existing streetlights", s["total"], f"Asset inventory · {s['mapped']} with valid coordinates{unmapped}", "moc-kpi-green", lambda: ctx.open_explorer("all"), sel("all")),
        kpi_card("Operational", s["operational"], f"{pct} · {src}", "moc-kpi-teal", lambda: ctx.open_explorer("operational"), sel("operational")),
        kpi_card("Fault / offline", s["fault_offline"], "Faulty + offline · requires attention", "moc-kpi-red", lambda: ctx.open_explorer("attention"), sel("attention")),
    ]
    if include_new_sites and role != "COMMUNITY":
        scored = ctx.scored()
        diag = priority_diagnostics(scored)
        n_rec = int(scored["funded_phase_1"].sum()) if not scored.empty else 0
        if diag["n_candidates"] == 0:
            sub = "No candidate sites generated — see Decision Engine diagnostics"
        else:
            sub = f"Best of {diag['n_candidates']} candidates · ${_widget_float(ctx.budget, 0):,.0f} budget · ≥ {int(_widget_float(ctx.min_spacing, 0))} m apart"
        cards.append(kpi_card("Recommended new solar lights", n_rec, sub, "moc-kpi-orange", lambda: ctx.open_explorer("recommended"), sel("recommended"), hint="Click to map the sites ›"))
    return pn.Row(*cards, sizing_mode="stretch_width")


def ahp_banner_content(ctx: "AppContext") -> Tuple[str, str]:
    """(alert kind, HTML text) describing where the current criterion weights come from."""
    if ctx.scenario.value == AHP_METHOD:
        a = ctx.ahp()
        text = (f"<b>Weights derived by AHP</b> from the pairwise-comparison matrix ({esc(ctx.ahp_source)}). "
                f"λmax = {a['lambda_max']:.3f} · CI = {a['ci']:.3f} · CR = {a['cr']:.3f} ")
        if a["consistent"]:
            return "ok", text + f"— the judgements are acceptably consistent (CR below {AHP_CR_THRESHOLD:.2f})."
        return "warn", text + (f"— <b>CR is not below {AHP_CR_THRESHOLD:.2f}: the judgements are inconsistent, so these weights and the resulting "
                               f"ranking are provisional.</b> Revise them on the AHP Criteria Weights page.")
    return "info", (f"Weights come from the preset “{esc(ctx.scenario.value)}” — fixed, expert-set weights that are <b>not</b> derived by AHP. "
                    f"Choose “{esc(AHP_METHOD)}” as the weighting method to use the pairwise-comparison weights.")


def ahp_banner(ctx: "AppContext"):
    return _hidden()


def make_table(df: pd.DataFrame, width=None, height=420, formatters=None, selectable: bool = False, page_size: int = 12):
    """A read-only paged table. With selectable=True one row can be clicked (used to highlight a
    streetlight on the map)."""
    if df is None or df.empty:
        return pn.pane.Markdown("_No records available._")
    kwargs: Dict[str, Any] = dict(pagination="local", page_size=page_size, height=height, width=width, layout="fit_data_stretch",
                                  show_index=False, formatters=formatters or {}, disabled=True)
    if selectable:
        kwargs["selectable"] = 1
    return pn.widgets.Tabulator(df, **kwargs)


def df_table(df: pd.DataFrame, width=None, height=420, formatters=None):
    return make_table(df, width=width, height=height, formatters=formatters)


def html_bar_chart(title: str, data: Dict[str, float], color: str = COLOR_GREEN, note: str = ""):
    """A dependency-free HTML/CSS bar chart, used instead of pulling in a charting library."""
    if not data:
        return pn.pane.HTML(f"<div class='moc-card'><div class='moc-card-title'>{esc(title)}</div><div class='moc-card-subtitle'>No data available.</div></div>", sizing_mode="stretch_width")
    maximum = max(data.values()) or 1
    rows = "".join(
        f"<div style='margin:7px 0;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:11px;color:#3A4B5C;'><span>{esc(k)}</span><span style='font-weight:800;'>{esc(v)}</span></div>"
        f"<div style='background:#EEF3F6;border-radius:6px;height:8px;overflow:hidden;margin-top:3px;'>"
        f"<div style='width:{max(4, 100 * float(v) / float(maximum)):.0f}%;background:{color};height:100%;'></div></div></div>"
        for k, v in data.items()
    )
    footer = f"<div class='moc-card-subtitle' style='margin-top:6px;'>{esc(note)}</div>" if note else ""
    return pn.pane.HTML(f"<div class='moc-card'><div class='moc-card-title'>{esc(title)}</div>{rows}{footer}</div>", sizing_mode="stretch_width")


def _download_callback(data: bytes):
    """FileDownload needs a file-like object, not raw bytes."""
    return lambda *args: io.BytesIO(data)


def recommended_geojson(scored: Optional[pd.DataFrame]) -> bytes:
    """GeoJSON of the recommended new lights (for QGIS / ArcGIS)."""
    if scored is None or scored.empty or "funded_phase_1" not in scored.columns or not bool(scored["funded_phase_1"].any()):
        return b'{"type": "FeatureCollection", "features": []}'
    rec = scored[scored["funded_phase_1"].astype(bool)].sort_values("rank")
    keep = [c for c in RECOMMENDED_COLUMNS if c in rec.columns] + ["geometry"]
    return rec[keep].to_json().encode("utf-8")

# =====================================================================
# LOGIN
# =====================================================================

def build_login(ctx: AppContext, on_success):
    username = pn.widgets.TextInput(name="Username", placeholder="Enter username", sizing_mode="stretch_width")
    password = pn.widgets.PasswordInput(name="Password", placeholder="Enter password", sizing_mode="stretch_width")
    sign_in = pn.widgets.Button(name="Sign in securely", button_type="success", height=46, sizing_mode="stretch_width")
    message = pn.pane.HTML("", sizing_mode="stretch_width")

    def do_login(_=None):
        sign_in.disabled = True
        try:
            user = authenticate(username.value or "", password.value or "")
            if user is None:
                audit(username.value, None, "login_failed", "Invalid credentials")
                message.object = "<div class='moc-alert moc-alert-danger'>Invalid username or password. Please check your credentials.</div>"
                return
            ctx.begin_session(user)
            audit(user["username"], user["role"], "login_success", "")
            on_success()
        finally:
            sign_in.disabled = False

    sign_in.on_click(do_login)

    hero = pn.pane.HTML(
        f"<div class='moc-login-hero' style='text-align:center;align-items:center;'>"
        f"{logo_html('moc-logo-lg')}"
        f"<div class='moc-login-title'>Smart Solar Streetlight Management</div>"
        f"<div class='moc-login-sub'>Municipality of Chinhoyi · GIS-enabled placement, monitoring, maintenance and decision support for high-density suburbs.</div>"
        f"<div style='margin-top:18px;display:flex;gap:6px;flex-wrap:wrap;justify-content:center;'>"
        f"<span class='moc-pill pill-gold'>☀ Solar streetlights</span>"
        f"<span class='moc-pill pill-green'>⌖ Interactive map</span>"
        f"<span class='moc-pill pill-blue'>⚙ AHP / MCDA decision support</span>"
        f"</div>"
        f"</div>", sizing_mode="stretch_width")

    credential_note = (
        "Development accounts are seeded automatically with the default passwords documented in this file. "
        "Replace them with ADMIN_DEFAULT_PASSWORD / STAFF_DEFAULT_PASSWORD / COMMUNITY_DEFAULT_PASSWORD before any real municipal deployment."
        if USING_DEFAULT_CREDENTIALS else
        "This deployment is using administrator-configured credentials rather than the documentation defaults."
    )

    form = pn.Column(
        pn.pane.HTML(f"<div style='font-size:24px;font-weight:800;color:#102A43;'>Welcome back</div><div style='font-size:12px;color:#64748B;margin:5px 0 18px;'>Sign in to continue. Your session is saved, so a page refresh will not sign you out (sessions last up to {SESSION_TTL_HOURS:g} hours).</div>"),
        username, password, message, sign_in,
        pn.pane.HTML(f"<div style='margin-top:15px;font-size:10px;color:#8190A0;line-height:1.6;'>{esc(credential_note)}</div>"),
        sizing_mode="stretch_width", css_classes=["moc-login-form"])

    row = pn.Row(hero, form, sizing_mode="stretch_width", css_classes=["moc-login-row"])
    return pn.Column(pn.Column(row, sizing_mode="stretch_width", css_classes=["moc-login-card"]),
                     sizing_mode="stretch_width", css_classes=["moc-login-shell"])

# =====================================================================
# TOP BAR / SIDEBAR  (one top bar, one navigation list, no repeated branding)
# =====================================================================

def status_pills_html(ctx: AppContext) -> str:
    """Compact coloured pills for the few facts a reader must never miss; the detail is on the
    admin 'System Notices' page."""
    pills: List[Tuple[str, str, str]] = []
    if not CORE_REAL_DATA:
        pills.append(("DEMO DATA", "#FFE08A", "#5B4100"))
    if streetlight_summary()["status_kind"] == "simulated":
        pills.append(("SIMULATED STATUS", "#FFD2A8", "#7A3A00"))
    if ctx.session.role != "COMMUNITY":
        try:
            if ctx.scenario.value == AHP_METHOD and not ctx.ahp()["consistent"]:
                pills.append(("AHP PROVISIONAL", "#FFC2C2", "#7A1010"))
        except Exception:
            pass
    return "".join(f"<span style='display:inline-block;margin:4px 6px 0 0;padding:2px 9px;border-radius:999px;font-size:9.5px;"
                   f"font-weight:800;background:{bg};color:{fg};'>{esc(t)}</span>" for t, bg, fg in pills)


def build_header(ctx: AppContext, rerender):
    """The single top bar, in three zones: the signed-in user (left), the Municipality logo and name
    (centre) and notifications / sign-out (right). Colours are inline so they always show."""
    user = ctx.session.user or {}
    role = ctx.session.role
    unread = len(get_notifications(user.get("id"), unread_only=True)) if user.get("id") else 0
    notif = pn.widgets.Button(name=f"🔔 {unread}", button_type="warning", width=78, height=36)
    logout = pn.widgets.Button(name="Sign out", button_type="danger", width=95, height=36)
    panel = pn.Column(visible=False, sizing_mode="stretch_width")

    def toggle(_=None):
        rows = get_notifications(user.get("id"), limit=8) if user.get("id") else pd.DataFrame()
        msgs = (rows.message.tolist() if not rows.empty else []) + list(ctx.session.notifications)
        deduped = list(dict.fromkeys(msgs))[-8:]
        panel[:] = [pn.pane.Markdown("**Notifications**"), *(pn.pane.Markdown(f"- {esc(m)}") for m in deduped)] if deduped else [pn.pane.Markdown("_No new notifications._")]
        panel.visible = not panel.visible
        if user.get("id"):
            mark_notifications_read(user["id"])

    def do_logout(_=None):
        ctx.end_session()
        rerender()

    notif.on_click(toggle)
    logout.on_click(do_logout)

    dept = DEPARTMENTS.get(user.get("department"), "") or ROLE_META.get(role, {}).get("label", "")
    role_label = {"ADMIN": "ADMIN", "MUNICIPAL_STAFF": "MUNICIPAL STAFF", "COMMUNITY": "COMMUNITY"}.get(role, str(role))
    left = pn.pane.HTML(
        f"<div style='text-align:left;line-height:1.35;'>"
        f"<div style='font-size:15px;font-weight:800;color:#FFFFFF;'>{esc(user.get('full_name', 'User'))}</div>"
        f"<div style='margin-top:3px;'><span style='display:inline-block;padding:2px 10px;border-radius:999px;font-size:10px;font-weight:800;"
        f"background:#F4B41A;color:#3B2A00;'>{esc(role_label)}</span> "
        f"<span style='font-size:11px;color:#E8F7EE;'>{esc(dept)}</span></div>"
        f"<div style='font-size:10.5px;color:#CDEBD8;margin-top:2px;'>{esc(user.get('email', ''))}</div>"
        f"<div>{status_pills_html(ctx)}</div></div>",
        css_classes=["moc-topbar-left"], sizing_mode="stretch_width")
    centre = pn.pane.HTML(
        f"<div class='moc-brand' style='display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:4px;'>"
        f"{logo_html()}"
        f"<div style='font-size:20px;font-weight:800;letter-spacing:.5px;color:#FFFFFF;line-height:1.15;'>MUNICIPALITY OF CHINHOYI</div>"
        f"<div style='font-size:11.5px;font-weight:600;color:#FFE08A;'>Smart Solar Streetlight Management &amp; Decision Support</div></div>",
        css_classes=["moc-topbar-center"])
    right = pn.Row(pn.HSpacer(), notif, logout, css_classes=["moc-topbar-right"], sizing_mode="stretch_width")
    try:
        topbar = pn.Row(left, centre, right, css_classes=["moc-topbar"], sizing_mode="stretch_width",
                        styles={"background": "linear-gradient(100deg,#054A28 0%,#0B8A4A 50%,#054A28 100%)", "border-bottom": "4px solid #F4B41A",
                                "padding": "10px 20px", "align-items": "center", "box-shadow": "0 4px 18px rgba(5,74,40,.30)"})
    except Exception:
        topbar = pn.Row(left, centre, right, css_classes=["moc-topbar"], sizing_mode="stretch_width")
    return pn.Column(topbar, panel, sizing_mode="stretch_width")


def build_sidebar(ctx: AppContext, navigate) -> List[Any]:
    """Navigation only. Who is signed in is shown once, in the top bar."""
    role = ctx.session.role
    items = [(route, label) for route, label in NAV.get(role, []) if can_route(role, route)]
    domain = ROLE_META.get(role, {})
    objects: List[Any] = [pn.pane.HTML(
        f"<div class='moc-side-head'><div class='moc-side-title'>{esc(domain.get('label', 'Dashboard'))}</div>"
        f"<div class='moc-side-sub'>{esc(domain.get('description', ''))}</div></div>")]
    for route, label in items:
        b = pn.widgets.Button(name=("●  " if ctx.route == route else "  ") + label, button_type="success" if ctx.route == route else "light", sizing_mode="stretch_width", height=36)
        b.on_click(lambda e, r=route: navigate(r))
        objects.append(b)
    objects.append(pn.pane.HTML(f"<div class='moc-side-note'>Your session and settings are saved automatically and restored when you return.</div>"))
    return objects

# =====================================================================
# INTERACTIVE STREETLIGHT MAP (Bokeh) — click a streetlight to see its details
# =====================================================================
# Streetlights, roads and the municipal boundary are drawn by Bokeh itself, so the map never comes up
# blank. The street basemap tiles are fetched by the browser from the internet when it is online.

STATUS_COLORS = {STATUS_OPERATIONAL: COLOR_GREEN, STATUS_FAULT_OFFLINE: COLOR_RED, STATUS_MAINTENANCE: COLOR_ORANGE, STATUS_UNCLASSIFIED: COLOR_GRAY}
STATUS_NAMES = {STATUS_OPERATIONAL: "Operational", STATUS_FAULT_OFFLINE: "Fault / offline", STATUS_MAINTENANCE: "Maintenance", STATUS_UNCLASSIFIED: "Status unknown"}
PRIORITY_COLORS = {"VERY HIGH": "#B71C1C", "HIGH": COLOR_ORANGE, "MODERATE": COLOR_AMBER, "LOW": "#4C8BC9", "VERY LOW": COLOR_GRAY}

LIGHT_DETAIL_FIELDS_STAFF = [
    ("asset_id", "Asset ID"), ("status_label", "Status"), ("status_raw", "Recorded as"), ("suburb", "Suburb"), ("light_type", "Light type"),
    ("latitude", "Latitude"), ("longitude", "Longitude"), ("utm_easting", "UTM easting (m)"), ("utm_northing", "UTM northing (m)"),
    ("fault_description", "Fault description"), ("install_date", "Installed"), ("last_inspection", "Last inspection"),
    ("last_maintenance", "Last maintenance"), ("maintenance_notes", "Maintenance notes"), ("service_status", "Service status"),
]
LIGHT_DETAIL_FIELDS_PUBLIC = [("asset_id", "Streetlight ID"), ("status_label", "Status"), ("suburb", "Suburb"), ("light_type", "Light type")]
SITE_DETAIL_FIELDS = [
    ("candidate_id", "Candidate ID"), ("new_light_id", "New light ID"), ("rank", "Rank"), ("road_name", "Road"), ("road_type", "Road type"),
    ("latitude", "Latitude"), ("longitude", "Longitude"), ("utm_easting", "UTM easting (m)"), ("utm_northing", "UTM northing (m)"),
    ("overall_priority", "Suitability (0–100)"), ("priority_class", "Priority class"), ("safety_score", "Demand & safety score"),
    ("technical_score", "Technical score"), ("main_driver", "Main driver"), ("selection_status", "Selection status"),
    ("estimated_cost", "Estimated cost"), ("implementation_phase", "Phase"), ("existing_light_dist_m", "Nearest existing light (m)"),
    ("population", "Population within 250 m"),
]
_MAP_BASE_CACHE: Dict[str, Any] = {}


def _mercator(lon, lat):
    """WGS84 degrees -> Web Mercator metres (the coordinate system of web basemap tiles)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -85.0, 85.0)
    k = 6378137.0
    return np.radians(lon) * k, np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0)) * k


def _plain(value):
    """A JSON/HTML-friendly Python scalar, or None for blanks."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    return value


def _format_value(key: str, value) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if key in ("latitude", "longitude"):
            return f"{float(value):.6f}"
        if key in ("utm_easting", "utm_northing"):
            return f"{float(value):,.2f}"
        if key == "estimated_cost":
            return f"${float(value):,.0f}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):.1f}"
    return str(value)


def _line_coords(gdf, limit: int) -> Tuple[List[List[float]], List[List[float]]]:
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    if gdf is None or gdf.empty:
        return xs, ys
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            geom = geom.boundary
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            if part.geom_type not in ("LineString", "LinearRing"):
                continue
            lon, lat = part.xy
            x, y = _mercator(lon, lat)
            xs.append(x.tolist())
            ys.append(y.tolist())
            if len(xs) >= limit:
                return xs, ys
    return xs, ys


def _basemap_vectors() -> Dict[str, Any]:
    """Roads, municipal boundary and existing-light points in Web Mercator, computed once."""
    if "vectors" not in _MAP_BASE_CACHE:
        vectors: Dict[str, Any] = {"roads": ([], []), "boundary": ([], []), "lights": ([], [])}
        try:
            vectors["roads"] = _line_coords(DATA.roads, 4000)
            vectors["boundary"] = _line_coords(DATA.boundary, 50)
            reg = STREETLIGHT_REGISTRY
            if not reg.empty and bool(reg["coord_valid"].any()):
                ok = reg[reg["coord_valid"]].head(4000)
                x, y = _mercator(ok["longitude"].to_numpy(), ok["latitude"].to_numpy())
                vectors["lights"] = (x.tolist(), y.tolist())
        except Exception as exc:
            logger.warning("Map base layers could not be prepared: %s", exc)
        _MAP_BASE_CACHE["vectors"] = vectors
    return _MAP_BASE_CACHE["vectors"]


BASEMAPS: Dict[str, Tuple[str, str, str]] = {
    "satellite": ("Satellite hybrid (Google)", "https://mt1.google.com/vt/lyrs=y&x={X}&y={Y}&z={Z}", "Imagery © Google"),
    "osm": ("OpenStreetMap", "https://tile.openstreetmap.org/{Z}/{X}/{Y}.png", "© OpenStreetMap contributors"),
}


def report_prefill_from(rec: Dict[str, Any]) -> Dict[str, Any]:
    """What a map click passes to the problem-report form."""
    where = " · ".join(str(x) for x in (rec.get("suburb"), rec.get("road_name")) if x)
    return {"asset_id": rec.get("asset_id") or rec.get("candidate_id"), "latitude": rec.get("_lat"), "longitude": rec.get("_lon"), "location_text": where}


class StreetlightMap:
    """An interactive map of the streetlights (or candidate sites) listed by explorer_view().
    Tap a point to show its details; ``select(id)`` highlights and zooms to one from the table."""

    def __init__(self, ctx: "AppContext", view: Dict[str, Any], on_report: Optional[Callable[[Dict[str, Any]], None]] = None, height: int = 440):
        self.ctx = ctx
        self.role = ctx.session.role
        self.community = self.role == "COMMUNITY"
        self.data_view = view
        self.kind = view.get("kind", "streetlights")
        self.on_report = on_report if (self.kind == "streetlights" and on_report is not None) else None
        self.records: List[Dict[str, Any]] = []
        self.index_by_id: Dict[str, int] = {}
        self.source = None
        self.fig = None
        self.simulated = view.get("summary", {}).get("status_kind") == "simulated"
        self.detail = pn.pane.HTML(self._hint_html(), sizing_mode="stretch_width")
        self.report_btn = pn.widgets.Button(name="Report a problem with this streetlight", button_type="warning", visible=False)
        self.report_btn.on_click(self._report_clicked)
        self.basemap_switch = pn.widgets.RadioButtonGroup(
            options={v[0]: k for k, v in BASEMAPS.items()}, value=ctx.basemap if ctx.basemap in BASEMAPS else "satellite", button_type="success")
        self.basemap_switch.param.watch(self._on_basemap, "value")
        self.tiles: Dict[str, Any] = {}
        self.road_renderer = None
        self.panel = pn.Column(sizing_mode="stretch_width")
        self._current: Optional[Dict[str, Any]] = None
        self._build(height)

    # -- construction ----------------------------------------------------
    def _hint_html(self) -> str:
        what = "streetlight" if self.kind == "streetlights" else "site"
        return f"<div class='moc-detail'><div class='moc-card-subtitle'>Click a {what} on the map to see its details" + \
               (", then report a problem with it." if self.community else ".") + "</div></div>"

    def _points(self) -> pd.DataFrame:
        mf = self.data_view.get("map_frame")
        if mf is None or len(mf) == 0 or not {"latitude", "longitude"} <= set(mf.columns):
            return pd.DataFrame()
        d = mf.copy()
        if "coord_valid" in d.columns:
            d = d[d["coord_valid"].astype(bool)]
        d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
        d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
        return d[d["latitude"].notna() & d["longitude"].notna()].reset_index(drop=True)

    def _build(self, height: int) -> None:
        if not BOKEH_AVAILABLE:
            self.panel[:] = [alert("The interactive map needs Bokeh, which is installed with Panel (pip install bokeh). The list below still works.", "warn")]
            return
        pts = self._points()
        if pts.empty:
            self.panel[:] = [alert("None of the listed records has valid coordinates, so there is nothing to show on the map. Change the filter, or check the coordinates in the data.", "warn")]
            return
        key_col = "asset_id" if self.kind == "streetlights" else "candidate_id"
        xs, ys = _mercator(pts["longitude"].to_numpy(), pts["latitude"].to_numpy())
        cols = {"x": xs.tolist(), "y": ys.tolist(), "color": [], "line_color": [], "size": [], "label": [], "sub": []}
        for i, (_, row) in enumerate(pts.iterrows()):
            rec = {k: _plain(v) for k, v in row.items() if k != "geometry"}
            rec["_lat"], rec["_lon"], rec["_x"], rec["_y"] = float(row["latitude"]), float(row["longitude"]), float(xs[i]), float(ys[i])
            if self.kind == "streetlights":
                color = STATUS_COLORS.get(str(row.get("status_class")), COLOR_GRAY)
                rec["_title"], rec["_sub"], rec["_color"] = str(rec.get("asset_id")), str(rec.get("status_label") or "Unknown"), color
                cols["color"].append(color); cols["line_color"].append("#FFFFFF"); cols["size"].append(11)
                cols["label"].append(rec["_title"]); cols["sub"].append(rec["_sub"])
            else:
                color = PRIORITY_COLORS.get(str(row.get("priority_class")), COLOR_GRAY)
                funded = bool(rec.get("new_light_id"))
                rec["_title"] = str(rec.get("new_light_id") or rec.get("candidate_id"))
                rec["_sub"] = f"{rec.get('priority_class', '')}" + (" · recommended" if funded else "")
                rec["_color"] = color
                cols["color"].append(color); cols["line_color"].append(COLOR_NAVY_DARK if funded else "#FFFFFF")
                cols["size"].append(15 if funded else 11)
                cols["label"].append(rec["_title"]); cols["sub"].append(rec["_sub"])
            rec["_id"] = str(rec.get(key_col))
            self.records.append(rec)
            self.index_by_id.setdefault(rec["_id"], i)
        self.source = ColumnDataSource(data=cols)

        pad_x = max((xs.max() - xs.min()) * 0.08, 500.0)
        pad_y = max((ys.max() - ys.min()) * 0.08, 500.0)
        fig = bokeh_figure(
            height=height, sizing_mode="stretch_width", x_axis_type="mercator", y_axis_type="mercator",
            x_range=Range1d(float(xs.min() - pad_x), float(xs.max() + pad_x)),
            y_range=Range1d(float(ys.min() - pad_y), float(ys.max() + pad_y)),
            tools="pan,wheel_zoom,box_zoom,reset", active_scroll="wheel_zoom", toolbar_location="above", match_aspect=True,
            background_fill_color="#DDE6E0", outline_line_color="#C9D6DF",
        )
        for key, (_label, url, attribution) in BASEMAPS.items():
            try:
                self.tiles[key] = fig.add_tile(WMTSTileSource(url=url, attribution=attribution, max_zoom=20), visible=(key == self.basemap_switch.value))
            except Exception as exc:
                logger.info("Basemap %s could not be added: %s", key, exc)
        base = _basemap_vectors()
        try:
            if base["roads"][0]:
                # Drawn over the satellite imagery only; the OpenStreetMap tiles already show roads.
                self.road_renderer = fig.multi_line(base["roads"][0], base["roads"][1], line_color="#FFE082", line_width=1.3, line_alpha=0.85,
                                                    visible=(self.basemap_switch.value == "satellite"))
            if base["boundary"][0]:
                fig.multi_line(base["boundary"][0], base["boundary"][1], line_color=COLOR_GOLD, line_width=2.5, line_dash="dashed", line_alpha=0.95)
            if self.kind == "sites" and base["lights"][0]:
                fig.scatter(base["lights"][0], base["lights"][1], size=5, fill_color="#E2E8F0", line_color="#334155", line_width=0.5, fill_alpha=0.9)
        except Exception as exc:
            logger.info("Map base layers skipped: %s", exc)
        renderer = fig.scatter(
            "x", "y", source=self.source, size="size", fill_color="color", line_color="line_color", line_width=1.5, fill_alpha=0.93,
            nonselection_fill_alpha=0.93, nonselection_line_alpha=1.0,
            selection_fill_alpha=1.0, selection_line_color=COLOR_NAVY_DARK, selection_line_width=4,
        )
        fig.add_tools(HoverTool(renderers=[renderer], tooltips="<b>@label</b><br>@sub"))
        fig.add_tools(TapTool(renderers=[renderer]))
        fig.grid.grid_line_alpha = 0.25
        try:
            fig.toolbar.logo = None
        except Exception:
            pass
        self.fig = fig
        self.source.selected.on_change("indices", self._on_selected)
        note = ("Basemaps (Google satellite hybrid, OpenStreetMap) need an internet connection and no API key; the streetlights are drawn either way. "
                "Scroll to zoom, drag to pan.")
        self.panel[:] = [
            pn.pane.HTML(self._legend_html(), sizing_mode="stretch_width"),
            self.basemap_switch,
            pn.pane.Bokeh(fig, height=height, sizing_mode="stretch_width"),
            pn.pane.HTML(f"<div class='moc-card-subtitle' style='padding:4px 2px 8px;'>{esc(note)}</div>", sizing_mode="stretch_width"),
            self.detail, self.report_btn,
        ]
        saved = self.ctx.map_selected
        if saved and saved in self.index_by_id:
            self.select(saved)

    def _legend_html(self) -> str:
        items = []
        if self.kind == "streetlights":
            counts: Dict[str, int] = {}
            for rec in self.records:
                counts[str(rec.get("status_class"))] = counts.get(str(rec.get("status_class")), 0) + 1
            for cls in (STATUS_OPERATIONAL, STATUS_FAULT_OFFLINE, STATUS_MAINTENANCE, STATUS_UNCLASSIFIED):
                if counts.get(cls):
                    items.append(f"<span><span class='moc-dot' style='background:{STATUS_COLORS[cls]}'></span>{STATUS_NAMES[cls]} ({counts[cls]})</span>")
        else:
            counts = {}
            for rec in self.records:
                counts[str(rec.get("priority_class"))] = counts.get(str(rec.get("priority_class")), 0) + 1
            for cls in ("VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"):
                if counts.get(cls):
                    items.append(f"<span><span class='moc-dot' style='background:{PRIORITY_COLORS[cls]}'></span>{cls.title()} priority ({counts[cls]})</span>")
            items.append("<span>Large dot with a dark ring = recommended new light · small grey dots = existing streetlights</span>")
        total = f"<span style='font-weight:800;color:{COLOR_NAVY};'>{len(self.records)} on the map</span>"
        return f"<div class='moc-legend'>{total}{''.join(items)}</div>"

    # -- interaction -----------------------------------------------------
    def _detail_html(self, rec: Dict[str, Any]) -> str:
        if self.kind == "streetlights":
            fields = LIGHT_DETAIL_FIELDS_PUBLIC if self.community else LIGHT_DETAIL_FIELDS_STAFF
        else:
            fields = SITE_DETAIL_FIELDS
        rows = []
        for key, label in fields:
            value = rec.get(key)
            if value is None or value == "":
                continue
            if key == "fault_description" and self.simulated:
                label += " (SIMULATED)"
            rows.append(f"<tr><td>{esc(label)}</td><td>{esc(_format_value(key, value))}</td></tr>")
        pill = (f"<span class='moc-pill' style='background:{rec.get('_color', COLOR_GRAY)};color:#fff;'>{esc(rec.get('_sub', ''))}</span>")
        warn = ""
        if self.simulated and self.kind == "streetlights":
            warn = "<div class='moc-card-subtitle' style='margin-top:6px;color:#8A5A00;'>Status is SIMULATED (no status column in the data).</div>"
        return f"<div class='moc-detail'><div class='moc-detail-title'>{esc(rec.get('_title', ''))} {pill}</div><table>{''.join(rows)}</table>{warn}</div>"

    def _show(self, index: Optional[int]) -> None:
        if index is None or not (0 <= index < len(self.records)):
            self._current = None
            self.detail.object = self._hint_html()
            self.report_btn.visible = False
            return
        rec = self.records[index]
        self._current = rec
        self.detail.object = self._detail_html(rec)
        self.report_btn.visible = bool(self.on_report)

    def _on_selected(self, attr, old, new) -> None:
        index = new[0] if new else None
        self._show(index)
        self.ctx.map_selected = self.records[index]["_id"] if index is not None else None
        self.ctx.save_state()

    def select(self, key) -> bool:
        """Highlight one record, show its details and centre the map on it (used by the table)."""
        index = self.index_by_id.get(str(key))
        if index is None or self.source is None:
            return False
        rec = self.records[index]
        self._show(index)
        try:
            half = 700.0
            self.fig.x_range.start, self.fig.x_range.end = rec["_x"] - half, rec["_x"] + half
            self.fig.y_range.start, self.fig.y_range.end = rec["_y"] - half, rec["_y"] + half
            self.source.selected.indices = [index]
        except Exception as exc:
            logger.debug("Could not centre the map on %s: %s", key, exc)
        return True

    def _on_basemap(self, event) -> None:
        """Show the chosen basemap (and the road overlay on the satellite imagery only)."""
        try:
            for key, renderer in self.tiles.items():
                renderer.visible = (key == event.new)
            if self.road_renderer is not None:
                self.road_renderer.visible = (event.new == "satellite")
        except Exception as exc:
            logger.debug("Basemap switch failed: %s", exc)
        self.ctx.basemap = event.new
        self.ctx.save_state()

    def _report_clicked(self, _=None) -> None:
        if self._current is not None and self.on_report is not None:
            self.on_report(self._current)


def home_map(ctx: "AppContext", on_report: Optional[Callable[[Dict[str, Any]], None]] = None, height: int = 420):
    """The streetlight map shown on every role's home page (all streetlights; click one for details)."""
    try:
        return StreetlightMap(ctx, explorer_view(ctx, "all"), on_report=on_report, height=height).panel
    except Exception as exc:
        logger.exception("Home map failed: %s", exc)
        return alert("The map could not be drawn. The problem has been logged; open the Streetlight Map page or try again.", "danger")


# =====================================================================
# STREETLIGHT EXPLORER — map + list that the KPI cards open
# =====================================================================
# Every number comes from STREETLIGHT_REGISTRY, so the cards, the map and the table cannot disagree.

EXPLORER_FILTERS: Dict[str, Dict[str, Any]] = {
    "all": {"label": "All streetlights", "noun": "streetlights", "roles": ("ADMIN", "MUNICIPAL_STAFF", "COMMUNITY")},
    "operational": {"label": "Operational", "noun": "operational streetlights", "roles": ("ADMIN", "MUNICIPAL_STAFF", "COMMUNITY")},
    "attention": {"label": "Fault / offline", "noun": "faulty or offline streetlights", "roles": ("ADMIN", "MUNICIPAL_STAFF", "COMMUNITY")},
    "maintenance": {"label": "Maintenance", "noun": "streetlights under maintenance", "roles": ("ADMIN", "MUNICIPAL_STAFF")},
    "priority": {"label": "High-priority sites", "noun": "high-priority candidate sites", "roles": ("ADMIN", "MUNICIPAL_STAFF")},
    "recommended": {"label": "Recommended new lights", "noun": "recommended new solar lights", "roles": ("ADMIN", "MUNICIPAL_STAFF")},
}
EXPLORER_COLUMN_LABELS = {
    "asset_id": "Asset ID", "status_label": "Status", "status_raw": "Recorded as", "suburb": "Suburb", "light_type": "Light type",
    "latitude": "Latitude", "longitude": "Longitude", "utm_easting": "UTM easting (m)", "utm_northing": "UTM northing (m)",
    "fault_description": "Fault description", "install_date": "Installed", "last_inspection": "Last inspection",
    "last_maintenance": "Last maintenance", "maintenance_notes": "Maintenance notes", "service_status": "Service status",
    "rank": "Rank", "candidate_id": "Candidate ID", "new_light_id": "New light ID", "road_name": "Road", "road_type": "Road type",
    "overall_priority": "Suitability (0–100)", "priority_class": "Priority class", "safety_score": "Demand & safety score",
    "technical_score": "Technical score", "main_driver": "Main driver", "selection_status": "Selection status",
    "estimated_cost": "Estimated cost (USD)", "implementation_phase": "Phase", "existing_light_dist_m": "Nearest existing light (m)",
    "population": "Population within 250 m",
}
SITE_EXPLORER_COLUMNS = ["rank", "candidate_id", "new_light_id", "road_name", "latitude", "longitude", "utm_easting", "utm_northing",
                         "overall_priority", "priority_class", "main_driver", "selection_status", "estimated_cost", "implementation_phase"]
_EXPLORER_CORE_COLUMNS = {"asset_id", "status_label", "latitude", "longitude", "utm_easting", "utm_northing"}


def allowed_explorer_filters(role: Optional[str]) -> List[str]:
    """Which filters a role may use. Community users get the public status filters only;
    prioritisation output and maintenance workload are internal municipal analytics."""
    return [k for k, v in EXPLORER_FILTERS.items() if role in v["roles"]]


def filter_registry(reg: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    """Rows matching a status filter, on the normalised status class, so capitalisation, whitespace
    and supported wording variants can neither leak operational lights into the fault list nor
    drop genuine faults from it."""
    if reg is None or reg.empty:
        return reg
    classes = {"operational": STATUS_OPERATIONAL, "attention": STATUS_FAULT_OFFLINE, "maintenance": STATUS_MAINTENANCE}
    return reg[reg["status_class"] == classes[filter_key]] if filter_key in classes else reg


def explorer_view(ctx: "AppContext", filter_key: str, search: str = "") -> Dict[str, Any]:
    """Everything the explorer page needs for one filter, computed without touching any widget (so it
    can be tested directly). ``table`` is the display/CSV version, ``export`` keeps raw column names
    for the GeoJSON export, ``map_frame`` feeds the map. Records without valid coordinates are listed
    with blank coordinates and counted — never given substitute coordinates and never silently dropped."""
    role = ctx.session.role
    community = role == "COMMUNITY"
    if filter_key not in allowed_explorer_filters(role):
        filter_key = "all"
    q = (search or "").strip()
    meta = EXPLORER_FILTERS[filter_key]
    view: Dict[str, Any] = {
        "key": filter_key, "label": meta["label"], "noun": meta["noun"], "role": role, "kind": "streetlights",
        "summary": streetlight_summary(), "search": q, "messages": [], "table": pd.DataFrame(), "export": pd.DataFrame(),
        "map_frame": pd.DataFrame(), "key_col": "asset_id", "total_matching": 0, "shown_count": 0, "diag": None,
    }
    if filter_key in ("priority", "recommended"):
        view["kind"] = "sites"
        view["key_col"] = "candidate_id"
        scored = ctx.scored()
        diag = priority_diagnostics(scored)
        view["diag"] = diag
        if filter_key == "priority":
            sites = scored[scored["priority_class"].isin(HIGH_PRIORITY_CLASSES)] if not scored.empty else scored
            frame = pd.DataFrame(sites[[c for c in SITE_EXPLORER_COLUMNS if c in sites.columns]]).round({"overall_priority": 1, "estimated_cost": 0}) if not sites.empty else pd.DataFrame(columns=SITE_EXPLORER_COLUMNS)
        else:
            frame = recommended_sites(scored)
        view["total_matching"] = int(len(frame))
        if view["total_matching"] == 0:
            if diag["n_candidates"] == 0:
                view["messages"].append(("warn", "<b>No candidate sites exist.</b> " + esc(diag["verdict"])))
            elif filter_key == "recommended":
                view["messages"].append(("warn", "<b>No site was recommended.</b> Check the budget against the unit cost, and the minimum-spacing setting, on the MCDA Site Selection page."))
            else:
                view["messages"].append(("warn", "No candidate site falls in the top two suitability classes."))
        if q and not frame.empty:
            mask = pd.Series(False, index=frame.index)
            for col in ("candidate_id", "road_name", "new_light_id"):
                if col in frame.columns:
                    mask |= frame[col].astype(str).str.contains(q, case=False, regex=False)
            frame = frame[mask]
            if frame.empty:
                view["messages"].append(("warn", f"No site in <b>{esc(meta['label'])}</b> matches “{esc(q)}”."))
            else:
                view["messages"].append(("ok", f"{len(frame)} site(s) match “{esc(q)}”."))
        view["export"] = frame.reset_index(drop=True)
        view["map_frame"] = view["export"].copy()
        view["table"] = view["export"].rename(columns=EXPLORER_COLUMN_LABELS)
        view["shown_count"] = int(len(frame))
        return view

    reg = STREETLIGHT_REGISTRY
    if reg.empty:
        view["messages"].append(("warn", "<b>No streetlight records are loaded.</b> Place a streetlight layer (GeoPackage, GeoJSON, Shapefile, Parquet or CSV with latitude/longitude columns) in the data folder."))
        return view
    matching = filter_registry(reg, filter_key)
    view["total_matching"] = int(len(matching))
    shown = matching
    if view["total_matching"] == 0:
        view["messages"].append(("warn", f"No streetlights are classified as <b>{esc(meta['label'])}</b> in the current data."))
    if q and not shown.empty:
        needle = q.lower()
        mask = shown["asset_id"].astype(str).str.lower().str.contains(needle, regex=False) | shown["suburb"].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
        shown = shown[mask]
        if shown.empty:
            elsewhere = reg[reg["asset_id"].astype(str).str.lower().str.contains(needle, regex=False) | reg["suburb"].fillna("").astype(str).str.lower().str.contains(needle, regex=False)]
            hint = f" {len(elsewhere)} match in “All streetlights”." if len(elsewhere) else ""
            view["messages"].append(("warn", f"No streetlight in <b>{esc(meta['label'])}</b> matches “{esc(q)}”.{hint}"))
        else:
            view["messages"].append(("ok", f"{len(shown)} streetlight(s) match “{esc(q)}”."))
    view["map_frame"] = shown.reset_index(drop=True)
    if community:
        wanted = ["asset_id", "status_label", "suburb", "light_type"]
    else:
        wanted = ["asset_id", "status_label"] + (["status_raw"] if view["summary"]["status_kind"] == "source" else []) + \
                 ["suburb", "light_type", "latitude", "longitude", "utm_easting", "utm_northing", "fault_description", "install_date",
                  "last_inspection", "last_maintenance", "maintenance_notes", "service_status"]
    keep = [c for c in wanted if c in shown.columns and (c in _EXPLORER_CORE_COLUMNS or shown[c].notna().any())]
    export = shown[keep].reset_index(drop=True)
    labels = dict(EXPLORER_COLUMN_LABELS)
    if view["summary"]["status_kind"] == "simulated":
        labels["fault_description"] = "Fault description (SIMULATED)"
    view["export"] = export
    view["table"] = export.rename(columns=labels)
    view["shown_count"] = int(len(export))
    if not community and not shown.empty:
        no_coord = int((~shown["coord_valid"]).sum())
        if no_coord:
            view["messages"].append(("info", f"{no_coord} listed record(s) have missing or invalid coordinates (their latitude / longitude / UTM cells are blank, and they cannot be drawn on the map)."))
    return view


def explorer_info_html(view: Dict[str, Any], role: str) -> str:
    """Active filter and record count, plus a warning only when the user's own filter / search
    returned nothing. Data-quality and method notes are on the admin 'System Notices' page."""
    html = (f"<div class='moc-card'><div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;'>"
            f"<span class='moc-pill pill-green'>FILTER · {esc(view['label']).upper()}</span>"
            f"<span style='font-size:15px;'><b style='font-size:22px;color:{COLOR_NAVY};'>{view['shown_count']}</b> {esc(view['noun'])} listed</span></div></div>")
    warns = [t for k, t in view["messages"] if k == "warn"]
    return html + "".join(f"<div class='moc-alert moc-alert-warn' style='margin-top:8px;'>{t}</div>" for t in warns)


def system_notices(ctx: "AppContext") -> List[Tuple[str, str]]:
    """Every data-quality, method and configuration note, for the admin 'System Notices' page."""
    items: List[Tuple[str, str]] = []
    s = streetlight_summary()
    if CORE_REAL_DATA:
        items.append(("ok", "<b>Datasets.</b> Core municipal datasets (streetlights, roads, population) were detected on disk. Any layer not detected falls back to a clearly-labelled demo layer."))
    else:
        items.append(("warn", "<b>DEMO / SIMULATED DATA.</b> One or more core datasets (streetlights, roads, population) were not found in the data folder, so generated demonstration layers are in use. "
                              "Do not present these figures as real Chinhoyi municipal evidence — see Data Management for the per-layer breakdown."))
    if DEMO_LAYERS:
        items.append(("warn", "<b>Demonstration layers in use:</b> " + ", ".join(sorted(DEMO_LAYERS)) + "."))
    kind = s["status_kind"]
    if kind == "simulated":
        items.append(("warn", "<b>Streetlight status is SIMULATED.</b> The layer has no recognisable status column, so statuses were generated randomly for demonstration; these counts are not real evidence. "
                              "Add a status column (values such as Operational / Faulty / Offline) to use real values."))
    elif kind == "unknown":
        items.append(("warn", "No status information is available, so no streetlight can be classified as operational or faulty."))
    else:
        items.append(("ok", f"Streetlight status is read from source column <b>{esc(s['status_column'])}</b>."))
    items.append(("info", f"<b>Inventory reconciliation:</b> {s['total']} records = {s['operational']} operational + {s['fault_offline']} fault/offline + {s['maintenance']} maintenance + "
                          f"{s['unclassified']} status not recognised. {s['mapped']} have valid coordinates; {s['unmapped']} do not."))
    if s["unrecognised_values"]:
        shown = ", ".join(f"“{esc(k)}” ×{v}" for k, v in list(s["unrecognised_values"].items())[:6])
        items.append(("warn", f"{s['unclassified']} record(s) have status values the system does not recognise ({shown}); they are listed as “status not recognised” rather than guessed."))
    have = ", ".join(ATTRIBUTE_LABELS[k] for k in STREETLIGHT_META["available_fields"]) or "none beyond ID, status and coordinates"
    lack = ", ".join(ATTRIBUTE_LABELS[k] for k in STREETLIGHT_META["missing_fields"]) or "none"
    note = f"<b>Attribute fields found:</b> {esc(have)}. <b>Not present (never shown):</b> {esc(lack)}. Asset IDs: {esc(STREETLIGHT_META['asset_id_source'])}."
    extras = []
    if s["duplicate_ids"]:
        extras.append(f"{s['duplicate_ids']} records share an asset ID")
    if s["duplicate_coords"]:
        extras.append(f"{s['duplicate_coords']} records share identical coordinates")
    if extras:
        note += " Data quality: " + "; ".join(extras) + "."
    items.append(("info", note))
    kind_a, text_a = ahp_banner_content(ctx)
    items.append((kind_a, text_a))
    try:
        diag = priority_diagnostics(ctx.scored())
        bullets = "".join(f"<li>{esc(f)}</li>" for f in diag["findings"])
        items.append(("info", f"<b>Prioritisation notes.</b> {esc(diag['verdict'])}<ul style='margin:6px 0 0 18px;padding:0;'>{bullets}</ul>"))
    except Exception as exc:
        logger.warning("Prioritisation notes unavailable: %s", exc)
    items.append(("info", "<b>Basemaps.</b> The map uses the public Google satellite-hybrid and OpenStreetMap tile servers, which need no API key. Google's tile endpoint is unofficial and its terms "
                          "restrict use outside Google's own products; OpenStreetMap's public servers are for light use under its tile usage policy. Move to a licensed or self-hosted basemap before a public municipal rollout. "
                          "Imagery dates are those of the provider."))
    return items


def explorer_geojson(view: Dict[str, Any]) -> bytes:
    """GeoJSON (WGS84) of the rows currently listed that have valid coordinates."""
    empty = b'{"type": "FeatureCollection", "features": []}'
    df = view.get("export")
    if df is None or df.empty or not {"latitude", "longitude"} <= set(df.columns):
        return empty
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = lat.notna() & lon.notna()
    if not ok.any():
        return empty
    d = df[ok].reset_index(drop=True)
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(lon[ok].astype(float).to_numpy(), lat[ok].astype(float).to_numpy()), crs=WGS84).to_json().encode("utf-8")


def build_streetlight_explorer(ctx: "AppContext"):
    """The page every streetlight KPI card opens: a filter bar, the interactive map and the list.
    Clicking a point on the map shows its details; clicking a row in the list highlights it on the
    map. Residents can report a problem with the selected light."""
    role = ctx.session.role
    community = role == "COMMUNITY"
    allowed = allowed_explorer_filters(role)
    if ctx.explorer_filter not in allowed:
        ctx.explorer_filter = "all"
    filter_group = pn.widgets.RadioButtonGroup(name="Filter", options={EXPLORER_FILTERS[k]["label"]: k for k in allowed},
                                               value=ctx.explorer_filter, button_type="primary", sizing_mode="stretch_width")
    search = pn.widgets.TextInput(name="Search", placeholder=("Streetlight ID or suburb — press Enter" if community else "Streetlight ID, suburb, candidate ID or road — press Enter"), value=ctx.explorer_search, sizing_mode="stretch_width")
    search_btn = pn.widgets.Button(name="Search", button_type="primary", width=90)
    reset_btn = pn.widgets.Button(name="Show all streetlights", button_type="success", width=190)
    back_btn = pn.widgets.Button(name="‹ Home" if community else "‹ Dashboard", button_type="light", width=140)
    info = pn.pane.HTML("", sizing_mode="stretch_width")
    map_col = pn.Column(sizing_mode="stretch_width")
    table_col = pn.Column(sizing_mode="stretch_width")
    state: Dict[str, Any] = {"view": None, "busy": False, "map": None}

    def _csv(*args):
        v = state["view"]
        return io.BytesIO(v["table"].to_csv(index=False).encode("utf-8") if v is not None else b"")

    def _geojson(*args):
        try:
            return io.BytesIO(explorer_geojson(state["view"]))
        except Exception as exc:
            logger.warning("Explorer GeoJSON export failed: %s", exc)
            return io.BytesIO(b'{"type": "FeatureCollection", "features": []}')

    dl_csv = pn.widgets.FileDownload(callback=_csv, filename="streetlights_filtered.csv", label="Download listed rows (CSV)", button_type="primary", width=250, visible=not community)
    dl_geo = pn.widgets.FileDownload(callback=_geojson, filename="streetlights_filtered.geojson", label="Download listed rows (GeoJSON)", button_type="primary", width=280, visible=not community)

    def go_report(rec: Dict[str, Any]) -> None:
        ctx.report_prefill = report_prefill_from(rec)
        ctx.navigate("submit_report")

    def on_row(event) -> None:
        view, m = state["view"], state["map"]
        if not event.new or view is None or m is None:
            return
        try:
            key = view["export"].iloc[int(event.new[0])][view["key_col"]]
        except Exception:
            return
        if not m.select(str(key)):
            try:
                pn.state.notifications.info("That record has no valid coordinates, so it cannot be shown on the map.", duration=3500)
            except Exception:
                pass

    def refresh(*_):
        if state["busy"]:
            return
        try:
            view = explorer_view(ctx, filter_group.value, search.value or "")
            state["view"] = view
            ctx.explorer_filter, ctx.explorer_search, ctx.explorer_selected = view["key"], (search.value or ""), True
            ctx.save_state()
            kind_name = "streetlights" if view["kind"] == "streetlights" else "sites"
            dl_csv.filename, dl_geo.filename = f"{kind_name}_{view['key']}.csv", f"{kind_name}_{view['key']}.geojson"
            info.object = explorer_info_html(view, role)
            m = StreetlightMap(ctx, view, on_report=go_report if community else None, height=470)
            state["map"] = m
            map_col[:] = [m.panel]
            table = make_table(view["table"], height=430, selectable=True, page_size=10)
            if isinstance(table, pn.widgets.Tabulator):
                table.param.watch(on_row, "selection")
            table_col[:] = [table]
        except Exception as exc:
            logger.exception("Streetlight explorer refresh failed: %s", exc)
            info.object = "<div class='moc-alert moc-alert-danger'>The map and list could not be updated. The problem has been logged; try another filter.</div>"

    def reset(_=None):
        state["busy"] = True
        filter_group.value, search.value = "all", ""
        state["busy"] = False
        ctx.map_selected = None
        refresh()

    filter_group.param.watch(refresh, "value")
    search.param.watch(refresh, "value")
    search_btn.on_click(refresh)
    reset_btn.on_click(reset)
    back_btn.on_click(lambda _: ctx.navigate("home"))
    refresh()
    title = "Streetlight map" if community else "Streetlight map & list"
    subtitle = ("Find streetlights in your area. Click a light on the map for its details, and report a problem with it." if community else
                "The map and lists behind the dashboard cards, with WGS84 and UTM coordinates for each record. Click a point for details, click a row to find it on the map, and download the listed rows for QGIS / ArcGIS.")
    return pn.Column(
        page_header(title, subtitle, role),
        pn.Row(back_btn, reset_btn, sizing_mode="stretch_width"),
        filter_group,
        pn.Row(search, search_btn, sizing_mode="stretch_width"),
        info,
        section_title("Map"), map_col,
        section_title("List"),
        pn.Row(dl_csv, dl_geo, sizing_mode="stretch_width"),
        table_col,
        sizing_mode="stretch_width")

# =====================================================================
# STAFF VIEWS (internal analytics — never shown to community users)
# =====================================================================

def build_staff_overview(ctx: AppContext):
    role = ctx.session.role
    scored = ctx.scored()
    report_df = get_reports()
    open_reports = int((~report_df["status"].isin(["Resolved", "Closed"])).sum()) if not report_df.empty else 0
    rec = recommended_sites(scored)
    a = ctx.ahp()
    spend = float(rec["estimated_cost"].sum()) if not rec.empty else 0.0
    top_cols = [c for c in ["new_light_id", "rank", "road_name", "latitude", "longitude", "overall_priority", "priority_class", "main_driver"] if c in rec.columns]
    method_line = (f"<b>Weighting method:</b> AHP · CR = {a['cr']:.3f} ({'consistent' if a['consistent'] else 'INCONSISTENT — provisional'})"
                   if ctx.scenario.value == AHP_METHOD else f"<b>Weighting method:</b> preset “{esc(ctx.scenario.value)}” (not AHP)")
    return pn.Column(
        page_header("Streetlight command centre", "Monitoring, maintenance and AHP-weighted MCDA selection of new solar streetlights.", role),
        data_mode_alert(),
        ahp_banner(ctx),
        pn.Spacer(height=6),
        streetlight_kpi_row(ctx),
        pn.Spacer(height=6),
        section_title("Streetlight map — click a light for its details"),
        home_map(ctx, height=400),
        pn.Spacer(height=6),
        pn.Spacer(height=10),
        section_title("Top recommended new solar streetlight sites"),
        df_table(rec.head(5)[top_cols] if top_cols else rec, height=260),
        sizing_mode="stretch_width")


def build_assets(ctx):
    status = DATA.maintenance.copy()
    assets = DATA.streetlights.copy() if DATA.streetlights is not None else gpd.GeoDataFrame()
    if not assets.empty and not status.empty:
        table = assets.drop(columns="geometry", errors="ignore").merge(status[["asset_id", "status", "fault_type", "assigned_to", "repair_date"]], on="asset_id", how="left")
    else:
        table = assets.drop(columns="geometry", errors="ignore")
    search = pn.widgets.TextInput(name="Search asset ID / status", placeholder="e.g. SL-CHN-0001 or Faulty", sizing_mode="stretch_width")
    table_pane = pn.Column(df_table(table), sizing_mode="stretch_width")

    def do_filter(_=None):
        q = (search.value or "").strip().lower()
        if not q or table.empty:
            table_pane[:] = [df_table(table)]
            return
        mask = table.apply(lambda row: q in " ".join(str(v).lower() for v in row.values), axis=1)
        table_pane[:] = [df_table(table[mask])]

    search.param.watch(do_filter, "value")
    return pn.Column(
        page_header("Streetlight asset registry", "Operational inventory for existing municipal streetlights.", ctx.session.role),
        data_mode_alert(),
        search, section_title("Asset list"), table_pane, sizing_mode="stretch_width")


def build_gaps(ctx):
    scored = ctx.scored()
    existing = DATA.streetlights
    cov = 0.0
    if existing is not None and DATA.population is not None and not existing.empty and not DATA.population.empty:
        try:
            pop = DATA.population.copy(); pop["geometry"] = pop.geometry.apply(representative_point); pop = pop[pop.geometry.notna()].to_crs(UTM)
            lights = existing.to_crs(UTM)
            union = unary_union([g.buffer(int(ctx.coverage_radius.value)) for g in lights.geometry])
            cov = round(100 * sum(g.within(union) for g in pop.geometry) / max(len(pop), 1), 1)
        except Exception as exc:
            logger.warning("Coverage calculation failed: %s", exc)
            cov = 0.0
    high_gap = int(scored["lighting_gap"].gt(scored["lighting_gap"].quantile(.75)).sum()) if not scored.empty else 0
    if scored.empty:
        gap_table = scored
    else:
        gap_cols = [c for c in ["candidate_id", "road_name", "latitude", "longitude", "existing_light_dist_m", "population", "overall_priority", "priority_class"] if c in scored.columns]
        gap_table = pd.DataFrame(scored.sort_values("existing_light_dist_m", ascending=False)[gap_cols].head(15)).round({"existing_light_dist_m": 0, "population": 0, "overall_priority": 1})
    return pn.Column(
        page_header("Lighting gaps & coverage", "Coverage is evaluated from existing streetlights using a configurable radius; the candidate sites furthest from any existing light are listed below.", ctx.session.role),
        data_mode_alert(),
        pn.Spacer(height=6),
        pn.Row(kpi("Estimated population coverage", f"{cov}%", f"Within {ctx.coverage_radius.value} m of an existing light", COLOR_TEAL), kpi("High-gap candidate sites", high_gap, "Upper quartile of lighting gap score", COLOR_RED), kpi("Coverage radius", f"{ctx.coverage_radius.value} m", "Adjust the control on the MCDA Site Selection page", COLOR_GREEN), sizing_mode="stretch_width"),
        pn.Spacer(height=10), section_title("15 candidate sites furthest from an existing streetlight"), df_table(gap_table, height=440), sizing_mode="stretch_width")


def build_prioritisation(ctx):
    scored = ctx.scored()
    cols = ["rank", "candidate_id", "new_light_id", "road_name", "latitude", "longitude", "population", "lighting_gap", "overall_priority", "priority_class", "main_driver", "selection_status", "estimated_cost", "implementation_phase"]
    shown = pd.DataFrame(scored[[c for c in cols if c in scored.columns]].head(200)).round({"population": 0, "lighting_gap": 3, "overall_priority": 1, "estimated_cost": 0}) if not scored.empty else scored
    top_priority = f"{float(scored.iloc[0]['overall_priority']):.1f}/100" if not scored.empty else "—"
    return pn.Column(
        page_header("Site prioritisation", "Candidate streetlight locations ranked by the AHP-weighted suitability index (safety, service, technical, accessibility and environmental criteria).", ctx.session.role),
        data_mode_alert(),
        ahp_banner(ctx),
        pn.Spacer(height=6),
        pn.Row(kpi("Weighting method", ctx.scenario.value, "Change it on the MCDA Site Selection page", COLOR_NAVY), kpi("Candidates", len(scored), "Generated along the road network", COLOR_GREEN), kpi("Top suitability", top_priority, "Highest AHP-weighted score", COLOR_RED), sizing_mode="stretch_width"),
        pn.Spacer(height=10), df_table(shown, height=500), sizing_mode="stretch_width")


# ---------------------------------------------------------------------
# OBJECTIVE 2 — AHP criteria weights page
# ---------------------------------------------------------------------

def ahp_matrix_html(matrix: np.ndarray, analysis: Dict[str, Any]) -> str:
    """The pairwise matrix as a colour-coded table with the derived weight and rank of each row."""
    n = len(CRITERIA)
    w = analysis["weights"]
    ranks = (-w).argsort().argsort() + 1
    th = "padding:5px 7px;border:1px solid #DFE7EE;background:#F3F7FA;color:#1B3A6B;font-weight:900;text-align:center;font-size:10.5px;"
    td = "padding:5px 7px;border:1px solid #DFE7EE;text-align:center;font-size:11px;"
    head = (f"<tr><th style='{th}text-align:left;'>Criterion (row vs column)</th>"
            + "".join(f"<th style='{th}' title='{esc(FEATURE_LABELS[k])}'>C{j + 1}</th>" for j, k in enumerate(CRITERIA))
            + f"<th style='{th}'>Weight</th><th style='{th}'>Rank</th></tr>")
    body = []
    for i, k in enumerate(CRITERIA):
        cells = []
        for j in range(n):
            v = float(matrix[i, j])
            if i == j:
                bg = "#EEF2F6"
            elif v > 1:
                bg = f"rgba(11,107,58,{0.12 + 0.5 * min(math.log(v) / math.log(9), 1):.2f})"
            elif v < 1:
                bg = f"rgba(239,143,0,{0.12 + 0.5 * min(math.log(1 / v) / math.log(9), 1):.2f})"
            else:
                bg = "#FFFFFF"
            cells.append(f"<td style='{td}background:{bg};'>{format_saaty(v)}</td>")
        body.append(
            f"<tr><td style='{td}text-align:left;white-space:nowrap;font-weight:800;'>C{i + 1} · {esc(FEATURE_LABELS[k])}</td>{''.join(cells)}"
            f"<td style='{td}font-weight:900;color:#1B3A6B;'>{100 * w[i]:.1f}%</td><td style='{td}'>{int(ranks[i])}</td></tr>")
    legend = ("<div style='font-size:10px;color:#64748B;margin-top:6px;'>Read each cell as “row criterion relative to column criterion”. "
              "<span style='background:rgba(11,107,58,.4);padding:1px 6px;border-radius:4px;'>green</span> = the row matters more · "
              "<span style='background:rgba(239,143,0,.4);padding:1px 6px;border-radius:4px;'>orange</span> = the column matters more.</div>")
    return (f"<div style='overflow-x:auto;background:#fff;border:1px solid #DFE7EE;border-radius:12px;padding:10px;'>"
            f"<table style='border-collapse:collapse;min-width:100%;'>{head}{''.join(body)}</table>{legend}</div>")


def build_ahp(ctx: AppContext):
    role = ctx.session.role
    labels = {FEATURE_LABELS[k]: k for k in CRITERIA}
    options = saaty_options()
    user = ctx.session.user or {}
    data_status = criteria_data_status(ctx.scored())   # independent of the weights, so computed once per visit

    sel_a = pn.widgets.Select(name="Criterion A", options=labels, value=CRITERIA[0], sizing_mode="stretch_width")
    sel_b = pn.widgets.Select(name="Criterion B", options=labels, value=CRITERIA[1], sizing_mode="stretch_width")
    sel_v = pn.widgets.Select(name="Judgement — how important is A relative to B?", options=options, value=1.0, sizing_mode="stretch_width")
    apply_btn = pn.widgets.Button(name="Apply judgement", button_type="success", width=170)
    reset_btn = pn.widgets.Button(name="Reset to starting judgements", button_type="warning", width=240)
    upload = pn.widgets.FileInput(accept=".csv", multiple=False)
    download = pn.widgets.FileDownload(callback=lambda *args: io.BytesIO(matrix_to_csv(ctx.ahp_matrix).encode("utf-8")),
                                       filename="ahp_pairwise_matrix.csv", label="Download matrix (CSV)", button_type="primary", width=230)
    msg = pn.pane.HTML("", sizing_mode="stretch_width")
    banner = pn.pane.HTML("", sizing_mode="stretch_width")
    matrix_pane = pn.pane.HTML("", sizing_mode="stretch_width")
    weights_col = pn.Column(sizing_mode="stretch_width")
    hints_pane = pn.pane.HTML("", sizing_mode="stretch_width")
    table_col = pn.Column(sizing_mode="stretch_width")

    def refresh():
        a = ctx.ahp()
        kind, text = ahp_banner_content(ctx)
        banner.object = f"<div class='moc-alert moc-alert-{kind}'>{text}</div>"
        matrix_pane.object = ahp_matrix_html(ctx.ahp_matrix, a)
        w = a["weights"]
        order = np.argsort(-w)
        bars = {FEATURE_LABELS[CRITERIA[i]]: round(100 * float(w[i]), 1) for i in order}
        effective = sum(float(w[i]) for i, k in enumerate(CRITERIA) if not data_status[k].startswith(("Neutral", "No candidates")))
        verdict = (f"<span style='color:#166534;font-weight:900;'>Acceptable</span> — CR is below {AHP_CR_THRESHOLD:.2f}." if a["consistent"]
                   else f"<span style='color:#991B1B;font-weight:900;'>Not acceptable</span> — CR is {AHP_CR_THRESHOLD:.2f} or more; revise the judgements.")
        stats = (f"<b>n</b> (criteria) = {a['n']}<br><b>λmax</b> = {a['lambda_max']:.4f}<br><b>CI</b> = (λmax − n) / (n − 1) = {a['ci']:.4f}<br>"
                 f"<b>RI</b> (Saaty, n = {a['n']}) = {a['ri']:.2f}<br><b>CR</b> = CI / RI = <b>{a['cr']:.4f}</b><br>{verdict}<br><br>"
                 f"<b>Cross-check:</b> the geometric-mean method differs from the eigenvector weights by at most "
                 f"{100 * float(np.abs(a['geometric_mean_weights'] - w).max()):.2f} percentage points.<br><br>"
                 f"<b>Weight on criteria that can change the ranking:</b> {100 * effective:.1f}%. The rest sits on criteria that are neutral "
                 f"(no data or no variation) and so has no effect — see section 4.")
        weights_col[:] = [pn.Row(html_bar_chart("AHP criteria weights (%)", bars, COLOR_GREEN), card("Consistency check", stats), sizing_mode="stretch_width")]
        if a["consistent"]:
            hints_pane.object = ""
        else:
            items = "".join(f"<li><b>{esc(h['a'])}</b> vs <b>{esc(h['b'])}</b>: you entered {format_saaty(h['current'])}; "
                            f"your other judgements imply about {format_saaty(h['suggested'])}.</li>" for h in ahp_inconsistency_hints(ctx.ahp_matrix, 3))
            hints_pane.object = (f"<div class='moc-alert moc-alert-warn'><b>Where to start revising.</b> The comparisons that depart most from the "
                                 f"weights implied by the rest of the matrix (a guide, not a rule):<ul style='margin:6px 0 0 18px;padding:0;'>{items}</ul></div>")
        rows = []
        for rank_no, i in enumerate(order, start=1):
            k = CRITERIA[i]
            rows.append({"Rank": rank_no, "Criterion": FEATURE_LABELS[k], "Group": CRITERIA_GROUPS[k], "AHP weight (%)": round(100 * float(w[i]), 1),
                         "Data status": data_status[k], "How it is measured": CRITERIA_MEASURE[k]})
        table_col[:] = [df_table(pd.DataFrame(rows), height=470)]

    def sync_value(*_):
        i, j = CRITERIA.index(sel_a.value), CRITERIA.index(sel_b.value)
        sel_v.value = nearest_saaty(float(ctx.ahp_matrix[i, j]))

    def on_apply(_=None):
        if sel_a.value == sel_b.value:
            msg.object = "<div class='moc-alert moc-alert-warn'>Choose two different criteria to compare.</div>"
            return
        i, j = CRITERIA.index(sel_a.value), CRITERIA.index(sel_b.value)
        v = float(sel_v.value)
        m = ctx.ahp_matrix.copy()
        m[i, j], m[j, i] = v, 1.0 / v
        ctx.set_ahp_matrix(m, "edited in this session")
        audit(user.get("username"), role, "ahp_judgement", f"{sel_a.value} vs {sel_b.value} = {format_saaty(v)}")
        msg.object = (f"<div class='moc-alert moc-alert-ok'>Recorded: <b>{esc(FEATURE_LABELS[sel_a.value])}</b> relative to "
                      f"<b>{esc(FEATURE_LABELS[sel_b.value])}</b> = {format_saaty(v)} (and the reciprocal {format_saaty(1.0 / v)} the other way). Saved with your session.</div>")
        refresh()

    def on_reset(_=None):
        ctx.set_ahp_matrix(AHP_INITIAL_MATRIX, AHP_INITIAL_SOURCE)
        audit(user.get("username"), role, "ahp_reset", "")
        msg.object = "<div class='moc-alert moc-alert-info'>Judgements reset to the starting matrix.</div>"
        sync_value()
        refresh()

    def on_upload(event):
        raw = event.new
        if not raw:
            return
        matrix, text = parse_pairwise_csv(raw)
        if matrix is None:
            msg.object = f"<div class='moc-alert moc-alert-danger'>{esc(text)}</div>"
            return
        ctx.set_ahp_matrix(matrix, f"uploaded file {upload.filename or ''}".strip())
        audit(user.get("username"), role, "ahp_upload", upload.filename or "")
        msg.object = "<div class='moc-alert moc-alert-ok'>Pairwise matrix uploaded; weights and consistency recalculated.</div>"
        sync_value()
        refresh()

    sel_a.param.watch(sync_value, "value")
    sel_b.param.watch(sync_value, "value")
    apply_btn.on_click(on_apply)
    reset_btn.on_click(on_reset)
    upload.param.watch(on_upload, "value")
    sync_value()
    refresh()

    guide = card(
        "How the AHP step works",
        "Compare the criteria two at a time on Saaty's scale: <b>1</b> = equally important, <b>3</b> moderately, <b>5</b> strongly, <b>7</b> very strongly, "
        "<b>9</b> extremely (2, 4, 6, 8 are in-between values). Entering A over B as 3 automatically stores B over A as 1/3. "
        "The weights are the principal eigenvector of the matrix. The consistency ratio CR = CI / RI tells you whether the judgements contradict each other; "
        f"CR below {AHP_CR_THRESHOLD:.2f} is acceptable.<br><br>"
        "<b>Criteria 1–7</b> describe demand and safety need; <b>criteria 8–11</b> describe technical and environmental feasibility. "
        "Your judgements are saved with your session and restored when you sign in again. To make a matrix the starting point for everyone, download it, edit it in a spreadsheet, "
        "and save it as <b>data/ahp_pairwise.csv</b>. The weights are applied on the MCDA Site Selection page when the weighting method is AHP.")
    return pn.Column(
        page_header("AHP criteria weights", "Derive the weights of the siting criteria from pairwise comparisons and check that the judgements are consistent (Objective 2).", role),
        guide, pn.Spacer(height=6),
        section_title("1. Enter or revise a pairwise judgement"),
        pn.Row(sel_a, sel_b, sel_v, sizing_mode="stretch_width"),
        pn.Row(apply_btn, reset_btn, sizing_mode="stretch_width"),
        pn.Row(pn.Column(pn.pane.Markdown("**Upload a pairwise matrix (CSV)**"), upload), download, sizing_mode="stretch_width"),
        msg,
        section_title("2. Pairwise comparison matrix"), matrix_pane,
        section_title("3. Derived weights and consistency"), weights_col, hints_pane,
        section_title("4. Criteria, measurement and data status"), table_col,
        sizing_mode="stretch_width")


# ---------------------------------------------------------------------
# OBJECTIVE 2 — MCDA site selection page
# ---------------------------------------------------------------------

def build_decision_engine(ctx: AppContext):
    role = ctx.session.role
    run = pn.widgets.Button(name="Run AHP-weighted site selection", button_type="success", width=270)
    banner = pn.pane.HTML("", sizing_mode="stretch_width")
    kpi_row = pn.Row(sizing_mode="stretch_width")
    status_pane = pn.pane.HTML("", sizing_mode="stretch_width")
    sites_col = pn.Column(sizing_mode="stretch_width")
    map_col = pn.Column(sizing_mode="stretch_width")
    weights_col = pn.Column(sizing_mode="stretch_width")
    sens_col = pn.Column(sizing_mode="stretch_width")
    diag_pane = pn.pane.HTML("", sizing_mode="stretch_width")
    dl_csv = pn.widgets.FileDownload(callback=lambda *args: io.BytesIO(recommended_sites(ctx.scored()).to_csv(index=False).encode("utf-8")),
                                     filename="recommended_new_solar_lights.csv", label="Download recommended sites (CSV)", button_type="primary", width=300)
    dl_geo = pn.widgets.FileDownload(callback=lambda *args: io.BytesIO(recommended_geojson(ctx.scored())),
                                     filename="recommended_new_solar_lights.geojson", label="Download recommended sites (GeoJSON)", button_type="primary", width=320)

    def update(_=None):
        ctx.invalidate()
        scored = ctx.scored()
        kind, text = ahp_banner_content(ctx)
        banner.object = f"<div class='moc-alert moc-alert-{kind}'>{text}</div>"
        rec = recommended_sites(scored)
        n_rec = len(rec)
        spend = float(rec["estimated_cost"].sum()) if n_rec else 0.0
        budget = _widget_float(ctx.budget, 0.0)
        top_score = f"{float(scored.iloc[0]['overall_priority']):.1f}" if not scored.empty else "—"
        kpi_row[:] = [
            kpi("Candidate sites assessed", len(scored), f"Along roads, ≥ {CANDIDATE_SPACING_M * 0.55:.0f} m from an existing light", COLOR_NAVY),
            kpi("Recommended new solar lights", n_rec, f"≥ {int(_widget_float(ctx.min_spacing, 0))} m apart, within budget", COLOR_GREEN),
            kpi("Estimated spend", f"${spend:,.0f}", f"Unit cost + {_widget_float(ctx.contingency, 0):g}% contingency", COLOR_ORANGE),
            kpi("Budget remaining", f"${max(budget - spend, 0):,.0f}", f"Top suitability score {top_score}/100", COLOR_GRAY),
        ]
        if scored.empty:
            status_pane.object = ""
        else:
            counts = scored["selection_status"].value_counts()
            items = "".join(f"<li>{esc(k)}: <b>{int(v)}</b></li>" for k, v in counts.items())
            status_pane.object = (f"<div class='moc-card'><div class='moc-card-title'>What happened to the {len(scored)} candidate sites</div>"
                                  f"<div class='moc-card-subtitle'><ul style='margin:4px 0 0 18px;padding:0;'>{items}</ul></div></div>")
        if n_rec:
            sites_col[:] = [df_table(rec, height=480), pn.Row(dl_csv, dl_geo, sizing_mode="stretch_width")]
            try:
                map_col[:] = [StreetlightMap(ctx, explorer_view(ctx, "recommended"), height=420).panel]
            except Exception as exc:
                logger.exception("Recommended-sites map failed: %s", exc)
                map_col[:] = []
        else:
            sites_col[:] = [alert("No site was recommended. Check the budget against the unit cost, the minimum-spacing setting, and the candidate count below.", "warn")]
            map_col[:] = []
        diag = priority_diagnostics(scored)
        bullets = "".join(f"<li>{esc(f)}</li>" for f in diag["findings"])
        diag_pane.object = (f"<div class='moc-alert moc-alert-info'><b>Reading these results.</b> {esc(diag['verdict'])}"
                            f"<ul style='margin:6px 0 0 18px;padding:0;'>{bullets}</ul></div>")
        w = ctx.weights()
        status = criteria_data_status(scored)
        wdf = pd.DataFrame([{"Criterion": FEATURE_LABELS[k], "Group": CRITERIA_GROUPS[k], "Weight (%)": round(100 * float(w.get(k, 0)) / max(sum(w.values()), 1e-9), 1),
                             "Data status": status[k]} for k in sorted(CRITERIA, key=lambda c: -float(w.get(c, 0)))])
        weights_col[:] = [df_table(wdf, height=420)]
        top_n = n_rec if n_rec > 0 else min(10, len(scored))
        if scored.empty or top_n == 0:
            sens_col[:] = [pn.pane.Markdown("_Sensitivity analysis needs at least one candidate site._")]
        else:
            alternatives = {name: {**DEFAULT_WEIGHTS, **ws} for name, ws in SCENARIOS.items() if name != ctx.scenario.value}
            if ctx.scenario.value != AHP_METHOD:
                alternatives[AHP_METHOD] = ctx.ahp()["weights_by_key"]
            sens_col[:] = [
                pn.pane.HTML(f"<div class='moc-card-subtitle'>How many of the top {top_n} sites (by suitability score) stay in the top {top_n} when the weights change. "
                             f"Values close to 100% mean the recommendation does not depend on the exact weights.</div>"),
                section_title("Alternative weight sets"),
                df_table(preset_comparison(scored, w, alternatives, top_n), height=260),
                section_title("Each criterion's weight raised and lowered by 25%"),
                df_table(oat_sensitivity(scored, w, top_n), height=430),
            ]

    run.on_click(update)
    update()
    controls = pn.Column(
        pn.Row(ctx.scenario, ctx.max_candidates, ctx.min_spacing, ctx.coverage_radius, sizing_mode="stretch_width"),
        pn.Row(ctx.budget, ctx.unit_cost, ctx.contingency, sizing_mode="stretch_width"),
        ctx.hard_constraints, run, sizing_mode="stretch_width")
    return pn.Column(
        page_header("MCDA site selection — new solar streetlights", "Score every candidate site with the AHP-weighted suitability index, then choose the new lights down the ranking within the budget, minimum spacing and environmental constraints (Objective 2). Settings are saved with your session.", role),
        section_title("Settings"), controls, pn.Spacer(height=6),
        kpi_row, pn.Spacer(height=6),
        section_title("Recommended new solar streetlight locations"), map_col, sites_col, pn.Spacer(height=6),
        status_pane,
        pn.Spacer(height=6),
        section_title("Criterion weights in use"), weights_col,
        section_title("Sensitivity of the recommendation to the weights"), sens_col,
        sizing_mode="stretch_width")


def build_what_if(ctx):
    new_count = pn.widgets.IntSlider(name="Illustrative new lights", start=10, end=200, step=10, value=25)
    run = pn.widgets.Button(name="Run scenario", button_type="primary")
    output = pn.pane.HTML("", sizing_mode="stretch_width")

    def calc(_=None):
        scored = ctx.scored(); n = min(int(new_count.value), len(scored)); top = scored.head(n)
        current = len(DATA.streetlights) if DATA.streetlights is not None else 0
        est_cost = n * _widget_float(ctx.unit_cost, 0) * (1 + _widget_float(ctx.contingency, 0) / 100)
        high_selected = int(top.priority_class.isin(["VERY HIGH", "HIGH"]).sum()) if not top.empty else 0
        output.object = (
            f"<div class='moc-card'><div class='moc-card-title'>Scenario result</div><div class='moc-card-subtitle'>"
            f"Illustrative expansion: <b>{current}</b> existing lights + <b>{n}</b> proposed sites = <b>{current + n}</b> lights. "
            f"Estimated installation cost at current assumptions: <b>${est_cost:,.0f}</b>. "
            f"High/very-high sites among the top {n} ranked: <b>{high_selected}</b>. "
            f"(This takes the top-ranked sites as they are; the MCDA Site Selection page also applies minimum spacing and constraints.)</div></div>"
        )

    run.on_click(calc); calc()
    return pn.Column(page_header("What-if planning", "Test indicative expansion sizes using the current decision model and costing assumptions.", ctx.session.role), pn.Row(new_count, ctx.unit_cost, ctx.contingency, run, sizing_mode="stretch_width"), output, sizing_mode="stretch_width")


def build_budget(ctx):
    scored = ctx.scored(); selected = scored[scored.funded_phase_1].copy() if not scored.empty else scored; spend = float(selected.estimated_cost.sum()) if not selected.empty else 0.0
    cols = ["rank", "candidate_id", "new_light_id", "overall_priority", "estimated_cost", "selection_status", "implementation_phase"]
    return pn.Column(
        page_header("Budget & implementation phasing", "Translate spatial priority into a budget-aware first phase, while retaining the ranked pipeline for future funding.", ctx.session.role),
        data_mode_alert(),
        pn.Spacer(height=6),
        pn.Row(kpi("Available budget", f"${_widget_float(ctx.budget, 0):,.0f}", "Current planning envelope", COLOR_NAVY), kpi("Recommended new lights", len(selected), "Within available budget", COLOR_GREEN), kpi("Estimated spend", f"${spend:,.0f}", "Includes contingency", COLOR_ORANGE), kpi("Future pipeline", max(len(scored) - len(selected), 0), "Not recommended in this round", COLOR_GRAY), sizing_mode="stretch_width"),
        pn.Spacer(height=10), df_table(scored[[c for c in cols if c in scored.columns]].head(200) if not scored.empty else scored, height=480), sizing_mode="stretch_width")


def build_maintenance(ctx):
    orders = get_work_orders()
    existing = DATA.maintenance.copy()
    if orders.empty:
        if not existing.empty and "status" in existing.columns:
            orders = existing.copy()
            orders["status"] = orders["status"].replace({"Operational": "Closed", "Faulty": "Open", "Maintenance": "In Progress", "Offline": "Open"})
            orders["work_order_id"] = orders["work_order_id"].astype(str)
        else:
            orders = pd.DataFrame(columns=["work_order_id", "asset_id", "status", "fault_type", "assigned_to", "reported_at", "notes"])
    faulty_n = int((existing.status == "Faulty").sum()) if not existing.empty and "status" in existing.columns else 0
    maint_n = int((existing.status == "Maintenance").sum()) if not existing.empty and "status" in existing.columns else 0
    offline_n = int((existing.status == "Offline").sum()) if not existing.empty and "status" in existing.columns else 0
    return pn.Column(
        page_header("Maintenance operations", "Monitor fault status, work orders and the service backlog for the streetlight asset registry.", ctx.session.role),
        pn.Row(kpi("Assets", len(existing), "Registered streetlights", COLOR_GREEN), kpi("Faulty", faulty_n, "Fault cases", COLOR_RED), kpi("Maintenance", maint_n, "Assets in service", COLOR_ORANGE), kpi("Offline", offline_n, "Needs attention", COLOR_GRAY), sizing_mode="stretch_width"),
        pn.Spacer(height=10), df_table(orders, height=460), sizing_mode="stretch_width")


def build_reports_staff(ctx):
    reports = get_reports()
    return pn.Column(page_header("Community streetlight reports", "Review public service reports (each names the streetlight concerned when the resident picked one on the map), verify issues and maintain an accountable municipal response trail.", ctx.session.role), df_table(reports, height=520), sizing_mode="stretch_width")


def build_analytics(ctx):
    scored = ctx.scored(); reports = get_reports()
    summary = pd.DataFrame({
        "Metric": ["Existing lights", "Candidates", "Recommended new lights", "Very high priority", "High priority", "Open community reports"],
        "Value": [
            len(DATA.streetlights) if DATA.streetlights is not None else 0,
            len(scored),
            int(scored["funded_phase_1"].sum()) if not scored.empty else 0,
            int((scored.priority_class == "VERY HIGH").sum()) if not scored.empty else 0,
            int((scored.priority_class == "HIGH").sum()) if not scored.empty else 0,
            0 if reports.empty else int((~reports.status.isin(["Resolved", "Closed"])).sum()),
        ],
    })
    status_counts: Dict[str, float] = {}
    if not DATA.maintenance.empty and "status" in DATA.maintenance.columns:
        status_counts = DATA.maintenance["status"].value_counts().to_dict()
    priority_counts: Dict[str, float] = {}
    if not scored.empty:
        priority_counts = scored["priority_class"].value_counts().reindex(["VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"]).dropna().to_dict()
    report_status_counts: Dict[str, float] = {}
    if not reports.empty:
        report_status_counts = reports["status"].value_counts().to_dict()
    return pn.Column(
        page_header("Operational & spatial analytics", "Compact decision-support indicators for management review and dissertation evidence.", ctx.session.role),
        data_mode_alert(),
        pn.Spacer(height=6),
        df_table(summary, height=270),
        pn.Spacer(height=10),
        pn.Row(
            html_bar_chart("Streetlight status distribution", status_counts, COLOR_TEAL),
            html_bar_chart("Candidate priority-class distribution (relative classes)", priority_counts, COLOR_ORANGE),
            html_bar_chart("Community report status", report_status_counts, COLOR_SKY),
            sizing_mode="stretch_width",
        ),
        pn.Spacer(height=10),
        card("Interpretation", "Use the analytics together with the AHP/MCDA results and the municipal verification process. The model does not replace engineering inspection, community engagement, or formal municipal approval."),
        sizing_mode="stretch_width")


def build_export(ctx):
    scored = ctx.scored()
    tabular = scored.drop(columns="geometry", errors="ignore")
    csv_bytes = tabular.to_csv(index=False).encode("utf-8")
    csv_path = EXPORT_DIR / "chinhoyi_solar_streetlight_prioritisation.csv"
    try:
        csv_path.write_bytes(csv_bytes)
    except Exception as exc:
        logger.warning("Could not write CSV export to disk: %s", exc)

    geojson_bytes = b"{}"
    geojson_ok = False
    try:
        if not scored.empty:
            geojson_bytes = scored.to_json().encode("utf-8")
            geojson_ok = True
    except Exception as exc:
        logger.warning("GeoJSON export failed: %s", exc)

    gpkg_ok = False
    gpkg_bytes = b""
    try:
        if not scored.empty:
            tmp_path = EXPORT_DIR / "chinhoyi_solar_streetlight_prioritisation.gpkg"
            scored.to_file(tmp_path, driver="GPKG")
            gpkg_bytes = tmp_path.read_bytes()
            gpkg_ok = True
    except Exception as exc:
        logger.warning("GeoPackage export failed (driver may be unavailable in this environment): %s", exc)

    rec = recommended_sites(scored)
    rec_downloads = [pn.widgets.FileDownload(callback=_download_callback(rec.to_csv(index=False).encode("utf-8")), filename="recommended_new_solar_lights.csv", label="Recommended new lights (CSV)", button_type="success")]
    try:
        rec_downloads.append(pn.widgets.FileDownload(callback=_download_callback(recommended_geojson(scored)), filename="recommended_new_solar_lights.geojson", label="Recommended new lights (GeoJSON)", button_type="success"))
    except Exception as exc:
        logger.warning("Recommended-sites GeoJSON export failed: %s", exc)

    downloads = [pn.widgets.FileDownload(callback=_download_callback(csv_bytes), filename=csv_path.name, label="All candidates (CSV)", button_type="primary")]
    if geojson_ok:
        downloads.append(pn.widgets.FileDownload(callback=_download_callback(geojson_bytes), filename="chinhoyi_solar_streetlight_prioritisation.geojson", label="All candidates (GeoJSON)", button_type="primary"))
    else:
        downloads.append(pn.widgets.StaticText(value="GeoJSON export unavailable for the current (empty) candidate set."))
    if gpkg_ok:
        downloads.append(pn.widgets.FileDownload(callback=_download_callback(gpkg_bytes), filename="chinhoyi_solar_streetlight_prioritisation.gpkg", label="All candidates (GeoPackage)", button_type="primary"))

    status = alert(
        "GeoJSON and GeoPackage files open directly in QGIS / ArcGIS (coordinates are WGS84 latitude/longitude, with UTM zone 36S easting/northing in the attribute table). "
        "GeoPackage needs a working Fiona/Pyogrio installation on the server; if unavailable, CSV and GeoJSON remain fully supported.",
        "ok" if geojson_ok else "warn",
    )
    return pn.Column(
        page_header("Export centre", "Download the recommended new solar lights, or the full ranked candidate dataset, for GIS, reporting or dissertation analysis.", ctx.session.role),
        data_mode_alert(),
        section_title("Recommended new solar lights"),
        pn.Row(*rec_downloads, sizing_mode="stretch_width"),
        section_title("Full ranked candidate dataset"),
        pn.Row(*downloads, sizing_mode="stretch_width"),
        sizing_mode="stretch_width")

# =====================================================================
# COMMUNITY VIEWS
# =====================================================================

def build_report_form(ctx: "AppContext"):
    """The problem-report form, shared by the resident home page and the Report a Problem page.
    Returns (form, set_target): ``set_target(rec)`` fills in the streetlight chosen on the map.
    A reference number is issued on submit, and the resident can track the report under My Reports."""
    cat = pn.widgets.Select(name="Problem category", options=REPORT_CATEGORIES)
    priority = pn.widgets.Select(name="How urgent is it?", options=["Low", "Medium", "High"], value="Medium")
    asset = pn.widgets.TextInput(name="Streetlight ID (optional — click a light on the map to fill this in)", placeholder="e.g. SL-CHN-0007")
    lat = pn.widgets.FloatInput(name="Latitude", value=CHINHOYI_LAT, step=.00001)
    lon = pn.widgets.FloatInput(name="Longitude", value=CHINHOYI_LON, step=.00001)
    loc = pn.widgets.TextInput(name="Location description", placeholder="Street / suburb / landmark")
    desc = pn.widgets.TextAreaInput(name="Describe the problem", placeholder="What is wrong with the streetlight? When did you notice it?", height=110)
    contact = pn.widgets.TextInput(name="Contact (optional)")
    centre = pn.widgets.Button(name="Use Chinhoyi centre", button_type="light", width=180)
    submit = pn.widgets.Button(name="Submit streetlight report", button_type="success", height=42, sizing_mode="stretch_width")
    target_note = pn.pane.HTML("", sizing_mode="stretch_width")
    message = pn.pane.HTML("", sizing_mode="stretch_width")

    def set_target(rec: Dict[str, Any]) -> None:
        pre = report_prefill_from(rec) if "_lat" in rec else dict(rec)
        if pre.get("asset_id"):
            asset.value = str(pre["asset_id"])
        if pre.get("latitude") is not None and pre.get("longitude") is not None:
            lat.value, lon.value = float(pre["latitude"]), float(pre["longitude"])
        if pre.get("location_text"):
            loc.value = str(pre["location_text"])
        target_note.object = (f"<div class='moc-alert moc-alert-info'>Reporting streetlight <b>{esc(pre.get('asset_id', ''))}</b>. "
                              f"Choose the problem category and describe what is wrong, then submit.</div>")
        try:
            pn.state.notifications.info(f"Streetlight {pre.get('asset_id', '')} selected for your report.", duration=3500)
        except Exception:
            pass

    def use_centre(_=None):
        lat.value, lon.value = CHINHOYI_LAT, CHINHOYI_LON

    def do_submit(_=None):
        if not (desc.value or "").strip():
            message.object = "<div class='moc-alert moc-alert-warn'>Please describe the streetlight problem before submitting.</div>"
            return
        if lat.value is None or lon.value is None:
            message.object = "<div class='moc-alert moc-alert-warn'>Please give a latitude and longitude, or use “Use Chinhoyi centre”.</div>"
            return
        rid = save_report(ctx.session.user, cat.value, lat.value, lon.value, loc.value, desc.value, contact.value, priority.value, asset_id=(asset.value or "").strip() or None)
        add_notification(ctx.session.user["id"], f"Report {rid} was submitted successfully.", "success")
        audit(ctx.session.user["username"], ctx.session.role, "submit_report", rid)
        message.object = (f"<div class='moc-alert moc-alert-ok'><b>Report {esc(rid)}</b> submitted. Keep this reference number; "
                          f"you can follow progress under <b>My Reports</b>.</div>")
        desc.value = ""
        asset.value = ""
        target_note.object = ""
        ctx.report_prefill = {}

    centre.on_click(use_centre)
    submit.on_click(do_submit)
    if ctx.report_prefill:
        set_target(ctx.report_prefill)
        ctx.report_prefill = {}
    form = pn.Column(
        target_note,
        pn.Row(cat, priority, sizing_mode="stretch_width"),
        asset,
        pn.Row(lat, lon, centre, sizing_mode="stretch_width"),
        loc, desc, contact, submit, message,
        sizing_mode="stretch_width")
    return form, set_target


def build_community_home(ctx: AppContext):
    """A deliberately simplified, public-safe home page: streetlight status, the map, and the problem
    report form. Unlike the staff overview, this never exposes internal analytics (budgets,
    rankings, recommended sites) and only ever shows the signed-in resident's OWN reports."""
    my_reports = get_reports(user_id=ctx.session.user["id"])
    open_my = int((~my_reports["status"].isin(["Resolved", "Closed"])).sum()) if not my_reports.empty else 0
    form, set_target = build_report_form(ctx)
    recent = my_reports.head(5)[[c for c in ["report_id", "reported_at", "category", "asset_id", "status"] if c in my_reports.columns]] if not my_reports.empty else pd.DataFrame()
    return pn.Column(
        page_header("Community streetlight portal", "See which streetlights are working in your area, and report a problem in a few steps.", "COMMUNITY"),
        pn.Row(*streetlight_kpi_row(ctx, include_new_sites=False).objects,
               kpi("Your open reports", open_my, "Reports you submitted awaiting action", COLOR_ORANGE),
               sizing_mode="stretch_width"),
        pn.Spacer(height=6),
        section_title("Streetlight map — click a light, then report a problem with it"),
        StreetlightMap(ctx, explorer_view(ctx, "all"), on_report=set_target, height=400).panel,
        pn.Spacer(height=6),
        section_title("Report a streetlight problem"),
        form,
        pn.Spacer(height=6),
        section_title("Your recent reports"),
        df_table(recent, height=260) if not recent.empty else pn.pane.Markdown("_You have not submitted any reports yet._"),
        sizing_mode="stretch_width")


def build_community_report(ctx):
    form, _ = build_report_form(ctx)
    return pn.Column(page_header("Report a streetlight problem", "Submit a public-service report for a failed, damaged or poorly performing streetlight. Only your own report records are exposed in the community account.", ctx.session.role), form, sizing_mode="stretch_width")


def build_my_reports(ctx):
    reports = get_reports(user_id=ctx.session.user["id"])
    return pn.Column(page_header("My streetlight reports", "Track the status and municipal response associated with reports submitted by this community account.", ctx.session.role), df_table(reports, height=500), sizing_mode="stretch_width")


def build_safety():
    return pn.Column(page_header("Streetlight safety information", "Public information about how the dashboard uses spatial indicators to support safer and better lit municipal environments.", "COMMUNITY"), card("How placement is prioritised", "New streetlight sites are ranked with a multi-criteria method. The criteria are weighted using pairwise comparisons (the Analytic Hierarchy Process) and cover lighting gaps, population pressure, pedestrian activity, public facilities, road hierarchy, available safety/crime evidence, solar potential, terrain, environmental constraints and maintenance access."), card("Important limitation", "A high spatial score is not a finding that a location is unsafe or that an incident has occurred there. Municipal verification and appropriate public safety procedures remain necessary."), sizing_mode="stretch_width")


def build_help():
    return pn.Column(page_header("Help & contact", "Guidance for reporting streetlight problems and understanding public dashboard information.", "COMMUNITY"), card("Streetlight reporting", "Click the streetlight on the map (or type its ID), choose a problem category and describe what is wrong. Avoid submitting private or sensitive information not needed to resolve the issue."), card("Municipal workflow", "Submitted reports can move through verification, assignment, work order creation, progress and closure. Status information becomes visible as municipal staff update the record."), sizing_mode="stretch_width")


def build_profile(ctx):
    u = ctx.session.user or {}
    return pn.Column(page_header("My profile", "Account information visible to the signed-in community user.", ctx.session.role), card("Account", f"<b>Name:</b> {esc(u.get('full_name',''))}<br><b>Username:</b> {esc(u.get('username',''))}<br><b>Email:</b> {esc(u.get('email',''))}<br><b>Role:</b> {esc(ROLE_META.get(u.get('role'),{}).get('label',''))}"), sizing_mode="stretch_width")

# =====================================================================
# ADMIN VIEWS
# =====================================================================

def build_admin_home(ctx):
    reports = get_reports(); audits = get_audit_logs(10); users = list_users(); scored = ctx.scored()
    return pn.Column(
        page_header("System administration overview", "Governance view for accounts, permissions, spatial datasets, security logs and system-level reports.", ctx.session.role),
        data_mode_alert(),
        pn.Spacer(height=6),
        section_title("Streetlight assets"),
        streetlight_kpi_row(ctx),
        pn.Spacer(height=6),
        home_map(ctx, height=380),
        pn.Spacer(height=10),
        pn.Row(kpi("Users", len(users), "Registered accounts", COLOR_NAVY), kpi("Active users", int(users.active.sum()) if not users.empty else 0, "Enabled accounts", COLOR_GREEN), kpi("Community reports", len(reports), "Service records", COLOR_ORANGE), kpi("Audit entries", len(audits), "Latest security trail", COLOR_RED), sizing_mode="stretch_width"),
        sizing_mode="stretch_width")


def build_admin_users(ctx):
    users = list_users(); selected = pn.widgets.IntInput(name="Selected user ID", value=int(users.iloc[0].id) if not users.empty else 0, start=0)
    full_name = pn.widgets.TextInput(name="Full name"); email = pn.widgets.TextInput(name="Email"); role = pn.widgets.Select(name="Role", options=list(ROLES), value="COMMUNITY"); dept = pn.widgets.Select(name="Department", options=list(DEPARTMENTS), value="PUBLIC_LIGHTING"); password = pn.widgets.PasswordInput(name="New password")
    create = pn.widgets.Button(name="Create user", button_type="success"); toggle = pn.widgets.Button(name="Toggle active state", button_type="warning"); reset = pn.widgets.Button(name="Reset password", button_type="danger"); msg = pn.pane.HTML("")

    def do_create(_=None):
        username = full_name.value.strip().lower().replace(" ", "_") or f"user_{secrets.token_hex(3)}"
        ok, text = create_user(username, email.value, full_name.value, role.value, password.value, dept.value if role.value != "COMMUNITY" else None)
        if ok:
            audit(ctx.session.user["username"], ctx.session.role, "create_user", username)
        msg.object = f"<div class='moc-alert moc-alert-{'ok' if ok else 'warn'}'>{esc(text)}</div>"

    def toggle_user(_=None):
        df = list_users(); row = df[df.id == selected.value]
        if row.empty:
            msg.object = "<div class='moc-alert moc-alert-warn'>Enter a valid user ID.</div>"; return
        set_user_active(selected.value, not bool(int(row.iloc[0].active))); audit(ctx.session.user["username"], ctx.session.role, "toggle_user", str(selected.value)); msg.object = "<div class='moc-alert moc-alert-ok'>User state updated (a disabled account is signed out everywhere).</div>"

    def reset_user(_=None):
        if len(password.value or "") < 8:
            msg.object = "<div class='moc-alert moc-alert-warn'>Password must contain at least 8 characters.</div>"; return
        reset_password(selected.value, password.value); audit(ctx.session.user["username"], ctx.session.role, "reset_password", str(selected.value)); msg.object = "<div class='moc-alert moc-alert-ok'>Password reset (the user's existing sessions were ended).</div>"

    create.on_click(do_create); toggle.on_click(toggle_user); reset.on_click(reset_user)
    return pn.Column(page_header("User management", "Create accounts, maintain role assignments and control active status. Default development credentials should be replaced for deployment.", ctx.session.role), df_table(users, height=300), pn.Row(selected, full_name, email, role, dept, sizing_mode="stretch_width"), password, pn.Row(create, toggle, reset, sizing_mode="stretch_width"), msg, sizing_mode="stretch_width")


def build_admin_permissions(ctx):
    role = pn.widgets.Select(name="Role", options=list(ROLES), value="MUNICIPAL_STAFF")
    feature = pn.widgets.Select(name="Feature", options=list(PERMISSIONS))
    enabled = pn.widgets.Checkbox(name="Enabled", value=True); save = pn.widgets.Button(name="Save permission", button_type="success"); msg = pn.pane.HTML("")
    table = pn.bind(lambda r: df_table(pd.DataFrame({"Feature": list(PERMISSIONS), "Allowed": [has_permission(r, f) for f in PERMISSIONS]}), height=520), role)

    def mutate(_=None):
        with connect_db() as conn:
            if enabled.value:
                conn.execute("INSERT OR IGNORE INTO role_permissions(role,feature) VALUES(?,?)", (role.value, feature.value))
            else:
                conn.execute("DELETE FROM role_permissions WHERE role=? AND feature=?", (role.value, feature.value))
            conn.commit()
        invalidate_permission_cache()
        audit(ctx.session.user["username"], ctx.session.role, "permission_change", f"{role.value}:{feature.value}:{enabled.value}")
        msg.object = "<div class='moc-alert moc-alert-ok'>Permission saved.</div>"

    save.on_click(mutate)
    return pn.Column(page_header("Role & permission management", "Control which modules are visible to each role. Changes apply to new route checks immediately.", ctx.session.role), pn.Row(role, feature, enabled, save, sizing_mode="stretch_width"), msg, table, sizing_mode="stretch_width")


def build_admin_data(ctx):
    rows = []
    for name in LAYER_ALIASES:
        detected_real = RAW_DATA_FLAGS.get(name, False)
        source = find_layer(name)
        rows.append({
            "Layer": name.replace("_", " ").title(),
            "Real data detected": "Yes" if detected_real else "No",
            "Currently showing": "Real municipal data" if detected_real else ("Demo / simulated" if name in DEMO_LAYERS else "Not available"),
            "Source file": str(source) if source else "—",
        })
    rows.append({"Layer": "DEM / Terrain (raster)", "Real data detected": "Yes" if RAW_DATA_FLAGS.get("dem") else "No", "Currently showing": "Detected, but terrain suitability is not calculated in this version" if RAW_DATA_FLAGS.get("dem") else "Not available — terrain suitability defaulted to neutral", "Source file": str(DATA.dem_path) if DATA.dem_path else "—"})
    rows.append({"Layer": "Solar irradiance (raster)", "Real data detected": "Yes" if RAW_DATA_FLAGS.get("solar") else "No", "Currently showing": "Real" if RAW_DATA_FLAGS.get("solar") else "Not available — neutral solar score used", "Source file": str(DATA.solar_path) if DATA.solar_path else "—"})
    return pn.Column(page_header("GIS data management", "Review which layers are available to the application. Real municipal/research data should be checked before analytical interpretation.", ctx.session.role), df_table(pd.DataFrame(rows), height=500), data_mode_alert(), sizing_mode="stretch_width")


def build_admin_gis(ctx):
    return pn.Column(page_header("GIS configuration", "Review the spatial configuration currently used by the decision-support engine.", ctx.session.role), card("Coordinate systems", f"Display CRS: {WGS84} (map in Web Mercator, EPSG:3857)<br>Analysis CRS: {UTM}<br>Chinhoyi centre: {CHINHOYI_LAT}, {CHINHOYI_LON}"), card("Current weighting method", f"{esc(ctx.scenario.value)} with {len(ctx.weights())} weighted criteria. Candidate spacing defaults to {CANDIDATE_SPACING_M:.0f} m along available roads; new lights are kept at least {int(_widget_float(ctx.min_spacing, 0))} m apart."), sizing_mode="stretch_width")


def build_admin_audit(ctx):
    """Audit trail plus the list of live sessions, with the ability to end a user's sessions."""
    sessions_col = pn.Column(sizing_mode="stretch_width")
    target = pn.widgets.IntInput(name="User ID whose sessions should be ended", value=0, start=0)
    end_btn = pn.widgets.Button(name="End this user's sessions", button_type="danger", width=230)
    msg = pn.pane.HTML("", sizing_mode="stretch_width")

    def show_sessions():
        sessions = list_active_sessions()
        if not sessions.empty:
            sessions = sessions.rename(columns={"user_id": "User ID", "username": "Username", "role": "Role", "created_at": "Signed in", "last_seen": "Last active", "expires_at": "Expires"})
        sessions_col[:] = [df_table(sessions, height=260)]

    def end_sessions(_=None):
        if int(target.value or 0) <= 0:
            msg.object = "<div class='moc-alert moc-alert-warn'>Enter the ID of the user (see User Management).</div>"
            return
        n = revoke_user_sessions(int(target.value))
        audit(ctx.session.user["username"], ctx.session.role, "revoke_sessions", f"user {int(target.value)}: {n} session(s)")
        msg.object = f"<div class='moc-alert moc-alert-ok'>Ended {n} active session(s) for user {int(target.value)}. They will need to sign in again.</div>"
        show_sessions()

    end_btn.on_click(end_sessions)
    show_sessions()
    return pn.Column(
        page_header("Audit & security", "Review authentication and administrative actions recorded by the application, and the sessions that are currently signed in.", ctx.session.role),
        section_title("Active sessions"),
        sessions_col,
        pn.Row(target, end_btn, sizing_mode="stretch_width"), msg,
        section_title("Audit trail"),
        df_table(get_audit_logs(300), height=550), sizing_mode="stretch_width")


def build_admin_reports(ctx):
    reports = get_reports(); orders = get_work_orders()
    open_n = 0 if reports.empty else int((~reports.status.isin(["Resolved", "Closed"])).sum())
    return pn.Column(page_header("System reports", "Management summary of public service reports and maintenance workload.", ctx.session.role), pn.Row(kpi("Reports", len(reports), "All community submissions", COLOR_ORANGE), kpi("Open reports", open_n, "Not yet closed", COLOR_RED), kpi("Work orders", len(orders), "Database work orders", COLOR_NAVY), sizing_mode="stretch_width"), pn.Spacer(height=10), df_table(reports, height=430), sizing_mode="stretch_width")


def build_admin_maintenance(ctx):
    orders = get_work_orders()
    table = orders if not orders.empty else DATA.maintenance
    return pn.Column(page_header("Maintenance oversight", "System-wide view of current work orders and asset status.", ctx.session.role), df_table(table, height=530), sizing_mode="stretch_width")


def build_admin_notices(ctx):
    """Back-end notes that used to sit at the top of working pages: data mode, status provenance,
    inventory reconciliation, attribute coverage, AHP state, prioritisation findings, basemap terms."""
    objs: List[Any] = [page_header("System notices", "Data-quality, method and configuration notes for administrators. Working pages stay uncluttered; only compact pills in the top bar flag demo data, simulated status or provisional AHP weights.", ctx.session.role)]
    for kind, text in system_notices(ctx):
        objs += [alert(text, kind), pn.Spacer(height=4)]
    return pn.Column(*objs, sizing_mode="stretch_width")


def build_admin_settings(ctx):
    municipal_real = MUNICIPAL_LOGO.exists()
    rows = pd.DataFrame({
        "Setting": ["Base directory", "Data directory", "Export directory", "Database", "Municipality logo", "Interactive map (Bokeh)", "LLM assistant", "Rasterio", "Scikit-learn", "Using default credentials", "Session lifetime", "Browser session cookie", "Streetlight status source", "Streetlight asset IDs", "AHP starting matrix", "Minimum spacing between new lights"],
        "Value": [
            str(BASE_DIR), str(DATA_DIR), str(EXPORT_DIR), str(DB_PATH),
            f"{MUNICIPAL_LOGO} ({'real file, white background removed when Pillow is installed' if municipal_real else 'placeholder badge shown'})",
            "Available" if BOKEH_AVAILABLE else "Not installed",
            "Enabled" if LLM_AVAILABLE else "Template/offline",
            "Available" if RASTERIO_AVAILABLE else "Not installed",
            "Available" if SKLEARN_AVAILABLE else "Not installed",
            "Yes — replace before deployment" if USING_DEFAULT_CREDENTIALS else "No",
            f"{SESSION_TTL_HOURS:g} hours, sliding (SESSION_TTL_HOURS)",
            "Active" if SessionBridge is not None else "Unavailable (ReactiveHTML missing) — sessions are saved on the server only",
            status_source_text(),
            STREETLIGHT_META["asset_id_source"],
            AHP_INITIAL_SOURCE,
            f"{DEFAULT_MIN_SPACING_M} m (MIN_NEW_LIGHT_SPACING_M)",
        ],
    })
    return pn.Column(
        page_header("System settings", "Operational paths and branding configuration used by this deployment.", ctx.session.role),
        df_table(rows, height=500),
        alert(
            "For deployment: set strong administrator credentials via environment variables, confirm data sources on the Data Management page, and place the Municipality of Chinhoyi's real approved logo file in the assets folder "
            "(or point MUNICIPAL_LOGO_PATH at it). Until then, the top bar shows an original generic placeholder badge, not the official crest.",
            "info" if municipal_real else "warn",
        ),
        sizing_mode="stretch_width")

# =====================================================================
# CHATBOT / DASHBOARD ASSISTANT
# =====================================================================

class DashboardAssistant:
    """Answers questions using only real, currently-computed dashboard figures. When an Anthropic
    API key is configured, those same figures are handed to the model as the ONLY source of truth
    it may draw on; if the key is absent, the request fails, or the SDK is unavailable, the
    deterministic template answer is used instead."""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def _ahp_facts(self, scored) -> dict:
        a = self.ctx.ahp()
        return {
            "weighting_method": self.ctx.scenario.value,
            "ahp_consistency_ratio": round(a["cr"], 3),
            "ahp_consistent": a["consistent"],
            "ahp_criteria_weights_pct": {FEATURE_LABELS[k]: round(100 * float(v), 1) for k, v in a["weights_by_key"].items()},
            "recommended_new_solar_lights": int(scored["funded_phase_1"].sum()) if not scored.empty else 0,
            "min_spacing_m": int(_widget_float(self.ctx.min_spacing, DEFAULT_MIN_SPACING_M)),
        }

    def _facts(self) -> dict:
        """The fact set the assistant (and, if enabled, the LLM) is allowed to draw on. This is a
        data-minimisation boundary: a Community session's dict never CONTAINS the budget, weights
        or candidate-site rankings, so there is nothing to leak."""
        role = self.ctx.session.role
        base = {
            "role": role,
            "existing_streetlights": int(len(DATA.streetlights)) if DATA.streetlights is not None else 0,
            "operational_streetlights": int((DATA.maintenance["status"] == "Operational").sum()) if not DATA.maintenance.empty else 0,
            "faulty_streetlights": int((DATA.maintenance["status"] == "Faulty").sum()) if not DATA.maintenance.empty else 0,
            "offline_streetlights": int((DATA.maintenance["status"] == "Offline").sum()) if not DATA.maintenance.empty else 0,
            "using_demo_data": not CORE_REAL_DATA,
        }
        if role == "COMMUNITY":
            user = self.ctx.session.user
            my_reports = get_reports(user_id=user["id"]) if user else pd.DataFrame()
            base.update({
                "your_total_reports": int(len(my_reports)),
                "your_open_reports": int((~my_reports["status"].isin(["Resolved", "Closed"])).sum()) if not my_reports.empty else 0,
                "assistant_scope_note": (
                    "This is the PUBLIC community assistant. Municipal budgets, candidate-site rankings, criterion weights, scenario "
                    "settings and other residents' reports are intentionally not included in these facts and must "
                    "never be guessed at or invented."
                ),
            })
            return base
        # MUNICIPAL_STAFF / ADMIN get the fuller internal operational picture.
        scored = self.ctx.last_scored if self.ctx.last_scored is not None else self.ctx.scored()
        reports = get_reports()
        top = scored.iloc[0].to_dict() if not scored.empty else None
        if top is not None:
            top = {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in top.items() if k != "geometry"}
        base.update({
            "scenario": self.ctx.scenario.value,
            "available_scenarios": list(SCENARIOS),
            "coverage_radius_m": int(_widget_float(self.ctx.coverage_radius, DEFAULT_COVERAGE_RADIUS_M)),
            "budget_usd": _widget_float(self.ctx.budget, 0.0),
            "unit_cost_usd": _widget_float(self.ctx.unit_cost, 0.0),
            "contingency_pct": _widget_float(self.ctx.contingency, 0.0),
            "candidate_count": int(len(scored)),
            "funded_phase_1_count": int(scored["funded_phase_1"].sum()) if not scored.empty else 0,
            "top_candidate": top,
            **self._ahp_facts(scored),
            "open_community_reports": int((~reports["status"].isin(["Resolved", "Closed"])).sum()) if not reports.empty else 0,
        })
        if role == "ADMIN":
            users_df = list_users()
            base.update({
                "total_user_accounts": int(len(users_df)),
                "active_user_accounts": int(users_df.active.sum()) if not users_df.empty else 0,
                "total_community_reports": int(len(reports)),
            })
        return base

    def _template_answer(self, q: str, facts: dict) -> str:
        role = facts.get("role")
        if role == "COMMUNITY":
            if "report" in q:
                return f"You have {facts['your_total_reports']} report(s) on file with the municipality, {facts['your_open_reports']} of which are still open."
            if any(k in q for k in ("light", "streetlight", "status", "operational", "working")):
                return f"There are {facts['existing_streetlights']} streetlights recorded for this service area; {facts['operational_streetlights']} are currently marked operational."
            if any(k in q for k in ("budget", "priority", "scenario", "candidate", "rank", "weight", "ahp")):
                return "Municipal budget figures, criterion weights and site-prioritisation details are part of the internal planning workspace and aren't available from the public assistant. If you have a specific streetlight concern, please use \"Report a Problem\"."
            return "Ask about streetlight status in your area, or the status of reports you've submitted. To report a fault, click the light on the map, or use \"Report a Problem\"."
        # MUNICIPAL_STAFF / ADMIN
        if any(k in q for k in ("ahp", "weight", "consisten", "criteria", "criterion")):
            top3 = sorted(facts["ahp_criteria_weights_pct"].items(), key=lambda kv: -kv[1])[:3]
            lead = ", ".join(f"{name} ({pct}%)" for name, pct in top3)
            verdict = "consistent" if facts["ahp_consistent"] else "NOT consistent (CR is 0.10 or more), so the weights are provisional"
            return f"The AHP consistency ratio is {facts['ahp_consistency_ratio']}, which is {verdict}. The three heaviest criteria are {lead}. The active weighting method is {facts['weighting_method']}."
        if any(k in q for k in ("new light", "recommend", "where should", "install", "placement", "new solar")):
            return f"The AHP/MCDA selection currently recommends {facts['recommended_new_solar_lights']} new solar streetlights within the ${facts['budget_usd']:,.0f} budget, kept at least {facts['min_spacing_m']} m apart. See the MCDA Site Selection page for the list and map."
        if "coverage" in q:
            return f"The configured coverage radius is {facts['coverage_radius_m']} m. Coverage analysis uses existing streetlights and population points where those datasets are available."
        if "top" in q or "priority" in q:
            top = facts.get("top_candidate")
            if not top:
                return "No candidate sites are currently available."
            rank = top.get("rank", "?")
            overall = top.get("overall_priority", 0)
            return f"Current top candidate: {top.get('candidate_id', 'Unknown')}, ranked {rank}, with a suitability score of {overall}/100 using {facts['weighting_method']} weights."
        if "budget" in q:
            return f"The planning budget is ${facts['budget_usd']:,.0f}. At the current unit cost and contingency settings, the plan funds {facts['funded_phase_1_count']} new lights."
        if "maintenance" in q or "fault" in q:
            return f"The current asset registry contains {facts['existing_streetlights']} streetlights, including {facts['faulty_streetlights']} faulty and {facts['offline_streetlights']} offline assets."
        if "scenario" in q:
            return "Available preset weight sets are: " + ", ".join(facts["available_scenarios"]) + f". AHP-derived weights are also available ({AHP_METHOD})."
        if "report" in q:
            return f"There are currently {facts['open_community_reports']} open (not resolved/closed) community reports."
        if role == "ADMIN" and any(k in q for k in ("user", "account")):
            return f"There are {facts.get('total_user_accounts', 0)} registered accounts, {facts.get('active_user_accounts', 0)} of which are active."
        extra = ", or user accounts" if role == "ADMIN" else ""
        return f"Ask about AHP weights and consistency, recommended new lights, coverage, priority sites, budget, maintenance, scenarios or community reports{extra}."

    def answer(self, text: str) -> str:
        q = (text or "").lower().strip()
        facts = self._facts()
        template_reply = self._template_answer(q, facts)
        if ANTHROPIC_CLIENT is None:
            return template_reply
        try:
            role = facts.get("role")
            audience = {
                "COMMUNITY": "a member of the public (a resident of Chinhoyi)",
                "MUNICIPAL_STAFF": "municipal staff with internal operational access",
                "ADMIN": "a system administrator with full internal access",
            }.get(role, "a dashboard user")
            system = (
                f"You are a factual assistant embedded in a municipal streetlight dashboard for Chinhoyi, Zimbabwe, "
                f"currently answering {audience}. Only use the JSON facts provided below — that JSON has already been "
                "filtered to what this specific audience is allowed to see, so treat its absence of a field (e.g. no "
                "budget or candidate-ranking data for a public/community audience) as information that must stay "
                "unavailable, not as something to infer or estimate. Never invent statistics, locations, budgets or "
                "dates that are not present in the facts. If the facts do not contain the answer, say plainly that the "
                "information is not available here. Keep the answer under 80 words and do not use markdown headers."
            )
            message = ANTHROPIC_CLIENT.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=220,
                system=system,
                messages=[{"role": "user", "content": f"Dashboard facts (JSON): {json.dumps(facts, default=str)}\n\nQuestion: {text}"}],
            )
            parts = [block.text for block in message.content if getattr(block, "type", "") == "text"]
            reply = "\n".join(parts).strip()
            return reply or template_reply
        except Exception as exc:
            logger.warning("LLM assistant call failed, using template answer: %s", exc)
            return template_reply


def build_chatbot(ctx):
    assistant = DashboardAssistant(ctx)
    chat = pn.chat.ChatInterface(callback=lambda content, user, instance: assistant.answer(content), height=520)
    mode_note = "LLM-assisted (grounded in live dashboard figures only)" if ANTHROPIC_CLIENT is not None else "Offline / template mode — set ANTHROPIC_API_KEY to enable LLM-assisted phrasing"
    role = ctx.session.role
    scope_note = (
        "As a community account, this assistant only ever sees streetlight status and your own report history — "
        "never municipal budgets, site rankings, criterion weights or other residents' reports."
        if role == "COMMUNITY" else
        "This account sees the full internal operational picture (budget, AHP weights, candidate rankings, reports)."
    )
    return pn.Column(
        page_header("Dashboard Assistant", "A lightweight question-and-answer layer for dashboard summaries. It is informational and does not replace municipal review.", role),
        chat, sizing_mode="stretch_width")

# =====================================================================
# ROUTER / APPLICATION
# =====================================================================

def render_route(ctx: AppContext):
    role, route = ctx.session.role, ctx.route
    if not can_route(role, route):
        return alert("This account does not have permission to access the selected module.", "danger")
    try:
        if role == "ADMIN":
            content = {
                "home": lambda: build_admin_home(ctx), "explorer": lambda: build_streetlight_explorer(ctx), "users": lambda: build_admin_users(ctx), "permissions": lambda: build_admin_permissions(ctx),
                "data": lambda: build_admin_data(ctx), "gis_config": lambda: build_admin_gis(ctx), "audit": lambda: build_admin_audit(ctx),
                "reports": lambda: build_admin_reports(ctx), "maintenance_ov": lambda: build_admin_maintenance(ctx), "settings": lambda: build_admin_settings(ctx), "notices": lambda: build_admin_notices(ctx),
                "chatbot": lambda: build_chatbot(ctx),
            }.get(route, lambda: build_admin_home(ctx))()
        elif role == "MUNICIPAL_STAFF":
            content = {
                "home": lambda: build_staff_overview(ctx), "explorer": lambda: build_streetlight_explorer(ctx), "assets": lambda: build_assets(ctx), "gaps": lambda: build_gaps(ctx),
                "prioritisation": lambda: build_prioritisation(ctx), "maintenance": lambda: build_maintenance(ctx), "reports": lambda: build_reports_staff(ctx),
                "ahp": lambda: build_ahp(ctx), "mcda": lambda: build_decision_engine(ctx), "what_if": lambda: build_what_if(ctx), "budget": lambda: build_budget(ctx),
                "analytics": lambda: build_analytics(ctx), "export": lambda: build_export(ctx), "chatbot": lambda: build_chatbot(ctx),
            }.get(route, lambda: build_staff_overview(ctx))()
        else:
            content = {
                "home": lambda: build_community_home(ctx), "explorer": lambda: build_streetlight_explorer(ctx), "submit_report": lambda: build_community_report(ctx), "my_reports": lambda: build_my_reports(ctx),
                "safety": build_safety, "help": build_help, "profile": lambda: build_profile(ctx),
                "chatbot": lambda: build_chatbot(ctx),
            }.get(route, lambda: build_community_home(ctx))()
    except Exception as exc:
        logger.exception("Page render failed for role=%s route=%s: %s", role, route, exc)
        return alert("This page could not be rendered because of an unexpected error. The problem has been logged; please try another section or contact support.", "danger")
    return content


APP_TITLE = "Chinhoyi Smart Solar Streetlight Management System"


def _session_cookie_token() -> Optional[str]:
    """The session cookie sent by this browser with the page request (None outside a server session)."""
    try:
        cookies = pn.state.cookies
        return cookies.get(SESSION_COOKIE) if cookies else None
    except Exception:
        return None


def create_dashboard() -> pn.Column:
    """Factory that builds one fully independent dashboard instance — its own AppContext, session
    and widgets — per browser session. This MUST remain a factory (rather than a module-level
    singleton) so that concurrent municipal users never share login state or in-progress inputs.
    Layout: [top bar] / [sidebar | page] / [footer] — each shown once. A valid session cookie
    resumes the saved session (and the saved page and settings) without signing in again."""
    ctx = AppContext()
    bridge = SessionBridge(width=0, height=0, margin=0) if SessionBridge is not None else None
    ctx.bridge = bridge
    topbar_slot = pn.Column(sizing_mode="stretch_width")
    side = pn.Column(width=250, css_classes=["moc-sidebar"], visible=False)
    main = pn.Column(sizing_mode="stretch_width", css_classes=["moc-main"])
    shell = pn.Row(side, main, sizing_mode="stretch_width", css_classes=["moc-shell"])
    footer = pn.pane.HTML("<div class='moc-footer'>Municipality of Chinhoyi · Smart Solar Streetlight Management &amp; Decision-Support Dashboard</div>", sizing_mode="stretch_width")
    root = pn.Column(*( [bridge] if bridge is not None else [] ), topbar_slot, shell, footer, sizing_mode="stretch_width", css_classes=["moc-root"])

    def navigate(route):
        ctx.route = route
        ctx.save_state()
        rerender()

    def rerender():
        if not ctx.session.authenticated:
            topbar_slot[:] = []
            side.visible = False
            side[:] = []
            main[:] = [build_login(ctx, rerender)]
            return
        topbar_slot[:] = [build_header(ctx, rerender)]
        side[:] = build_sidebar(ctx, navigate)
        side.visible = True
        main[:] = [render_route(ctx)]

    ctx.navigate = navigate  # lets KPI cards and map buttons on any page open another page through the real router

    token = _session_cookie_token()
    if token:
        user = get_session_user(token)
        if user is not None:
            ctx.resume_session(user, token)
    rerender()
    return root

# =====================================================================
# SERVER
# =====================================================================

def server_config():
    port = int(os.environ.get("PORT", os.environ.get("PANEL_PORT", "5006")))
    address = os.environ.get("PANEL_ADDRESS", "0.0.0.0")
    show = os.environ.get("PANEL_SHOW", "false").lower() == "true"
    origin = os.environ.get("PANEL_ALLOW_WS_ORIGIN", "localhost:5006,127.0.0.1:5006")
    origins = [x.strip() for x in origin.split(",") if x.strip()] or ["*"]
    return dict(port=port, address=address, show=show, origins=origins)


def startup_banner(cfg):
    logger.info("=" * 78)
    logger.info("CHINHOYI SMART SOLAR STREETLIGHT MANAGEMENT SYSTEM")
    logger.info("=" * 78)
    logger.info("Core real data : %s", "YES" if CORE_REAL_DATA else "NO — demo/data-limited fallback")
    logger.info("Interactive map: %s", "YES (Bokeh)" if BOKEH_AVAILABLE else "NO — Bokeh missing")
    logger.info("Rasterio       : %s", "YES" if RASTERIO_AVAILABLE else "NO")
    logger.info("Scikit-learn   : %s", "YES" if SKLEARN_AVAILABLE else "NO")
    logger.info("LLM assistant  : %s", "YES" if LLM_AVAILABLE else "OFFLINE / templated")
    logger.info("AHP matrix     : %s", AHP_INITIAL_SOURCE)
    logger.info("Sessions       : %s, %g h sliding", "cookie + server" if SessionBridge is not None else "server only", SESSION_TTL_HOURS)
    logger.info("Municipal logo : %s", MUNICIPAL_LOGO)
    logger.info("Database       : %s", DB_PATH)
    logger.info("URL            : http://localhost:%s/", cfg["port"])
    logger.info("=" * 78)


# Initialise the database once at import time (idempotent — safe even if the module is
# re-executed per Bokeh session under `panel serve`).
init_db()

# Module-level instance kept for `panel serve GIS_SCRIPT.py`, which re-executes this whole module
# per session, giving each visitor an independent `application` object automatically.
application = create_dashboard()
application.servable(title=APP_TITLE)

if __name__ == "__main__":
    cfg = server_config()
    startup_banner(cfg)
    # IMPORTANT: pass the *factory function*, not the pre-built `application` instance, so pn.serve
    # constructs one independent AppContext/session per connecting browser.
    pn.serve(
        create_dashboard,
        port=cfg["port"],
        address=cfg["address"],
        show=cfg["show"],
        allow_websocket_origin=cfg["origins"],
        title=APP_TITLE,
    )
