import io
import os
import folium
from folium.plugins import HeatMap
import geopandas as gpd
import numpy as np
import pandas as pd
import panel as pn
from pyproj import Transformer
from shapely.geometry import Point

# Initialize Panel extension
pn.extension('tabulator', notifications=True)

# ---------------------------------------------------------
# 1. AUTOMATIC DATA CONVERSION & SPATIAL LOADER
# ---------------------------------------------------------
DATA_DIR = 'data'
CENSUS_SHP = os.path.join(DATA_DIR, 'moc_census_data_addresses.shp')
CENSUS_PARQUET = os.path.join(DATA_DIR, 'moc_census_data_addresses.parquet')


def check_and_convert_data():
  """Converts address shapefile to fast GeoParquet format if missing."""
  if not os.path.exists(CENSUS_PARQUET) and os.path.exists(CENSUS_SHP):
    print("⚡ Preprocessing address shapefile for high performance...")
    try:
      gdf = gpd.read_file(CENSUS_SHP)
      if gdf.crs is None:
        gdf.set_crs('EPSG:32736', inplace=True)
      gdf = gdf.to_crs('EPSG:4326')
      gdf.to_parquet(CENSUS_PARQUET)
      print("✅ Converted address shapefile to GeoParquet successfully!")
    except Exception as e:
      print(f'⚠️ Address conversion warning: {e}')


check_and_convert_data()


@pn.cache
def load_spatial_layers():
  boundary_gdf = None
  suburbs_gdf = None
  census_coords = []

  # 1. Read address GeoParquet dataset for Heatmap
  if os.path.exists(CENSUS_PARQUET):
    try:
      gdf = gpd.read_parquet(CENSUS_PARQUET)
      if len(gdf) > 3000:
        gdf = gdf.sample(n=3000, random_state=42)
      census_coords = [
          [geom.y, geom.x] for geom in gdf.geometry if geom.type == 'Point'
      ]
    except Exception as e:
      print(f'Error reading Parquet heatmap data: {e}')

  # 2. Read Suburbs shapefile and Municipal Boundary shapefile
  if os.path.exists(DATA_DIR):
    for file in os.listdir(DATA_DIR):
      file_lower = file.lower()
      full_path = os.path.join(DATA_DIR, file)

      if file.endswith('.shp'):
        try:
          if 'moc_suburbs' in file_lower or 'suburb' in file_lower:
            suburbs_gdf = gpd.read_file(full_path)
            if suburbs_gdf.crs is None:
              suburbs_gdf.set_crs('EPSG:32736', inplace=True)
            suburbs_gdf = suburbs_gdf.to_crs('EPSG:4326')
          elif any(
              k in file_lower for k in ['bound', 'admin', 'chinhoyi', 'mcd']
          ):
            boundary_gdf = gpd.read_file(full_path)
            if boundary_gdf.crs is None:
              boundary_gdf.set_crs('EPSG:32736', inplace=True)
            boundary_gdf = boundary_gdf.to_crs('EPSG:4326')
        except Exception as e:
          print(f'Error loading vector layer {file}: {e}')

  return boundary_gdf, suburbs_gdf, census_coords


boundary_gdf, suburbs_gdf, census_coords = load_spatial_layers()

# ---------------------------------------------------------
# 2. BRAND PALETTES & UPDATED PRIORITY COLOR SCHEME
# ---------------------------------------------------------
COLOR_MUNI_GREEN = '#006837'
COLOR_MUNI_GOLD = '#FDB913'
COLOR_CUT_NAVY = '#002147'
COLOR_CUT_GOLD = '#D4AF37'

COLOR_PRIORITY_HIGH = '#DC2626'  # Rank 1-10 (Red)
COLOR_PRIORITY_MED = '#EA580C'  # Rank 11-25 (Orange)
COLOR_PRIORITY_LOW = '#16A34A'  # Rank 26+ (Green)

CHINHOYI_LAT, CHINHOYI_LON = -17.3667, 30.2000

# ---------------------------------------------------------
# 3. MCDA DATA ENGINE & COORDINATE PROJECTION
# ---------------------------------------------------------
transformer_to_utm = Transformer.from_crs(
    'EPSG:4326', 'EPSG:32736', always_xy=True
)


def generate_base_candidates(num_points=50):
  np.random.seed(42)
  lats = CHINHOYI_LAT + np.random.uniform(-0.02, 0.02, num_points)
  lons = CHINHOYI_LON + np.random.uniform(-0.02, 0.02, num_points)
  dem_score = np.random.uniform(0.3, 1.0, num_points)
  bus_dist_m = np.random.uniform(30, 1000, num_points)
  light_gap_score = np.random.uniform(0.2, 1.0, num_points)

  utme_list, utmn_list = [], []
  for lon, lat in zip(lons, lats):
    e, n = transformer_to_utm.transform(lon, lat)
    utme_list.append(np.round(e, 2))
    utmn_list.append(np.round(n, 2))

  return pd.DataFrame({
      'candidate_id': [f'CSL-{i+1:03d}' for i in range(num_points)],
      'lat': np.round(lats, 6),
      'lon': np.round(lons, 6),
      'utm_easting': utme_list,
      'utm_northing': utmn_list,
      'bus_dist_m': np.round(bus_dist_m, 1),
      'dem_score': np.round(dem_score, 2),
      'light_gap_score': np.round(light_gap_score, 2),
      'bus_proximity_score': np.round(1.0 - (bus_dist_m / 1000), 2),
  })


raw_candidates_df = generate_base_candidates()

# ---------------------------------------------------------
# 4. INTERACTIVE AHP SLIDERS
# ---------------------------------------------------------
w_bus = pn.widgets.FloatSlider(
    name='Bus Stop Proximity Weight',
    start=0.0,
    end=1.0,
    step=0.05,
    value=0.45,
    sizing_mode='stretch_width',
)
w_gap = pn.widgets.FloatSlider(
    name='Light Gap Coverage Weight',
    start=0.0,
    end=1.0,
    step=0.05,
    value=0.30,
    sizing_mode='stretch_width',
)
w_dem = pn.widgets.FloatSlider(
    name='DEM Terrain Weight',
    start=0.0,
    end=1.0,
    step=0.05,
    value=0.25,
    sizing_mode='stretch_width',
)


def calculate_mcda(df, w1, w2, w3):
  total_w = w1 + w2 + w3
  w1_n = w1 / total_w if total_w > 0 else 0.33
  w2_n = w2 / total_w if total_w > 0 else 0.33
  w3_n = w3 / total_w if total_w > 0 else 0.34

  calc_df = df.copy()
  calc_df['mcda_score'] = np.round(
      (calc_df['bus_proximity_score'] * w1_n)
      + (calc_df['light_gap_score'] * w2_n)
      + (calc_df['dem_score'] * w3_n),
      3,
  )
  calc_df = calc_df.sort_values(by='mcda_score', ascending=False).reset_index(
      drop=True
  )
  calc_df['rank'] = calc_df.index + 1
  return calc_df


# ---------------------------------------------------------
# 5. MAP ENGINE
# ---------------------------------------------------------
def create_map(df, boundary, suburbs, density_points):
  m = folium.Map(
      location=[CHINHOYI_LAT, CHINHOYI_LON],
      zoom_start=14,
      max_zoom=20,
      tiles=None,
  )

  folium.TileLayer(
      tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attr='OpenStreetMap',
      name='OpenStreetMap (Standard)',
      max_zoom=20,
      show=False,
  ).add_to(m)

  folium.TileLayer(
      tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attr='Esri World Imagery',
      name='🛰️ Satellite Imagery (High Resolution)',
      max_zoom=20,
      max_native_zoom=19,
      show=True,
  ).add_to(m)

  if boundary is not None:
    folium.GeoJson(
        boundary,
        name='Municipal Boundary',
        style_function=lambda x: {
            'fillColor': COLOR_MUNI_GREEN,
            'color': COLOR_MUNI_GREEN,
            'weight': 2.5,
            'fillOpacity': 0.05,
        },
    ).add_to(m)

  if suburbs is not None:
    suburb_field = None
    for col in suburbs.columns:
      if any(
          k in col.lower()
          for k in ['suburb', 'name', 'ward', 'sub_name', 'label']
      ):
        suburb_field = col
        break

    tooltip_obj = (
        folium.GeoJsonTooltip(
            fields=[suburb_field],
            aliases=['Suburb/Ward:'],
            style=(
                'background-color: white; color: #002147; font-weight: bold;'
                ' font-size: 12px; padding: 4px;'
            ),
        )
        if suburb_field
        else None
    )

    folium.GeoJson(
        suburbs,
        name='🏡 Chinhoyi Suburbs & Wards',
        style_function=lambda x: {
            'fillColor': COLOR_MUNI_GOLD,
            'color': COLOR_CUT_NAVY,
            'weight': 1.8,
            'fillOpacity': 0.15,
            'dashArray': '3, 3',
        },
        tooltip=tooltip_obj,
    ).add_to(m)

  if density_points:
    HeatMap(
        density_points,
        name='🔥 Population Density (moc_census_addresses)',
        radius=14,
        blur=10,
        max_zoom=18,
        min_opacity=0.35,
        show=True,
    ).add_to(m)

  candidates_group = folium.FeatureGroup(
      name='📍 MCDA Proposed Streetlights', show=True
  )
  for _, row in df.iterrows():
    if row['rank'] <= 10:
      marker_color, radius = COLOR_PRIORITY_HIGH, 8.5
    elif row['rank'] <= 25:
      marker_color, radius = COLOR_PRIORITY_MED, 6.5
    else:
      marker_color, radius = COLOR_PRIORITY_LOW, 5.0

    popup_html = f"""
        <div style="font-family: Arial; width: 190px;">
            <h4 style="margin:0; color:{COLOR_CUT_NAVY};">Site #{row['rank']} ({row['candidate_id']})</h4>
            <hr style="margin:4px 0;">
            <b>MCDA Score:</b> {row['mcda_score']}<br>
            <b>Lat/Lon:</b> {row['lat']}, {row['lon']}<br>
            <b>UTM E/N:</b> {row['utm_easting']}, {row['utm_northing']}<br>
            <b>Bus Stop Dist:</b> {row['bus_dist_m']} m
        </div>
        """

    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=240),
        color='#ffffff',
        weight=1.5,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.9,
    ).add_to(candidates_group)

  candidates_group.add_to(m)

  legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 210px; z-index:9999; background-color: white;
                padding: 10px; border-radius: 6px; border: 1px solid #cbd5e1; font-family: Arial; font-size: 11px;">
        <b style="color: {COLOR_CUT_NAVY};">MCDA Streetlight Priorities</b><br>
        <i style="background: {COLOR_PRIORITY_HIGH}; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Rank 1–10 (High Priority / Red)<br>
        <i style="background: {COLOR_PRIORITY_MED}; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Rank 11–25 (Medium / Orange)<br>
        <i style="background: {COLOR_PRIORITY_LOW}; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Rank 26+ (Low Priority / Green)<br>
        <hr style="margin: 4px 0;">
        <b>Census Address Density</b>
        <span style="background: linear-gradient(to right, blue, green, yellow, red); width: 100%; height: 6px; display: block; border-radius: 2px; margin-top:2px;"></span>
    </div>
    """
  m.get_root().html.add_child(folium.Element(legend_html))
  folium.LayerControl(collapsed=False).add_to(m)
  return m


# ---------------------------------------------------------
# 6. REACTIVE PIPELINE & COORDINATE EXPORTS
# ---------------------------------------------------------
@pn.depends(
    w_bus.param.value_throttled,
    w_gap.param.value_throttled,
    w_dem.param.value_throttled,
)
def update_main_content(w1, w2, w3):
  scored_df = calculate_mcda(raw_candidates_df, w1, w2, w3)
  map_obj = create_map(scored_df, boundary_gdf, suburbs_gdf, census_coords)

  map_pane = pn.pane.HTML(
      map_obj._repr_html_(), height=530, sizing_mode='stretch_width'
  )
  table = pn.widgets.Tabulator(
      scored_df[[
          'rank',
          'candidate_id',
          'mcda_score',
          'lat',
          'lon',
          'utm_easting',
          'utm_northing',
          'bus_dist_m',
      ]],
      page_size=6,
      sizing_mode='stretch_width',
  )
  return pn.Column(
      map_pane,
      pn.pane.Markdown(
          '### 🏆 **Ranked Candidate Sites (Spatial Coordinates Included)**'
      ),
      table,
      sizing_mode='stretch_width',
  )


@pn.depends(
    w_bus.param.value_throttled,
    w_gap.param.value_throttled,
    w_dem.param.value_throttled,
)
def update_sidebar_kpis(w1, w2, w3):
  scored_df = calculate_mcda(raw_candidates_df, w1, w2, w3)
  avg_dist = int(scored_df.head(10)['bus_dist_m'].mean())

  kpi_card = f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
        <h4 style="margin: 0 0 10px 0; color: {COLOR_CUT_NAVY}; font-size: 13px; text-transform: uppercase;">📊 Spatial Analytics Summary</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 12px; color: #475569;">Total Streetlights:</span>
            <b style="color: {COLOR_MUNI_GREEN}; font-size: 14px;">{len(scored_df)}</b>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 12px; color: #475569;">High Priority (Red):</span>
            <b style="color: {COLOR_PRIORITY_HIGH}; font-size: 14px;">10</b>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 12px; color: #475569;">Avg Bus Stop Dist:</span>
            <b style="color: {COLOR_CUT_NAVY}; font-size: 14px;">{avg_dist} m</b>
        </div>
    </div>
    """
  return pn.pane.HTML(kpi_card, sizing_mode='stretch_width')


def get_csv_bytes():
  df = calculate_mcda(raw_candidates_df, w_bus.value, w_gap.value, w_dem.value)
  return io.BytesIO(df.to_csv(index=False).encode())


def get_geojson_bytes():
  df = calculate_mcda(raw_candidates_df, w_bus.value, w_gap.value, w_dem.value)
  geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
  gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
  return io.BytesIO(gdf.to_json().encode())


csv_download = pn.widgets.FileDownload(
    callback=get_csv_bytes,
    filename='chinhoyi_streetlights_coords.csv',
    button_type='success',
    label='📍 Export CSV (with Coords)',
    sizing_mode='stretch_width',
)

geojson_download = pn.widgets.FileDownload(
    callback=get_geojson_bytes,
    filename='chinhoyi_streetlights_coords.geojson',
    button_type='primary',
    label='🌐 Export GeoJSON (with Coords)',
    sizing_mode='stretch_width',
)

# ---------------------------------------------------------
# 7. LAYOUT & LAUNCHER
# ---------------------------------------------------------
header_html = f"""
<div style="background: linear-gradient(90deg, {COLOR_MUNI_GREEN} 0%, {COLOR_CUT_NAVY} 100%); padding: 14px 20px; border-radius: 6px; color: white; display: flex; align-items: center; justify-content: space-between;">
    <div>
        <h2 style="margin: 0; font-size: 19px; font-weight: bold; letter-spacing: 0.5px;">🏛️ MUNICIPALITY OF CHINHOYI</h2>
        <span style="font-size: 12px; opacity: 0.9;">GIS & Geoinformatics Section | Public Solar Lighting Optimization Dashboard</span>
    </div>
    <div style="text-align: right;">
        <span style="background-color: {COLOR_MUNI_GOLD}; color: black; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin-right: 6px;">
            AHP / MCDA Active
        </span>
        <span style="background-color: {COLOR_CUT_GOLD}; color: black; font-weight: bold; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
            CUT Geoinformatics
        </span>
    </div>
</div>
"""

sidebar = pn.Column(
    update_sidebar_kpis,
    pn.pane.Markdown('### ⚙️ **Interactive AHP Weights**'),
    w_bus,
    w_gap,
    w_dem,
    '---',
    pn.pane.Markdown('### 💾 **Data Export**'),
    csv_download,
    geojson_download,
    width=290,
)

dashboard = pn.Column(
    pn.pane.HTML(header_html, sizing_mode='stretch_width'),
    pn.Spacer(height=10),
    pn.Row(sidebar, update_main_content, sizing_mode='stretch_width'),
    sizing_mode='stretch_width',
)

if __name__ == '__main__':
  print('🚀 Launching Chinhoyi Solar Dashboard...')
  dashboard.show(title='Chinhoyi Solar Dashboard', host='0.0.0.0', port=5006)
else:
  dashboard.servable()