import io
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import panel as pn

# Initialize Panel extension safely
pn.extension()

# --- 1. HEADER & BRANDING LOGOS ---
header = pn.Row(
    pn.pane.Markdown(
        """
        # ☀️ **Chinhoyi Solar Streetlight Decision Support System**
        **Joint Initiative: Municipality of Chinhoyi & Chinhoyi University of Technology (CUT)**
        """,
        sizing_mode="stretch_width"
    ),
    pn.Row(
        pn.pane.HTML('<img src="https://img.icons8.com/color/96/university.png" width="55" title="CUT Logo"/>'),
        pn.pane.HTML('<img src="https://img.icons8.com/color/96/city-hall.png" width="55" title="Municipality of Chinhoyi"/>'),
        align="center",
        margin=(0, 15)
    ),
    background="#1B2631",
    styles={"color": "white", "padding": "12px 20px", "border-radius": "8px"},
    sizing_mode="stretch_width"
)

# --- 2. SIDEBAR CONTROLS & WIDGETS ---
bus_weight = pn.widgets.FloatSlider(name="Bus Stop Proximity Weight", start=0.0, end=1.0, step=0.05, value=0.45)
gap_weight = pn.widgets.FloatSlider(name="Light Gap Coverage Weight", start=0.0, end=1.0, step=0.05, value=0.30)
dem_weight = pn.widgets.FloatSlider(name="DEM Terrain Elevation Weight", start=0.0, end=1.0, step=0.05, value=0.25)

buffer_radius = pn.widgets.Select(name="Illumination Buffer Radius (m)", options=[0, 30, 50, 75, 100], value=50)
enable_clustering = pn.widgets.Checkbox(name="Enable Point Clustering", value=True)
enable_heatmap = pn.widgets.Checkbox(name="Enable Address Density Heatmap", value=True)

sidebar = pn.Column(
    "### 🎛️ **Dynamic MCDA Weights**",
    bus_weight,
    gap_weight,
    dem_weight,
    "---",
    "### 🛠️ **Spatial Map Layers**",
    buffer_radius,
    enable_clustering,
    enable_heatmap,
    width=300,
    styles={"background": "#F4F6F7", "padding": "15px", "border-radius": "8px"}
)

# --- 3. DATA PROCESSING PIPELINE ---
def load_and_score_data(w_bus, w_gap, w_dem):
    try:
        gdf = gpd.read_parquet("data/moc_census_data_addresses.parquet")
    except Exception:
        np.random.seed(42)
        data = {
            'candidate_id': [f'CSL-{i:03d}' for i in range(1, 31)],
            'lat': np.random.uniform(-17.38, -17.34, 30),
            'lon': np.random.uniform(30.18, 30.22, 30),
            'bus_score': np.random.rand(30),
            'gap_score': np.random.rand(30),
            'dem_score': np.random.rand(30),
            'bus_dist_m': np.random.uniform(30, 950, 30)
        }
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    total_w = w_bus + w_gap + w_dem
    norm_bus = w_bus / total_w if total_w > 0 else 0.333
    norm_gap = w_gap / total_w if total_w > 0 else 0.333
    norm_dem = w_dem / total_w if total_w > 0 else 0.334

    bus_s = gdf['bus_score'] if 'bus_score' in gdf else np.random.rand(len(gdf))
    gap_s = gdf['gap_score'] if 'gap_score' in gdf else np.random.rand(len(gdf))
    dem_s = gdf['dem_score'] if 'dem_score' in gdf else np.random.rand(len(gdf))

    gdf['mcda_score'] = (bus_s * norm_bus) + (gap_s * norm_gap) + (dem_s * norm_dem)
    gdf = gdf.sort_values(by='mcda_score', ascending=False).reset_index(drop=True)
    gdf['rank'] = gdf.index + 1
    return gdf

# --- 4. DYNAMIC VIEW UPDATER ---
@pn.depends(bus_weight, gap_weight, dem_weight, buffer_radius, enable_clustering, enable_heatmap)
def update_view(w_bus, w_gap, w_dem, radius, cluster_flag, heatmap_flag):
    gdf = load_and_score_data(w_bus, w_gap, w_dem)

    # Base Folium Map
    c_lat = gdf['lat'].mean() if 'lat' in gdf else -17.36
    c_lon = gdf['lon'].mean() if 'lon' in gdf else 30.20
    m = folium.Map(location=[c_lat, c_lon], zoom_start=13, tiles="CartoDB positron")

    # Add Address Heatmap
    if heatmap_flag and 'lat' in gdf and 'lon' in gdf:
        heat_data = [[row.lat, row.lon] for _, row in gdf.iterrows()]
        HeatMap(heat_data, radius=15, blur=20, name="Address Heatmap").add_to(m)

    # Add Cluster/Markers
    container = MarkerCluster(name="Candidate Sites") if cluster_flag else m

    for _, row in gdf.iterrows():
        color = "#E74C3C" if row['rank'] <= 10 else ("#F39C12" if row['rank'] <= 25 else "#2ECC71")

        if radius > 0:
            folium.Circle(
                location=[row.lat, row.lon],
                radius=radius,
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.15
            ).add_to(m)

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"<b>Site:</b> {row.get('candidate_id', 'CSL')}<br><b>Rank:</b> #{row['rank']}<br><b>Score:</b> {row['mcda_score']:.3f}"
        ).add_to(container)

    # Unified Legend Overlay
    legend_html = """
    <div style="position: fixed; bottom: 25px; right: 20px; width: 210px; 
                background-color: white; border: 1px solid #BDC3C7; z-index:9999; 
                font-size: 11px; padding: 10px; border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.15);">
        <b>🗺️ Combined Legend</b><br><br>
        <b>MCDA Ranks</b><br>
        <i style="background:#E74C3C; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Rank 1–10 (High)<br>
        <i style="background:#F39C12; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Rank 11–25 (Medium)<br>
        <i style="background:#2ECC71; width:10px; height:10px; display:inline-block; border-radius:50%;"></i> Rank 26+ (Low)
        <hr style="margin: 6px 0;">
        <b>Address Density</b><br>
        <div style="background: linear-gradient(to right, blue, green, yellow, red); height: 8px; width: 100%;"></div>
        <span style="float:left;">Low</span><span style="float:right;">High</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render components
    map_pane = pn.pane.HTML(m._repr_html_(), height=550, sizing_mode="stretch_width")
    
    # Simple DataFrame Pane (prevents JavaScript/Tabulator errors)
    disp_cols = [c for c in ['rank', 'candidate_id', 'mcda_score', 'lat', 'lon', 'bus_dist_m'] if c in gdf.columns]
    table_pane = pn.pane.DataFrame(gdf[disp_cols], height=500, sizing_mode="stretch_width")

    return pn.Tabs(
        ("🌍 Spatial Map & Analysis", map_pane),
        ("📊 Backend Candidate Sites Data", table_pane),
        sizing_mode="stretch_width"
    )

# --- 5. APP LAYOUT ---
app = pn.Column(
    header,
    pn.Row(sidebar, update_view, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

app.servable()
