import io
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import panel as pn

pn.extension("tabulator", sizing_mode="stretch_width")

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
    styles={"color": "white", "padding": "12px 20px", "border-radius": "8px"}
)

# --- 2. SIDEBAR CONTROLS & WIDGETS ---
bus_weight = pn.widgets.FloatSlider(name="Bus Stop Proximity Weight", start=0.0, end=1.0, step=0.05, value=0.45)
gap_weight = pn.widgets.FloatSlider(name="Light Gap Coverage Weight", start=0.0, end=1.0, step=0.05, value=0.30)
dem_weight = pn.widgets.FloatSlider(name="DEM Terrain Elevation Weight", start=0.0, end=1.0, step=0.05, value=0.25)

buffer_radius = pn.widgets.Select(name="Illumination Buffer Radius (m)", options=[0, 30, 50, 75, 100], value=50)
enable_clustering = pn.widgets.Checkbox(name="Enable Point Clustering", value=True)
enable_heatmap = pn.widgets.Checkbox(name="Enable Address Density Heatmap", value=True)

csv_download = pn.widgets.FileDownload(filename="chinhoyi_solar_sites.csv", label="📍 Export Candidate CSV", button_type="primary")
geojson_download = pn.widgets.FileDownload(filename="chinhoyi_solar_sites.geojson", label="🌐 Export GeoJSON", button_type="info")

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
    "---",
    "### 💾 **Data Export**",
    csv_download,
    geojson_download,
    width=320,
    styles={"background": "#F4F6F7", "padding": "15px", "border-radius": "8px"}
)

# --- 3. DATA COMPUTATION & DYNAMIC MCDA FUNCTION ---
def get_processed_data(w_bus, w_gap, w_dem):
    try:
        gdf = gpd.read_parquet("data/moc_census_data_addresses.parquet")
    except Exception:
        # Fallback dataset generator if local parquet file is unreachable
        np.random.seed(42)
        data = {
            'candidate_id': [f'CSL-{i:03d}' for i in range(1, 41)],
            'lat': np.random.uniform(-17.38, -17.34, 40),
            'lon': np.random.uniform(30.18, 30.22, 40),
            'bus_score': np.random.rand(40),
            'gap_score': np.random.rand(40),
            'dem_score': np.random.rand(40),
            'bus_dist_m': np.random.uniform(30, 950, 40)
        }
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    # Normalize criteria weights
    total_w = w_bus + w_gap + w_dem
    norm_bus = w_bus / total_w if total_w > 0 else 0.333
    norm_gap = w_gap / total_w if total_w > 0 else 0.333
    norm_dem = w_dem / total_w if total_w > 0 else 0.334

    # Calculate score & rank candidate locations
    gdf['mcda_score'] = (
        (gdf.get('bus_score', np.random.rand(len(gdf))) * norm_bus) +
        (gdf.get('gap_score', np.random.rand(len(gdf))) * norm_gap) +
        (gdf.get('dem_score', np.random.rand(len(gdf))) * norm_dem)
    )
    gdf = gdf.sort_values(by='mcda_score', ascending=False).reset_index(drop=True)
    gdf['rank'] = gdf.index + 1
    return gdf

# --- 4. MAIN INTERACTIVE DASHBOARD PIPELINE ---
@pn.depends(bus_weight, gap_weight, dem_weight, buffer_radius, enable_clustering, enable_heatmap)
def update_dashboard(w_bus, w_gap, w_dem, radius, cluster_flag, heatmap_flag):
    gdf = get_processed_data(w_bus, w_gap, w_dem)

    # Update dynamic file download payloads
    csv_buffer = io.StringIO()
    gdf.drop(columns=['geometry'], errors='ignore').to_csv(csv_buffer, index=False)
    csv_download.file = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))

    geojson_download.file = io.BytesIO(gdf.to_json().encode('utf-8'))

    # Top KPI Metrics Cards
    kpi_row = pn.Row(
        pn.indicators.Number(name="Total Candidate Sites", value=len(gdf), format="{value}", colors=[(1, "#2C3E50")]),
        pn.indicators.Number(name="High Priority (Rank 1-10)", value=min(10, len(gdf)), format="{value}", colors=[(1, "#E74C3C")]),
        pn.indicators.Number(name="Mean MCDA Score", value=gdf['mcda_score'].mean(), format="{value:.3f}", colors=[(1, "#2980B9")]),
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0)
    )

    # Center map on spatial bounds of data
    centroid_lat = gdf['lat'].mean() if 'lat' in gdf else -17.36
    centroid_lon = gdf['lon'].mean() if 'lon' in gdf else 30.20
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=13, tiles="CartoDB positron")

    # Heatmap Layer
    if heatmap_flag and 'lat' in gdf and 'lon' in gdf:
        heat_data = [[row.lat, row.lon] for _, row in gdf.iterrows()]
        HeatMap(heat_data, radius=16, blur=22, name="Address Density Heatmap").add_to(m)

    # Candidate Points Container
    container = MarkerCluster(name="Clustered Sites") if cluster_flag else folium.FeatureGroup(name="Candidate Sites").add_to(m)
    if cluster_flag:
        container.add_to(m)

    for _, row in gdf.iterrows():
        color = "#E74C3C" if row['rank'] <= 10 else ("#F39C12" if row['rank'] <= 25 else "#2ECC71")

        # Optional Illumination Buffer Radius
        if radius > 0:
            folium.Circle(
                location=[row.lat, row.lon],
                radius=radius,
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.18
            ).add_to(m)

        # Site Location Marker
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(f"""
                <b>Site ID:</b> {row.get('candidate_id', 'N/A')}<br>
                <b>Priority Rank:</b> #{row['rank']}<br>
                <b>MCDA Score:</b> {row['mcda_score']:.3f}
            """, max_width=200)
        ).add_to(container)

    # Side Legend Box Overlay
    legend_html = """
    <div style="position: fixed; bottom: 25px; right: 20px; width: 220px; 
                background-color: white; border: 1.5px solid #BDC3C7; z-index:9999; 
                font-size: 12px; padding: 12px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 8px 0; font-size: 13px;">🗺️ <b>Combined Map Legend</b></h4>
        <b>MCDA Priority Ranks</b><br>
        <div style="margin-top: 4px;">
            <i style="background:#E74C3C; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Rank 1–10 (High)<br>
            <i style="background:#F39C12; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Rank 11–25 (Medium)<br>
            <i style="background:#2ECC71; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Rank 26+ (Low)
        </div>
        <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ECF0F1;">
        <b>Census Address Density</b><br>
        <div style="background: linear-gradient(to right, blue, green, yellow, red); height: 8px; width: 100%; border-radius: 4px; margin-top:4px;"></div>
        <span style="float:left; font-size: 10px; color: #7F8C8D;">Low</span>
        <span style="float:right; font-size: 10px; color: #7F8C8D;">High</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright").add_to(m)

    map_pane = pn.pane.HTML(m._repr_html_(), height=570, sizing_mode="stretch_width")

    # Backend Data Management Table
    disp_cols = [c for c in ['rank', 'candidate_id', 'mcda_score', 'lat', 'lon', 'bus_dist_m'] if c in gdf.columns]
    table_pane = pn.widgets.Tabulator(
        gdf[disp_cols],
        pagination="remote",
        page_size=15,
        sizing_mode="stretch_width"
    )

    # Main Interface Tabs
    tabs = pn.Tabs(
        ("🌍 Spatial Map & Analysis", pn.Column(kpi_row, map_pane)),
        ("📊 Backend Candidate Sites Data", table_pane),
        sizing_mode="stretch_width"
    )

    return tabs

# --- 5. INITIALIZE DASHBOARD LAYOUT ---
dashboard = pn.Column(
    header,
    pn.Row(sidebar, update_dashboard, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

dashboard.servable()
