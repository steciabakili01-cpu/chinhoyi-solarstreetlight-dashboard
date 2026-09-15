import geopandas as gpd

def create_coverage_buffers(gdf: gpd.GeoDataFrame, buffer_radius_meters: float = 30.0) -> gpd.GeoDataFrame:
    """
    Generates circular coverage buffer zones around light point locations.
    Reprojects temporarily to a metric projected CRS (UTM Zone 36S for Chinhoyi, Zimbabwe)
    for accurate distance metrics before returning to EPSG:4326.
    """
    if gdf.empty:
        return gdf
    
    # Project to UTM 36S (EPSG:32736) for accurate meter-based buffer calculation
    gdf_projected = gdf.to_crs(epsg=32736)
    gdf_projected['geometry'] = gdf_projected.buffer(buffer_radius_meters)
    
    # Reproject back to standard WGS84 for web mapping (Folium)
    return gdf_projected.to_crs(epsg=4326)

def calculate_priority_score(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates a basic priority score for streetlight maintenance or placement
    based on operational status and light coverage.
    """
    if gdf.empty:
        return gdf
    
    # Example logic: assign priority based on status column if present
    if 'status' in gdf.columns:
        gdf['priority'] = gdf['status'].apply(
            lambda x: 'High' if str(x).lower() in ['faulty', 'broken', 'offline'] else 'Low'
        )
    else:
        gdf['priority'] = 'Normal'
        
    return gdf
