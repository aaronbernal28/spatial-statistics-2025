import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
import random
from typing import List, Tuple, Optional

def generate_circular_subsets(
    df: pd.DataFrame, 
    x_col: str = 'x', 
    y_col: str = 'y', 
    z_col: str = 'z', 
    t_col: str = 't', 
    m_col: str = 'm',
    xlims: Tuple[float, float] = None,
    ylims: Tuple[float, float] = None,
    radius_km: float = 10.0,
    num_subsets: int = 5,
    min_points_per_subset: int = 1,
    random_seed: Optional[int] = None,
    t_start: float = 0
) -> List[pd.DataFrame]:
    """
    Generate random circular subsets from a pandas DataFrame with lat/lon coordinates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the points with x, y, z, t, m columns
    xlims : tuple
        Longitude limits (min_lon, max_lon). If None, uses data bounds
    ylims : tuple
        Latitude limits (min_lat, max_lat). If None, uses data bounds
    radius_km : float
        Radius of circular subsets in kilometers (default: 10.0)
    num_subsets : int
        Number of random circular subsets to generate (default: 5)
    min_points_per_subset : int
        Minimum number of points required in each subset (default: 1)
    random_seed : int, optional
        Random seed for reproducibility
    t_start : float

    Returns:
    --------
    List[pd.DataFrame]
        List of DataFrames, each containing points within a random circle
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Validate input columns
    required_cols = [x_col, y_col, z_col, t_col, m_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    if df.empty:
        return []
    
    # Set bounds
    if xlims is None:
        xlims = (df[x_col].min(), df[x_col].max())
    if ylims is None:
        ylims = (df[y_col].min(), df[y_col].max())
    
    # Create Shapely points from lat/lon
    points_geo = [Point(lon, lat) for lon, lat in zip(df[x_col], df[y_col])]
    
    # Set up coordinate transformation (WGS84 to UTM for accurate distance calculations)
    # Use the center of the data to determine appropriate UTM zone
    center_lon = (xlims[0] + xlims[1]) / 2
    center_lat = (ylims[0] + ylims[1]) / 2
    
    # Determine UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # Define projections
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
    utm = pyproj.CRS(f'+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    
    # Create transformation functions
    project_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    
    # Transform all points to UTM for accurate distance calculations
    points_utm = [transform(project_to_utm, point) for point in points_geo]
    
    subsets = []
    attempts = 0
    max_attempts = num_subsets * 10  # Prevent infinite loops
    
    while len(subsets) < num_subsets and attempts < max_attempts:
        attempts += 1
        
        # Generate random center point within bounds
        center_lon = random.uniform(xlims[0], xlims[1])
        center_lat = random.uniform(ylims[0], ylims[1])
        center_geo = Point(center_lon, center_lat)
        
        # Transform center to UTM
        center_utm = transform(project_to_utm, center_geo)
        
        # Create circle in UTM coordinates (radius in meters)
        radius_m = radius_km * 1000
        circle_utm = center_utm.buffer(radius_m)
        
        # Find points within the circle
        indices_in_circle = []
        for i, point_utm in enumerate(points_utm):
            if circle_utm.contains(point_utm):
                indices_in_circle.append(i)
        
        # Check if we have enough points
        if len(indices_in_circle) >= min_points_per_subset:
            subset_df = df.iloc[indices_in_circle].copy()
            
            # Transform coordinates to be relative to center and t_start
            subset_df[x_col] = subset_df[x_col] - center_lon
            subset_df[y_col] = subset_df[y_col] - center_lat
            subset_df[t_col] = subset_df[t_col] - t_start
            
            # Create reference point as first row with meaningful values
            reference_point = pd.DataFrame({
                x_col: [center_lon],  # Center becomes origin
                y_col: [center_lat],  # Center becomes origin
                z_col: [0],  # Use median z value from subset
                t_col: [0],  # t_start becomes origin
                m_col: [0]   # Use median m value from subset
            })
            
            # Concatenate reference point as first row with the rest of the data
            subset_df = pd.concat([reference_point, subset_df], ignore_index=True)
            
            # Add metadata about the circle
            subset_df.attrs = {
                'circle_center_lon': center_lon,
                'circle_center_lat': center_lat,
                'radius_km': radius_km,
                'num_points': len(indices_in_circle) + 1,  # +1 for reference point
                't_start': t_start
            }
            
            subsets.append(subset_df)
    
    if len(subsets) < num_subsets:
        print(f"Warning: Only generated {len(subsets)} subsets out of {num_subsets} requested.")
        print(f"Consider increasing the area bounds, decreasing radius, or lowering min_points_per_subset.")
    
    return subsets

def plot_subsets_summary(subsets: List[pd.DataFrame], 
                        x_col: str = 'x', 
                        y_col: str = 'y') -> None:
    """
    Print summary information about the generated subsets.
    
    Parameters:
    -----------
    subsets : List[pd.DataFrame]
        List of subset DataFrames
    x_col : str
        Column name for longitude
    y_col : str
        Column name for latitude
    """
    print(f"Generated {len(subsets)} circular subsets:")
    print("-" * 50)
    
    for i, subset in enumerate(subsets):
        attrs = getattr(subset, 'attrs', {})
        center_lon = attrs.get('circle_center_lon', 'Unknown')
        center_lat = attrs.get('circle_center_lat', 'Unknown')
        radius_km = attrs.get('radius_km', 'Unknown')
        t_start = attrs.get('t_start', 'Unknown')
        
        print(f"Subset {i+1}:")
        print(f"  Circle center: ({center_lon:.4f}, {center_lat:.4f})")
        print(f"  t_start: {t_start:.0f}")
        print(f"  Radius: {radius_km} km")
        print(f"  Points: {len(subset)} (including reference point)")
        if len(subset) > 0:
            print(f"  x range: [{subset[x_col].min():.4f}, {subset[x_col].max():.4f}]")
            print(f"  y range: [{subset[y_col].min():.4f}, {subset[y_col].max():.4f}]")
            print(f"  First row (reference): x={subset[x_col].iloc[0]:.4f}, y={subset[y_col].iloc[0]:.4f}, z={subset['z'].iloc[0]:.4f}, t={subset['t'].iloc[0]:.0f}, m={subset['m'].iloc[0]:.4f}")
            if len(subset) > 1:
                print(f"  Second row (first data): x={subset[x_col].iloc[1]:.4f}, y={subset[y_col].iloc[1]:.4f}, z={subset['z'].iloc[1]:.4f}, t={subset['t'].iloc[1]:.0f}, m={subset['m'].iloc[1]:.4f}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    
    # Generate random points around a central location (e.g., around San Francisco)
    center_lat, center_lon = 37.7749, -122.4194
    
    sample_data = pd.DataFrame({
        'x': np.random.normal(center_lon, 0.1, n_points),  # longitude
        'y': np.random.normal(center_lat, 0.1, n_points),  # latitude  
        'z': np.random.uniform(0, 1000, n_points),         # elevation
        't': pd.date_range('2023-01-01', periods=n_points, freq='1h'),  # time
        'm': np.random.exponential(2.0, n_points)          # magnitude
    })
    sample_data['t'] = pd.to_numeric(pd.to_datetime(sample_data['t'])) / 10**9
    
    # Generate circular subsets
    subsets = generate_circular_subsets(
        df=sample_data,
        xlims=(-122.6, -122.2),  # longitude bounds
        ylims=(37.6, 37.9),      # latitude bounds
        radius_km=10.0,          # 10 km radius
        num_subsets=9,           # generate 9 subsets
        min_points_per_subset=10, # at least 10 points per subset
        random_seed=42,
        t_start=sample_data['t'].iloc[0]+100  # Use first timestamp as t_start
    )
    
    # Display results
    plot_subsets_summary(subsets)