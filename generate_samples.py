import argparse
import random
import numpy as np
import pandas as pd
from codebase.utils import save_samples
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
import random
from typing import List, Tuple
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y_MIN = -40.98
Y_MAX = -12.983
X_MIN = -79.805
X_MAX = -62.402

def generate_circular_subsets(df: pd.DataFrame, 
                              x_col = 'x', y_col = 'y', z_col = 'z', t_col = 't', m_col = 'm',
                              xlims: Tuple[float, float] = None, ylims: Tuple[float, float] = None,
                              radius_km: float = 10.0,
                              num_subsets: int = 5,
                              min_points_per_subset: int = 1,
                              t_start: float = 0
                              ) -> List[pd.DataFrame]:
    """
    Inputs:
    df : pd.DataFrame
    xlims : (min_lon, max_lon)
    ylims : (min_lat, max_lat)
    radius_km : float
    num_subsets : int
    min_points_per_subset : int

    Output: List[pd.DataFrame] cada DataFrame contiene eventos dentro de un circunferencia aleatoria
    """

    # Bounding box
    if xlims is None:
        xlims = (df[x_col].min(), df[x_col].max())
    if ylims is None:
        ylims = (df[y_col].min(), df[y_col].max())
    
    # Crea Shapely
    points_geo = [Point(lon, lat) for lon, lat in zip(df[x_col], df[y_col])]
    
    # Utiliza el centro para determinar la zona UTM apropiada
    center_lon = (xlims[0] + xlims[1]) / 2
    center_lat = (ylims[0] + ylims[1]) / 2
    
    # Define y proyecta a zona UTM 
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'

    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
    utm = pyproj.CRS(f'+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    project_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    points_utm = [transform(project_to_utm, point) for point in points_geo]
    
    output = []
    attempts = 0
    max_attempts = num_subsets * 10  # Cota superior
    
    while len(output) < num_subsets and attempts < max_attempts:
        attempts += 1
        
        # Generar un punto central aleatorio dentro de los límites
        center_lon = random.uniform(xlims[0], xlims[1])
        center_lat = random.uniform(ylims[0], ylims[1])
        center_geo = Point(center_lon, center_lat)
        
        # Transformar centro a zona UTM
        center_utm = transform(project_to_utm, center_geo)
        
        # Crea la circunferencia
        radius_m = radius_km * 1000
        circle_utm = center_utm.buffer(radius_m)
        
        # Filtra
        indices_in_circle = []
        for i, point_utm in enumerate(points_utm):
            if circle_utm.contains(point_utm):
                indices_in_circle.append(i)
        
        if len(indices_in_circle) >= min_points_per_subset:
            subset_df = df.iloc[indices_in_circle].copy()
            
            # Tiempo relativo al t_start
            subset_df[t_col] = subset_df[t_col] - t_start
            
            output.append(subset_df)
    
    if len(output) < num_subsets:
        print(f"Warning: Only generated {len(output)} subsets out of {num_subsets} requested.")
        print(f"Consider increasing the area bounds, decreasing radius, or lowering min_points_per_subset.")
    
    return output

def samples(filepath, samples=10000, radius_km=100, time_intervals=100, random_seed = 28):
    '''
    Genera *samples* muestras tomadas al azar de radio 10 dentro del bounding box
    los intervalos de tiempo son disjuntos
    en cada intervalo de tiempo puede existir interseccion
    output: pd.DataFrame: ['x', 'y', 'z', 't', 'm', 'idx_rad', 'idx_time']
    '''
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    df = pd.read_csv(filepath)
    df['time'] = pd.to_numeric(pd.to_datetime(df["time"])) / 10**9 # ns a s
    df = df[['longitude', 'latitude', 'depth', 'time', 'mag']]
    df.columns = ['x', 'y', 'z', 't', 'm']
    T_MIN = df['t'].min()
    T_MAX = df['t'].max()

    # Intervalos de tiempo
    time_edges = np.linspace(T_MIN, T_MAX, time_intervals + 1)
    samples_per_interval = samples // time_intervals
    remaining_samples = samples % time_intervals
    
    output = pd.DataFrame(columns=['x', 'y', 'z', 't', 'm','idx_rad', 'idx_time'])
    
    for i in range(time_intervals):
        t_start = time_edges[i]
        t_end = time_edges[i + 1]
        
        # filtrado
        time_mask = (df['t'] >= t_start) & (df['t'] < t_end)
        df_interval = df[time_mask].copy()
        
        if len(df_interval) <= 1:
            print(f"Advertencia: No hay datos suficientes en el intervalo de tiempo {i+1}/{time_intervals}")
            continue
        else:
            # distribuye equitativamente las muestras restantes
            n_samples = samples_per_interval + (1 if i < remaining_samples else 0)

            # subconjuntos para este intervalo de tiempo
            interval_subsets = generate_circular_subsets(
                df=df_interval,
                xlims=(X_MIN, X_MAX),
                ylims=(Y_MIN, Y_MAX),
                radius_km=radius_km,
                num_subsets=n_samples,
                min_points_per_subset=2,
                t_start = t_start
            )

            interval_subsets['idx_time'] = i
            output = pd.concat([output, interval_subsets], ignore_index=True)
    
    print(f"Generados {len(output)} subconjuntos en total a través de {time_intervals} intervalos de tiempo")
    return output

def main():
    parser = argparse.ArgumentParser(description="Generar muestras")
    parser.add_argument('filepath', help='Ruta al archivo CSV con datos de terremotos')
    parser.add_argument('samples', type=int, help='Número de muestras a generar')
    parser.add_argument('radius_km', type=int, help='Radio en kilómetros para subconjuntos circulares')
    parser.add_argument('time_intervals', type=int, help='Número de intervalos de tiempo en los que dividir los datos')
    args = parser.parse_args()

    # Generar subconjuntos
    subsets = samples(
        filepath=args.filepath,
        samples=args.samples,
        radius_km=args.radius_km,
        time_intervals=args.time_intervals
    )

    # Guardar subconjuntos en archivo
    output_filename = f'{args.filepath.split(".")[0]}_{args.samples}_{args.radius_km}_{args.time_intervals}.pt'
    print(f"Generados {len(subsets)} subconjuntos")
    save_samples(output_filename, subsets)
    print(f"El nombre del archivo de salida sería: {output_filename}")

if __name__ == "__main__":
    import sys
    
    # Verificar si se proporcionan argumentos de línea de comandos
    if len(sys.argv) > 1:
        main()
    else:
        print("No se proporcionaron argumentos de línea de comandos.")