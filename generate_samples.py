import argparse
import numpy as np
import pandas as pd
from codebase.subsets import generate_circular_subsets
from codebase.utils import save
import torch
from typing import List
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y_MIN = -40.98
Y_MAX = -12.983
X_MIN = -79.805
X_MAX = -62.402

def samples(filepath, samples=10000, radius_km=100, time_intervals=100):
    '''
    Genera *samples* muestras tomadas al azar de radio 10 dentro del bounding box
    output: list[pd.DataFrame]
    '''
    df = pd.read_csv(filepath)
    df['time'] = pd.to_numeric(pd.to_datetime(df["time"])) / 10**9 # ns a s
    df = df[['longitude', 'latitude', 'depth', 'time', 'mag']]
    df.columns = ['x', 'y', 'z', 't', 'm']
    T_MIN = df['t'].min()
    T_MAX = df['t'].max()

    subsets = []

    # Intervalos de tiempo
    time_edges = np.linspace(T_MIN, T_MAX, time_intervals + 1)
    samples_per_interval = samples // time_intervals
    remaining_samples = samples % time_intervals
    
    subsets = []
    
    for i in range(time_intervals):
        t_start = time_edges[i]
        t_end = time_edges[i + 1]
        
        # filtrado
        time_mask = (df['t'] >= t_start) & (df['t'] < t_end)
        df_interval = df[time_mask].copy()
        
        if len(df_interval) <= 1:
            print(f"Advertencia: No hay datos suficientes en el intervalo de tiempo {i+1}/{time_intervals}")
            continue
        
        n_samples = samples_per_interval + (1 if i < remaining_samples else 0)
        
        # subconjuntos para este intervalo de tiempo
        if len(df_interval) >= 2:
            interval_subsets = generate_circular_subsets(
                df=df_interval,
                xlims=(X_MIN, X_MAX),
                ylims=(Y_MIN, Y_MAX),
                radius_km=radius_km,
                num_subsets=n_samples,
                min_points_per_subset=2,
                random_seed=None,
                t_start = t_start
            )
            subsets.extend(interval_subsets)

            # Agregar metadatos del intervalo de tiempo a cada subconjunto
            for subset in interval_subsets:
                subset.attrs.update({
                    'time_interval': i + 1,
                    'time_start': t_start,
                    'time_end': t_end,
                    'total_time_intervals': time_intervals
                })
    
    print(f"Generados {len(subsets)} subconjuntos en total a través de {time_intervals} intervalos de tiempo")
    print(f"Promedio de subconjuntos por intervalo: {len(subsets) / time_intervals:.1f}")

    return subsets

def show_subset_metadata_example(subsets: list[pd.DataFrame]) -> None:
    '''
    Mostrar qué metadatos están almacenados en subset.attrs
    '''
    if not subsets:
        print("No hay subconjuntos disponibles")
        return
        
    print("=== ESTRUCTURA DE METADATOS DEL SUBCONJUNTO ===")
    print("Tipo de subconjuntos:", type(subsets))
    print("Tipo de cada subconjunto:", type(subsets[0]) if subsets else "N/A")
    print()
    
    for i, subset in enumerate(subsets[:3]):  # Mostrar los primeros 3 ejemplos
        print(f"--- Subconjunto {i+1} ---")
        print("Forma del DataFrame:", subset.shape)
        print("Columnas del DataFrame:", list(subset.columns))
        print("Tipo de attrs:", type(subset.attrs))
        print("Contenidos de attrs:")
        for key, value in subset.attrs.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        print()
        
        # Mostrar cómo acceder a metadatos específicos
        print("Ejemplo de acceso:")
        print(f"  Centro del círculo: ({subset.attrs.get('circle_center_lon')}, {subset.attrs.get('circle_center_lat')})")
        print(f"  Intervalo de tiempo: {subset.attrs.get('time_interval')}")
        print(f"  Rango de tiempo: {subset.attrs.get('time_start')} a {subset.attrs.get('time_end')}")
        print(f"  Radio: {subset.attrs.get('radius_km')} km")
        print(f"  Puntos en el subconjunto: {subset.attrs.get('num_points')}")
        print("=" * 50)

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
    print(f"El nombre del archivo de salida sería: {output_filename}")
    save(output_filename, subsets)

def run_test_example():
    """Ejecutar ejemplo de prueba cuando no se proporcionan argumentos de línea de comandos"""
    test_file = 'data/test_earthquakes.csv'
    
    # Verificar si el archivo de prueba existe
    try:
        # Leer archivo de prueba para obtener rangos de datos para depuración
        test_df = pd.read_csv(test_file)
        print(f"Usando archivo de prueba existente: {test_file}")
        
        # Usar la función samples()
        print("\n=== Usando la función samples() ===")
        subsets = samples(
            filepath=test_file,
            samples=50,           # Generar 50 subconjuntos circulares
            radius_km=100,        # Círculos de radio 100km  
            time_intervals=5      # A través de 5 intervalos de tiempo
        )
        
        print(f"\nGenerados {len(subsets)} subconjuntos")
        
        # Mostrar metadatos detallados para los primeros subconjuntos
        print("\n" + "="*60)
        show_subset_metadata_example(subsets)
        
        # Mostrar cómo acceder a datos de subconjuntos individuales
        if subsets:
            print(f"\n=== Ejemplo: Accediendo al primer subconjunto ===")
            first_subset = subsets[0]
            print(f"Forma del subconjunto: {first_subset.shape}")
            print(f"Rango de tiempo en el subconjunto: {first_subset['t'].min():.0f} a {first_subset['t'].max():.0f}")
            print(f"Rango de magnitud: {first_subset['m'].min():.2f} a {first_subset['m'].max():.2f}")
            print(f"Centro geográfico: ({first_subset.attrs['circle_center_lon']:.3f}, {first_subset.attrs['circle_center_lat']:.3f})")
            print(f"Intervalo de tiempo: {first_subset.attrs['time_interval']}/{first_subset.attrs['total_time_intervals']}")
        
        # También imprimir información de depuración
        print(f"\n=== INFORMACIÓN DE DEPURACIÓN ===")
        print(f"Caja delimitadora: X({X_MIN}, {X_MAX}), Y({Y_MIN}, {Y_MAX})")
        print(f"Rangos de datos de prueba: lon({test_df['longitude'].min():.2f}, {test_df['longitude'].max():.2f}), lat({test_df['latitude'].min():.2f}, {test_df['latitude'].max():.2f})")
        
    except FileNotFoundError:
        print(f"Archivo de prueba {test_file} no encontrado. Por favor créalo primero o proporciona argumentos de línea de comandos.")

if __name__ == "__main__":
    import sys
    
    # Verificar si se proporcionan argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Usar interfaz de línea de comandos
        main()
    else:
        # Ejecutar ejemplo de prueba
        print("No se proporcionaron argumentos de línea de comandos. Ejecutando ejemplo de prueba...")
        run_test_example()