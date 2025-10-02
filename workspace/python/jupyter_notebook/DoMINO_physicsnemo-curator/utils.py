import os
import vtk
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def analyze_and_plot_distribution(data_array, field_name):
    """
    Calculates statistics and plots the distribution for a given 1D data array.

    Args:
        data_array (np.ndarray): A NumPy array of scalar values.
        field_name (str): The name of the field being analyzed (e.g., "Temperature").
    """
    if data_array.size == 0:
        print(f"\n--- No data found for '{field_name}'. Skipping analysis. ---\n")
        return

    data_array = data_array.flatten()

    print(f"\n{'---'*5} Analysis for: {field_name.upper()} {'---'*5}")
    print(f"Shape of aggregated data: {data_array.shape}")
    print(f"Total values calculated: {len(data_array)}")
    
    # Basic statistics
    stats = {
        'Min': np.min(data_array),
        'Max': np.max(data_array),
        'Mean': np.mean(data_array),
        'Std Dev': np.std(data_array)
    }
    for stat, value in stats.items():
        print(f"  {stat}: {value:.4f}")

    # Percentile distribution
    print("\n--- Percentile Distribution ---")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = np.percentile(data_array, percentiles)
    for p, v in zip(percentiles, percentile_values):
        print(f"  {p:2d}th percentile: {v:.4f}")
    print("***************************************\n")

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"Distribution for {field_name}", fontsize=20, y=0.98)

    # Plot 1: Histogram
    ax1.hist(data_array, bins=100, color="skyblue", edgecolor="black", alpha=0.8)
    ax1.set_title("Histogram")
    ax1.set_xlabel(field_name)
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Plot 2: Box plot
    ax2.boxplot(
        data_array,
        vert=False,
        whis=[5, 95],
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.3),
    )
    ax2.set_title("Box Plot")
    ax2.set_xlabel(field_name)
    ax2.set_yticks([])
    ax2.grid(True, linestyle='--', linewidth=0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def process_and_plot_directory(data_dir):
    """
    Loads all .npz files from a directory, processes only 'volume_fields' and 
    'surface_fields', aggregates the data, and plots a histogram for each field.
    """
    file_paths = glob.glob(os.path.join(data_dir, "*.npz"))

    if not file_paths:
        print(f"Error: No .npz files found in the directory '{data_dir}'.")
        return

    aggregated_data = defaultdict(list)
    target_keys = ['volume_fields', 'surface_fields']

    print(f"Found {len(file_paths)} files. Processing for keys: {target_keys}...")

    for file_path in file_paths:
        try:
            with np.load(file_path, allow_pickle=True) as data_dict:
                for key in target_keys:
                    if key not in data_dict:
                        continue
                    array = data_dict[key]

                    if not isinstance(array, np.ndarray):
                        print(f"Warning: Item '{key}' in {os.path.basename(file_path)} is not a NumPy array. Skipping.")
                        continue

                    if array.ndim == 1:
                        aggregated_data[key].extend(array)
                    elif array.ndim == 2:
                        num_cols = array.shape[1]
                        for i in range(num_cols):
                            col_name = f"{key}_col_{i}"
                            aggregated_data[col_name].extend(array[:, i])
                        
                        if key == 'volume_fields' and num_cols >= 3:
                            velocities = array[:, :3]
                            magnitudes = np.linalg.norm(velocities, axis=1)
                            aggregated_data[f"{key}_Magnitude"].extend(magnitudes)
                    else:
                        print(f"Warning: Array '{key}' has an unsupported dimension {array.ndim}. Skipping.")
        except Exception as e:
            print(f"Error loading or processing file {os.path.basename(file_path)}: {e}")

    print("\n--- All files processed. Generating plots for aggregated data. ---\n")

    if not aggregated_data:
        print(f"No data for keys {target_keys} was found in any files. Exiting.")
        return

    for field_name, data_list in aggregated_data.items():
        data_array = np.array(data_list)
        analyze_and_plot_distribution(data_array, field_name)


def get_vtp_bounds(vtk_filename: str):
    """
    Reads a .vtp file and returns its coordinate bounds.

    Args:
        vtk_filename: The full path to the .vtp file.

    Returns:
        A tuple containing the bounds (xmin, xmax, ymin, ymax, zmin, zmax),
        or None if the file cannot be read or is empty.
    """
    if not os.path.exists(vtk_filename) or os.path.getsize(vtk_filename) == 0:
        print(f"[WARNING] File is missing or empty, skipping: {vtk_filename}")
        return None

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()

    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"[WARNING] No data or points found in {vtk_filename}")
        return None

    return polydata.GetBounds()


def find_min_max_in_folders(base_dir="."):
    """
    Scans all subdirectories of base_dir that start with 'run_',
    reads all .vtp files inside them, and computes the global
    bounding box across all files.

    Args:
        base_dir: The base directory where the 'run_*' folders are located.

    Returns:
        Tuple (global_min, global_max, files_processed)
            global_min: [xmin, ymin, zmin]
            global_max: [xmax, ymax, zmax]
            files_processed: number of .vtp files processed
    """
    global_min = [float('inf'), float('inf'), float('inf')]
    global_max = [float('-inf'), float('-inf'), float('-inf')]
    files_processed = 0

    # Find all subfolders starting with "run_"
    for folder_name in os.listdir(base_dir):

        if not folder_name.startswith("run_"):
            continue

        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"[INFO] Processing folder: {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.endswith(".vtp"):
                vtp_path = os.path.join(folder_path, filename)
                bounds = get_vtp_bounds(vtp_path)

                if bounds:
                    files_processed += 1
                    lxmin, lxmax, lymin, lymax, lzmin, lzmax = bounds
                    global_min[0] = min(global_min[0], lxmin)
                    global_min[1] = min(global_min[1], lymin)
                    global_min[2] = min(global_min[2], lzmin)
                    global_max[0] = max(global_max[0], lxmax)
                    global_max[1] = max(global_max[1], lymax)
                    global_max[2] = max(global_max[2], lzmax)

    if files_processed == 0:
        print("\n[ERROR] No .vtp files were found or processed.")
        return None, None, 0

    return global_min, global_max, files_processed
