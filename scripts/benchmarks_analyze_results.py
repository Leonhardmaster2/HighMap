import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------

def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["benchmarks"]


# ------------------------------------------------------------
# Parse benchmark name
# Example:
# BM_Convolution_CPU/512/5
# → base = BM_Convolution
# → variant = CPU
# ------------------------------------------------------------

def parse_entry(name):
    parts = name.split("/")
    if len(parts) < 3:
        return None

    base_variant = parts[0]

    # split CPU / GPU
    if "_CPU" in base_variant:
        base = base_variant.replace("_CPU", "")
        variant = "CPU"
    elif "_GPU" in base_variant:
        base = base_variant.replace("_GPU", "")
        variant = "GPU"
    else:
        return None

    try:
        size = int(parts[1])
        radius = int(parts[2])
    except:
        return None

    return base, variant, size, radius


# ------------------------------------------------------------
# Group data
# structure:
# grouped[base]["CPU"][(size, r)] = time
# ------------------------------------------------------------

def group_benchmarks(benchmarks):
    grouped = defaultdict(lambda: {"CPU": {}, "GPU": {}})

    for b in benchmarks:
        parsed = parse_entry(b["name"])
        if not parsed:
            continue

        base, variant, size, radius = parsed
        time = b["real_time"]

        grouped[base][variant][(size, radius)] = time

    return grouped


# ------------------------------------------------------------
# Build matrix
# ------------------------------------------------------------

def build_matrix(data):
    sizes = sorted({k[0] for k in data})
    radii = sorted({k[1] for k in data})

    matrix = []

    for r in radii:
        row = []
        for s in sizes:
            row.append(data.get((s, r), np.nan))
        matrix.append(row)

    return np.array(matrix), sizes, radii


# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

def plot_heatmap(matrix, sizes, radii, title, cmap="viridis"):
    plt.figure()
    plt.imshow(matrix, aspect='auto', cmap=cmap)
    plt.colorbar()

    plt.xticks(range(len(sizes)), sizes)
    plt.yticks(range(len(radii)), radii)

    plt.xlabel("Grid size")
    plt.ylabel("Kernel radius")
    plt.title(title)

    plt.tight_layout()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    # ./bin/highmap_benchmarks --benchmark_format=json --benchmark_out=results.json

    benchmarks = load_results("build/results.json")
    grouped = group_benchmarks(benchmarks)

    for base, variants in grouped.items():
        cpu_data = variants["CPU"]
        gpu_data = variants["GPU"]

        if not cpu_data or not gpu_data:
            continue

        cpu_matrix, sizes, radii = build_matrix(cpu_data)
        gpu_matrix, _, _ = build_matrix(gpu_data)

        # --- Plot CPU & GPU
        if False:
            plot_heatmap(cpu_matrix, sizes, radii, f"{base} CPU")
            plot_heatmap(gpu_matrix, sizes, radii, f"{base} GPU")

        # --- Speedup CPU / GPU
        speedup = cpu_matrix / gpu_matrix

        plot_heatmap(speedup, sizes, radii,
                     f"{base} Speedup (CPU/GPU)",
                     cmap="jet")

        # print some summary stats
        print(f"\n=== {base} ===")
        print(f"Max speedup: {np.nanmax(speedup):.2f}")
        print(f"Min speedup: {np.nanmin(speedup):.2f}")

    plt.show()


if __name__ == "__main__":
    main()
