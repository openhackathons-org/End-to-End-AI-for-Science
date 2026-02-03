# ===============================================================
# VISUALIZING REPRESENTATION DRIFT IN DEEP NEURAL NETWORKS
# Comparing Standard Transolver vs GeoTransolver (GALE)
# ===============================================================
"""
LEARNING OBJECTIVE:
Understanding why GeoTransolver introduces "Geometry-Aware Latent Embeddings" (GALE).

The core problem: In deep networks, the original geometric information 
gradually "fades" as it passes through many layers. GeoTransolver solves
this by periodically re-injecting geometry context via cross-attention.

This notebook simulates that effect using the Ahmed Body geometry.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zarr
import os
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

# ===============================================================
# LOAD AHMED BODY GEOMETRY
# ===============================================================
def load_ahmed_body_geometry(zarr_path, downsample_factor=20):
    """
    Load Ahmed Body surface mesh from zarr dataset.
    
    The Ahmed Body is a standard automotive aerodynamics benchmark - 
    a simplified car shape used to study wake flows and drag.
    
    Args:
        zarr_path: Path to validation dataset
        downsample_factor: Reduce points for cleaner visualization
    
    Returns:
        torch.Tensor: (N, 3) geometry coordinates
    """
    try:
        sample_file = sorted(os.listdir(zarr_path))[0]
        full_path = os.path.join(zarr_path, sample_file)

        
        root = zarr.open_group(full_path, mode='r')
        geometry_np = root['surface_mesh_centers'][:]
        
        # Downsample for visualization clarity
        geometry_np = geometry_np[::downsample_factor]
        
        print(f"✅ Loaded Ahmed Body: {sample_file}")
        print(f"   Original points: {len(root['surface_mesh_centers'][:])}")
        print(f"   Visualizing: {len(geometry_np)} points (1/{downsample_factor} sampling)")
        
        return torch.from_numpy(geometry_np).float()
        
    except Exception as e:
        print(f"⚠️  Could not load Ahmed Body: {e}")
        print("   Falling back to synthetic car-like shape...")
        return create_synthetic_car_shape()

def create_synthetic_car_shape():
    """Create a simple car-like box shape as fallback."""
    # Body
    x = torch.linspace(-1, 1, 30)
    y = torch.linspace(-0.3, 0.3, 15)
    z = torch.linspace(0, 0.4, 10)
    
    points = []
    for xi in x:
        for yi in y:
            points.append([xi, yi, 0])      # Bottom
            points.append([xi, yi, 0.4])    # Top
        for zi in z:
            points.append([xi, -0.3, zi])   # Side
            points.append([xi, 0.3, zi])    # Side
    
    return torch.tensor(points).float()

# Load geometry
zarr_path = "/workspace/DLI/val"
geometry = load_ahmed_body_geometry(zarr_path, downsample_factor=20)

# ===============================================================
# SECTION 2: SIMULATE REPRESENTATION DRIFT
# ===============================================================
"""
🔬 THE PHYSICS OF DRIFT:

In each layer of a neural network, transformations introduce small perturbations.
Over many layers, these accumulate and the original geometric signal degrades.

Standard Transolver:
    h^(ℓ) = Transform(h^(ℓ-1)) + noise
    
GeoTransolver (GALE):
    h^(ℓ) = (1-α) · Transform(h^(ℓ-1)) + α · CrossAttn(h^(ℓ-1), Context)
    
The key insight: GALE re-injects geometry context C at EVERY layer via cross-attention,
preventing the "forgetting" problem.
"""

def simulate_drift(geometry, num_layers=50, noise_scale=0.08, alpha=0.35, seed=42):
    """
    Simulate representation drift through deep network layers.
    
    Args:
        geometry: (N, 3) original geometry tensor
        num_layers: Number of network layers to simulate
        noise_scale: Per-layer noise magnitude (simulates transformation drift)
        alpha: GALE injection strength (0=no injection, 1=full replacement)
        seed: Random seed for reproducibility
    
    Returns:
        dict: Contains histories and metrics for both methods
    """
    torch.manual_seed(seed)
    
    # Initialize states
    current_standard = geometry.clone()
    current_geo = geometry.clone()
    context = geometry.clone()  # Persistent geometry anchor
    
    # Storage for visualization
    history_standard = [geometry.numpy().copy()]
    history_geo = [geometry.numpy().copy()]
    
    # Quantitative metrics
    mse_standard = [0.0]  # MSE from original
    mse_geo = [0.0]
    max_deviation_standard = [0.0]
    max_deviation_geo = [0.0]
    
    for layer in range(num_layers):
        # Generate layer-specific noise (same for fair comparison)
        noise = torch.randn_like(geometry) * noise_scale
        
        # ─────────────────────────────────────────────────────────
        # METHOD A: STANDARD TRANSOLVER
        # Only processes previous layer output - no geometry recall
        # ─────────────────────────────────────────────────────────
        current_standard = current_standard + noise
        
        # ─────────────────────────────────────────────────────────
        # METHOD B: GEOTRANSOLVER (GALE)
        # Combines physics update with geometry context injection
        # Formula: h = (1-α)·physics_update + α·context
        # ─────────────────────────────────────────────────────────
        physics_update = current_geo + noise
        current_geo = (1 - alpha) * physics_update + alpha * context
        
        # Record history
        history_standard.append(current_standard.numpy().copy())
        history_geo.append(current_geo.numpy().copy())
        
        # Compute drift metrics
        mse_s = torch.mean((current_standard - geometry) ** 2).item()
        mse_g = torch.mean((current_geo - geometry) ** 2).item()
        max_s = torch.max(torch.abs(current_standard - geometry)).item()
        max_g = torch.max(torch.abs(current_geo - geometry)).item()
        
        mse_standard.append(mse_s)
        mse_geo.append(mse_g)
        max_deviation_standard.append(max_s)
        max_deviation_geo.append(max_g)
    
    return {
        'history_standard': history_standard,
        'history_geo': history_geo,
        'mse_standard': mse_standard,
        'mse_geo': mse_geo,
        'max_deviation_standard': max_deviation_standard,
        'max_deviation_geo': max_deviation_geo,
        'params': {'num_layers': num_layers, 'noise_scale': noise_scale, 'alpha': alpha}
    }

# Run simulation
print("\n🔄 Simulating representation drift...")
results = simulate_drift(geometry, num_layers=50, noise_scale=0.08, alpha=0.35)
print(f"   Parameters: {results['params']}")

# ===============================================================
# SECTION 3: VISUALIZATION - SIDE-BY-SIDE 3D COMPARISON
# ===============================================================

def plot_side_by_side_drift(results, geometry, layers_to_show=[0, 15, 30, 49]):
    """
    Create side-by-side comparison of drift evolution.
    Top row: Standard Transolver (drifts away)
    Bottom row: GeoTransolver (maintains structure)
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, len(layers_to_show) + 1, width_ratios=[1]*len(layers_to_show) + [0.3])
    
    # Color normalization based on original geometry
    z_min, z_max = geometry[:, 2].min(), geometry[:, 2].max()
    
    # Axis limits with padding
    x_lim = [geometry[:, 0].min() - 0.3, geometry[:, 0].max() + 0.3]
    y_lim = [geometry[:, 1].min() - 0.3, geometry[:, 1].max() + 0.3]
    z_lim = [geometry[:, 2].min() - 0.3, geometry[:, 2].max() + 0.3]
    
    for col, layer_idx in enumerate(layers_to_show):
        # ─── TOP ROW: STANDARD TRANSOLVER ───
        ax_std = fig.add_subplot(gs[0, col], projection='3d')
        data_std = results['history_standard'][layer_idx]
        
        scatter_std = ax_std.scatter(
            data_std[:, 0], data_std[:, 1], data_std[:, 2],
            c=data_std[:, 2], cmap='plasma', s=3, alpha=0.7,
            vmin=z_min, vmax=z_max
        )
        
        ax_std.set_xlim(x_lim)
        ax_std.set_ylim(y_lim)
        ax_std.set_zlim(z_lim)
        ax_std.view_init(elev=20, azim=-60)
        ax_std.set_title(f"Layer {layer_idx}", fontweight='bold', pad=5)
        ax_std.axis('off')
        
        if col == 0:
            ax_std.text2D(-0.15, 0.5, "Standard\nTransolver", 
                         transform=ax_std.transAxes, fontsize=12, 
                         fontweight='bold', color='#d62728',
                         va='center', ha='center', rotation=90)
        
        # ─── BOTTOM ROW: GEOTRANSOLVER ───
        ax_geo = fig.add_subplot(gs[1, col], projection='3d')
        data_geo = results['history_geo'][layer_idx]
        
        scatter_geo = ax_geo.scatter(
            data_geo[:, 0], data_geo[:, 1], data_geo[:, 2],
            c=data_geo[:, 2], cmap='viridis', s=3, alpha=0.7,
            vmin=z_min, vmax=z_max
        )
        
        ax_geo.set_xlim(x_lim)
        ax_geo.set_ylim(y_lim)
        ax_geo.set_zlim(z_lim)
        ax_geo.view_init(elev=20, azim=-60)
        ax_geo.axis('off')
        
        if col == 0:
            ax_geo.text2D(-0.15, 0.5, "GeoTransolver\n(GALE)", 
                         transform=ax_geo.transAxes, fontsize=12,
                         fontweight='bold', color='#2ca02c',
                         va='center', ha='center', rotation=90)
        
        # Add MSE annotation
        mse_std = results['mse_standard'][layer_idx]
        mse_geo = results['mse_geo'][layer_idx]
        ax_std.text2D(0.5, -0.05, f"MSE: {mse_std:.3f}", 
                     transform=ax_std.transAxes, fontsize=9, ha='center')
        ax_geo.text2D(0.5, -0.05, f"MSE: {mse_geo:.3f}", 
                     transform=ax_geo.transAxes, fontsize=9, ha='center')
    
    # Legend/explanation panel
    ax_legend = fig.add_subplot(gs[:, -1])
    ax_legend.axis('off')
    
    explanation = """
    📊 WHAT YOU'RE SEEING
    
    Top Row (Red label):
    Standard Transolver
    • No geometry re-injection
    • Information "drifts" away
    • Shape completely lost
    
    Bottom Row (Green label):
    GeoTransolver (GALE)
    • Cross-attention to geometry
    • Context re-injected each layer
    • Structure preserved!
    
    ─────────────────────
    
    🔬 KEY INSIGHT:
    
    GALE formula at layer ℓ:
    
    h = (1-α)·physics + α·context
    
    α = {:.2f} (injection strength)
    
    This "pulls" representations
    back toward original geometry.
    """.format(results['params']['alpha'])
    
    ax_legend.text(0.1, 0.95, explanation, transform=ax_legend.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.suptitle("Representation Drift: Why GeoTransolver Preserves Geometry",
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('drift_comparison_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===============================================================
# SECTION 4: QUANTITATIVE DRIFT ANALYSIS
# ===============================================================

def plot_drift_metrics(results):
    """
    Plot quantitative metrics showing drift over layers.
    This provides the mathematical evidence for what we see visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = np.arange(len(results['mse_standard']))
    
    # ─── LEFT: MSE FROM ORIGINAL ───
    ax1 = axes[0]
    ax1.plot(layers, results['mse_standard'], 'o-', color='#d62728', 
             label='Standard Transolver', linewidth=2, markersize=3, alpha=0.8)
    ax1.plot(layers, results['mse_geo'], 's-', color='#2ca02c',
             label='GeoTransolver (GALE)', linewidth=2, markersize=3, alpha=0.8)
    
    ax1.set_xlabel('Layer Number', fontsize=11)
    ax1.set_ylabel('Mean Squared Error from Original', fontsize=11)
    ax1.set_title('Drift Accumulation: MSE vs Layer Depth', fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(layers)-1])
    
    # Annotate final values
    final_std = results['mse_standard'][-1]
    final_geo = results['mse_geo'][-1]
    ax1.annotate(f'Final: {final_std:.2f}', xy=(len(layers)-1, final_std),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=9, color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.7))
    ax1.annotate(f'Final: {final_geo:.3f}', xy=(len(layers)-1, final_geo),
                xytext=(-50, -20), textcoords='offset points',
                fontsize=9, color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.7))
    
    # ─── RIGHT: DRIFT RATIO ───
    ax2 = axes[1]
    ratio = np.array(results['mse_standard']) / (np.array(results['mse_geo']) + 1e-8)
    ax2.fill_between(layers, 1, ratio, alpha=0.3, color='#2ca02c')
    ax2.plot(layers, ratio, '-', color='#2ca02c', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Layer Number', fontsize=11)
    ax2.set_ylabel('Drift Ratio (Standard / GALE)', fontsize=11)
    ax2.set_title('GALE Advantage: How Much Better?', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(layers)-1])
    
    # Annotate
    final_ratio = ratio[-1]
    ax2.text(0.95, 0.95, f'Final ratio: {final_ratio:.1f}×\nGALE is {final_ratio:.0f}× better',
            transform=ax2.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('drift_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📈 QUANTITATIVE SUMMARY")
    print("="*60)
    print(f"After {results['params']['num_layers']} layers:")
    print(f"  • Standard Transolver MSE: {final_std:.4f}")
    print(f"  • GeoTransolver MSE:       {final_geo:.4f}")
    print(f"  • Improvement Factor:      {final_ratio:.1f}×")
    print("="*60)
