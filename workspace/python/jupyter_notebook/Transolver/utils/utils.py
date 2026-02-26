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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr
import os

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


class StandardBlock3D(nn.Module):
    """
    Standard block that adds NOISE-LIKE perturbations.
    
    Key insight: We want the geometry to get NOISY, not just shifted.
    This is achieved by:
    1. MLP produces spatially-varying perturbations
    2. Additional random noise accumulates
    3. Result: Fuzzy/degraded geometry over layers
    """
    def __init__(self, hidden_dim=64, noise_scale=0.02):
        super().__init__()
        self.noise_scale = noise_scale
        
        # MLP produces structured perturbation
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Random initialization - different for each layer
                nn.init.normal_(m.weight, std=0.5)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.2, 0.2)
    
    def forward(self, x):
        # MLP perturbation (structured drift)
        mlp_perturbation = self.mlp(x) * 0.02
        
        # Random noise (this creates the "fuzzy" effect)
        random_noise = torch.randn_like(x) * self.noise_scale
        
        # Combined: structured + random perturbation
        return x + mlp_perturbation + random_noise


class GALEBlock3D(nn.Module):
    """
    GALE block: Same noise but with context anchoring that "denoises".
    
    Key insight: Context injection pulls noisy points back toward
    their original positions, effectively cleaning up the noise.
    """
    def __init__(self, hidden_dim=64, noise_scale=0.02, context_strength=0.6):
        super().__init__()
        self.noise_scale = noise_scale
        self.context_strength = context_strength
        
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.5)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.2, 0.2)
    
    def forward(self, x, context):
        # Same perturbation as Standard
        mlp_perturbation = self.mlp(x) * 0.02
        random_noise = torch.randn_like(x) * self.noise_scale
        physics_out = x + mlp_perturbation + random_noise
        
        # GALE: Pull back toward clean original context
        # This is like "denoising" - the clean context corrects the noise
        alpha = self.context_strength
        output = (1 - alpha) * physics_out + alpha * context
        
        return output, alpha


class StandardNetwork3D(nn.Module):
    """Standard network: Noise accumulates → geometry becomes fuzzy."""
    def __init__(self, num_layers=30, hidden_dim=64, noise_scale=0.02):
        super().__init__()
        self.layers = nn.ModuleList([
            StandardBlock3D(hidden_dim, noise_scale) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, return_intermediates=False):
        intermediates = [x.clone()]
        for layer in self.layers:
            x = layer(x)
            if return_intermediates:
                intermediates.append(x.clone())
        return (x, intermediates) if return_intermediates else x


class GALENetwork3D(nn.Module):
    """GALE network: Context injection prevents noise accumulation."""
    def __init__(self, num_layers=30, hidden_dim=64, noise_scale=0.02, context_strength=0.6):
        super().__init__()
        self.layers = nn.ModuleList([
            GALEBlock3D(hidden_dim, noise_scale, context_strength) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, return_intermediates=False):
        context = x.clone()  # Clean original geometry
        intermediates = [x.clone()]
        gates = []
        
        for layer in self.layers:
            x, alpha = layer(x, context)
            gates.append(alpha)
            if return_intermediates:
                intermediates.append(x.clone())
        
        return (x, intermediates, gates) if return_intermediates else x


def load_ahmed_body_geometry(zarr_path, downsample_factor=40):
    """Load Ahmed Body geometry."""
    try:
        sample_file = sorted(os.listdir(zarr_path))[0]
        root = zarr.open_group(os.path.join(zarr_path, sample_file), mode='r')
        geom = root['surface_mesh_centers'][::downsample_factor]
        print(f"✅ Loaded {len(geom)} points")
        return torch.from_numpy(geom).float()
    except Exception as e:
        print(f"⚠️ Using synthetic shape: {e}")
        pts = []
        for x in np.linspace(-1, 1, 30):
            for y in np.linspace(-0.3, 0.3, 15):
                pts.extend([[x, y, 0], [x, y, 0.35]])
        return torch.tensor(pts).float()


def run_mlp_drift_comparison(geometry, num_layers=30, hidden_dim=64, 
                              noise_scale=0.03, context_strength=0.6):
    """
    Run comparison showing NOISY degradation.
    
    Args:
        noise_scale: Per-layer noise magnitude (0.02-0.05)
                    Higher = more visible noise accumulation
        context_strength: How strongly GALE denoises (0.5-0.8)
    """
    print(f"\n{'='*65}")
    print("🧠 NOISY DRIFT COMPARISON (EDUCATIONAL VERSION)")
    print(f"{'='*65}")
    print(f"   Geometry points:    {len(geometry)}")
    print(f"   Number of layers:   {num_layers}")
    print(f"   Noise scale:        {noise_scale}")
    print(f"   Context strength:   {context_strength}")
    print(f"{'='*65}")
    
    # Set eval mode but keep noise active
    standard = StandardNetwork3D(num_layers, hidden_dim, noise_scale)
    gale = GALENetwork3D(num_layers, hidden_dim, noise_scale, context_strength)
    
    # Note: We use train mode to keep the random noise active
    standard.train()
    gale.train()
    
    with torch.no_grad():
        _, std_hist = standard(geometry.clone(), True)
        _, gale_hist, gates = gale(geometry.clone(), True)
    
    # Compute MSE from original
    mse_std = [torch.mean((h - geometry)**2).item() for h in std_hist]
    mse_gale = [torch.mean((h - geometry)**2).item() for h in gale_hist]
    
    # Compute "noise level" (standard deviation of displacement)
    noise_std = [torch.std(h - geometry).item() for h in std_hist]
    noise_gale = [torch.std(h - geometry).item() for h in gale_hist]
    
    improvement = mse_std[-1] / (mse_gale[-1] + 1e-8)
    
    print(f"\n📊 Results Summary:")
    print(f"   Layer 0 MSE:        {mse_std[0]:.6f}")
    print(f"   Final MSE (Std):    {mse_std[-1]:.4f}")
    print(f"   Final MSE (GALE):   {mse_gale[-1]:.4f}")
    print(f"   Improvement:        {improvement:.1f}×")
    print(f"\n📐 Noise Level (Std Dev of displacement):")
    print(f"   Final noise (Std):  {noise_std[-1]:.4f}")
    print(f"   Final noise (GALE): {noise_gale[-1]:.4f}")
    print(f"{'='*65}\n")
    
    return {
        'std_hist': [h.numpy() for h in std_hist],
        'gale_hist': [h.numpy() for h in gale_hist],
        'mse_std': mse_std,
        'mse_gale': mse_gale,
        'noise_std': noise_std,
        'noise_gale': noise_gale,
        'gates': gates,
        'params': {
            'num_layers': num_layers,
            'noise_scale': noise_scale,
            'context_strength': context_strength
        }
    }


def plot_mlp_drift_comparison(results, geometry, layers_to_show=None):
    """Visualize noisy drift comparison."""
    num_layers = results['params']['num_layers']
    
    if layers_to_show is None:
        layers_to_show = [0, num_layers//3, 2*num_layers//3, num_layers]
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, len(layers_to_show), height_ratios=[1.3, 1.3, 0.7],
                  hspace=0.12, wspace=0.08)
    
    geom_np = geometry.numpy()
    z_min, z_max = geom_np[:, 2].min(), geom_np[:, 2].max()
    
    # Fixed axis limits based on original geometry + some padding for noise
    padding = 0.4
    x_lim = [geom_np[:, 0].min() - padding, geom_np[:, 0].max() + padding]
    y_lim = [geom_np[:, 1].min() - padding, geom_np[:, 1].max() + padding]
    z_lim = [geom_np[:, 2].min() - padding, geom_np[:, 2].max() + padding]
    
    for col, layer_idx in enumerate(layers_to_show):
        # Standard row - should look increasingly noisy
        ax = fig.add_subplot(gs[0, col], projection='3d')
        d = results['std_hist'][layer_idx]
        ax.scatter(d[:,0], d[:,1], d[:,2], c=geom_np[:,2], cmap='plasma',
                  s=10, alpha=0.8, vmin=z_min, vmax=z_max)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
        ax.view_init(25, -55); ax.axis('off')
        ax.set_title(f"Layer {layer_idx}", fontsize=14, fontweight='bold')
        if col == 0:
            ax.text2D(-0.08, 0.5, "Standard\n(No GALE)", transform=ax.transAxes,
                     fontsize=13, fontweight='bold', color='#d62728',
                     rotation=90, va='center')
        
        # Show both MSE and noise level
        mse = results['mse_std'][layer_idx]
        noise = results['noise_std'][layer_idx]
        ax.text2D(0.5, -0.02, f"MSE: {mse:.4f}\nNoise: {noise:.4f}",
                 transform=ax.transAxes, fontsize=10, ha='center', color='#d62728',
                 fontweight='bold')
        
        # GALE row - should stay clean
        ax = fig.add_subplot(gs[1, col], projection='3d')
        d = results['gale_hist'][layer_idx]
        ax.scatter(d[:,0], d[:,1], d[:,2], c=geom_np[:,2], cmap='viridis',
                  s=10, alpha=0.8, vmin=z_min, vmax=z_max)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
        ax.view_init(25, -55); ax.axis('off')
        if col == 0:
            ax.text2D(-0.08, 0.5, "GALE\n(Context)", transform=ax.transAxes,
                     fontsize=13, fontweight='bold', color='#2ca02c',
                     rotation=90, va='center')
        
        mse = results['mse_gale'][layer_idx]
        noise = results['noise_gale'][layer_idx]
        ax.text2D(0.5, -0.02, f"MSE: {mse:.4f}\nNoise: {noise:.4f}",
                 transform=ax.transAxes, fontsize=10, ha='center', color='#2ca02c',
                 fontweight='bold')
    
    # Metrics plot
    ax = fig.add_subplot(gs[2, :])
    layers = range(len(results['mse_std']))
    
    # Plot MSE
    ax.plot(layers, results['mse_std'], 'o-', color='#d62728', lw=2.5, ms=5,
           label='Standard MSE')
    ax.plot(layers, results['mse_gale'], 's-', color='#2ca02c', lw=2.5, ms=5,
           label='GALE MSE')
    ax.fill_between(layers, results['mse_gale'], results['mse_std'],
                   alpha=0.2, color='red')
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE from Original', fontsize=12, fontweight='bold')
    ax.set_title('Noise Accumulation: Standard vs GALE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ratio = results['mse_std'][-1] / (results['mse_gale'][-1] + 1e-8)
    ax.text(0.98, 0.95, 
           f'GALE is {ratio:.1f}× better\n(Less noisy at final layer)', 
           transform=ax.transAxes,
           fontsize=12, ha='right', va='top', fontweight='bold',
           bbox=dict(facecolor='#e8f5e9', alpha=0.95, edgecolor='#2ca02c'))
    
    plt.suptitle('Representation Drift as Noise Accumulation: Standard vs GALE',
                fontsize=18, fontweight='bold')
    plt.subplots_adjust(top=0.93, hspace=0.18, wspace=0.12)
#    plt.savefig('noisy_drift_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
