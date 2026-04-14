import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import openmc


def lhs(n_samples, bounds, seed=None, centered=False):
    """
    Latin hypercube sampling in continuous bounds.
    bounds: list of (low, high) per dimension.
    """
    rng = np.random.default_rng(seed)
    n = int(n_samples)
    d = len(bounds)

    base = (np.arange(n) + (0.5 if centered else rng.random(n))) / n
    X = np.empty((n, d), dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        u = base.copy()
        rng.shuffle(u)
        X[:, j] = lo + u * (hi - lo)
    return X


def generate_pincells(n_samples, bounds, pitch_max, seed=None, centered=False):
    """
    Generate pincell parameters ensuring the pin fits
    within the pitch bounds (pitch >= 2 x outer radius)

    bounds: bounds for (enrichment, fuel_r, gap, clad)
    pitch_max: upper bound for pitch
    """
    X = lhs(n_samples, bounds, seed=seed, centered=centered)
    enrich, fuel_r, gap, clad = X.T
    outer_r = fuel_r + gap + clad

    rng = np.random.default_rng(seed)
    pitch = np.array([rng.uniform(2 * orad, pitch_max) for orad in outer_r])

    return enrich, pitch, fuel_r, gap, clad


def plot_input(mask):
    """
    Visualise input
    """
    mask_np = mask.numpy()

    plt.figure(figsize=(8, 6))

    cmap = mcolors.ListedColormap(["mediumseagreen", "black", "skyblue"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    img = plt.imshow(mask_np, cmap=cmap, norm=norm, origin="lower")

    cbar = plt.colorbar(img, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Fuel", "Cladding", "Moderator"])

    plt.title(f"Resolution: {mask_np.shape}")
    plt.xlabel("x")
    plt.ylabel("y")


def plot_output(flux):
    """
    Visualise the neutron flux field.
    """

    flux_np = flux.numpy()

    plt.figure(figsize=(8, 6))

    plt.imshow(flux_np, origin="lower", cmap="inferno")
    plt.colorbar(label="Neutron Flux")
    plt.title(f"Resolution: {flux_np.shape}")
    plt.xlabel("x [index]")
    plt.ylabel("y [index]")
    plt.axis("equal")


def plot_comparison(
    true_flux, pred_flux, dark=False, transparent=False, save_path=None
):
    """
    Plots true and predicted flux side-by-side with a shared color scale.
    """
    t_data = true_flux.numpy() if hasattr(true_flux, "numpy") else true_flux
    p_data = pred_flux.numpy() if hasattr(pred_flux, "numpy") else pred_flux

    vmin = min(t_data.min(), p_data.min())
    vmax = max(t_data.max(), p_data.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dark:
        fig.patch.set_facecolor("black")
        for ax in axes:
            ax.set_facecolor("black")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.tick_params(colors="white")
    if transparent:
        fig.patch.set_alpha(0.0)

    for ax, data, title in zip(axes, [t_data, p_data], ["Observed", "Predicted"]):
        im = ax.imshow(data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set(title=title)
        ax.axis("equal")
        ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=axes,
        label=r"Neutron Flux $n \cdot cm^{-2} \cdot s^{-1}$",
        fraction=0.05,
        pad=0.04,
    )

    if dark:
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")
        cbar.outline.set_edgecolor("white")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", transparent=transparent)
    else:
        plt.show()


def compute_homogenized_xs(sigma_a, flux):
    """
    Compute the homogenized macroscopic absorption cross-section.

    Sigma_homog = sum(sigma_a * flux) / sum(flux)

    Args:
        sigma_a: Per-cell macroscopic absorption cross-section (numpy or torch)
        flux: Neutron flux field (observed or predicted)
    """
    weighted = (sigma_a * flux).sum()
    total_flux = flux.sum()
    result = weighted / total_flux
    return result.item() if hasattr(result, "item") else float(result)


def compute_homogenized_xs_from_statepoint(statepoint_path, tally_name="physics"):
    """
    Compute the homogenized macroscopic absorption cross-section
    directly from an OpenMC statepoint file.
    """
    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)
        flux = tally.get_values(scores=["flux"]).ravel()
        absorption = tally.get_values(scores=["absorption"]).ravel()

    safe_flux = np.where(flux > 0, flux, 1.0)
    sigma_a = np.where(flux > 0, absorption / safe_flux, 0.0)
    return compute_homogenized_xs(sigma_a, flux)
