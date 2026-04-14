import matplotlib.pyplot as plt
import openmc


def plot_flux_xy(statepoint_path, tally_name="physics", z_slice=0):
    """
    Plot the neutron flux in the XY plane at a given z index (top-down view).
    """
    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)

        mesh = tally.filters[0].mesh
        flux = tally.mean.ravel()
        mesh_shape = mesh.dimension
        flux_3d = flux.reshape(mesh_shape)

        x_min, y_min, _ = mesh.lower_left
        x_max, y_max, _ = mesh.upper_right

    plt.imshow(
        flux_3d[:, :, z_slice],
        origin="lower",
        cmap="inferno",
        extent=[x_min, x_max, y_min, y_max],
    )
    plt.colorbar(label="Neutron Flux")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title(f"Neutron Flux (z={z_slice})")
    plt.axis("equal")
    plt.show()


def plot_flux_xz(statepoint_path, tally_name="physics", y_index=0):
    """
    Plot the neutron flux in the XZ plane at a given y index (side profile).
    """
    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)

        mesh = tally.filters[0].mesh
        flux = tally.mean.ravel()
        mesh_shape = mesh.dimension
        flux_3d = flux.reshape(mesh_shape)

        x_min, _, z_min = mesh.lower_left
        x_max, _, z_max = mesh.upper_right

    plt.imshow(
        flux_3d[:, y_index, :].T,
        origin="lower",
        cmap="inferno",
        extent=[x_min, x_max, z_min, z_max],
    )
    plt.colorbar(label="Neutron Flux")
    plt.xlabel("x [cm]")
    plt.ylabel("z [cm]")
    plt.title(f"Neutron Flux (y index={y_index})")
    plt.axis("equal")
    plt.show()
