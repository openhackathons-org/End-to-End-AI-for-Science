import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openmc


class Pincell:

    def __init__(
        self,
        instance_id,
        enrichment,
        pitch,
        fuel_radius,
        gap_thickness,
        clad_thickness,
        boundary_type="reflective",
        output_dir="data/input",
        save_geometry_figure=False,
    ):

        self.instance_id = instance_id

        self.enrichment = enrichment
        self.pitch = pitch
        self.fuel_radius = fuel_radius
        self.gap_thickness = gap_thickness
        self.clad_thickness = clad_thickness
        self.boundary_type = boundary_type

        self.gap_radius = self.fuel_radius + self.gap_thickness
        self.clad_radius = self.gap_radius + self.clad_thickness

        self.fuel = self._create_fuel()
        self.cladding = self._create_cladding()
        self.moderator = self._create_moderator()
        self.materials = openmc.Materials([self.fuel, self.cladding, self.moderator])

        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "npy"), exist_ok=True)

        self.geometry = self._create_geometry()
        if save_geometry_figure:
            self.save_geometry_figure()

        self.save_masks()

    def _create_fuel(self):
        """
        Uranium dioxide UO2
        """
        fuel = openmc.Material(name="uo2")
        fuel.add_element("U", 1.0, enrichment=self.enrichment, percent_type="ao")
        fuel.add_nuclide("O16", 2.0, percent_type="ao")
        fuel.set_density("g/cm3", 10.0)
        return fuel

    def _create_cladding(self):
        """
        Zirconium cladding
        """
        cladding = openmc.Material(name="zirconium")
        cladding.add_element("Zr", 1.0)
        cladding.set_density("g/cm3", 6.6)
        return cladding

    def _create_moderator(self):
        """
        Light-water coolant/moderator
        """
        moderator = openmc.Material(name="water")
        moderator.add_nuclide("H1", 2.0)
        moderator.add_nuclide("O16", 1.0)
        moderator.set_density("g/cm3", 1.0)
        moderator.add_s_alpha_beta("c_H_in_H2O")
        return moderator

    def _create_geometry(self):
        fuel_or = openmc.ZCylinder(r=self.fuel_radius)
        clad_ir = openmc.ZCylinder(r=self.gap_radius)
        clad_or = openmc.ZCylinder(r=self.clad_radius)

        fuel_cell = openmc.Cell(fill=self.fuel, region=-fuel_or)
        gap_cell = openmc.Cell(fill=None, region=+fuel_or & -clad_ir)
        clad_cell = openmc.Cell(fill=self.cladding, region=+clad_ir & -clad_or)

        box = openmc.model.RectangularPrism(
            width=self.pitch, height=self.pitch, boundary_type=self.boundary_type
        )
        moderator_cell = openmc.Cell(fill=self.moderator, region=-box & +clad_or)

        root_universe = openmc.Universe(
            cells=[fuel_cell, gap_cell, clad_cell, moderator_cell]
        )
        return openmc.Geometry(root_universe)

    def save_geometry_figure(self, basis="xy", pixels=(1000, 1000)):
        """
        Saves a 2D plot of the pin-cell geometry. Auto-generates
        filename based on object count.
        """
        fig = self.geometry.plot(
            width=(self.pitch, self.pitch),
            basis=basis,
            pixels=pixels,
            origin=(0.0, 0.0, 0.0),
        )

        plt.title(
            f"Pincell #{self.instance_id} Geometry ({basis.upper()} view)",
            fontsize=14,
            fontweight="bold",
        )
        plt.axis("equal")
        plt.tight_layout()

        filename = f"png/pincell_{self.instance_id}_{basis}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def save_material_map(self, filename="material-map.h5", nx=1000, ny=1000, nz=10):
        """
        Generate and save a voxelised material map using vectorised geometry.
        Material IDs: fuel.id for fuel, cladding.id for cladding,
        moderator.id for moderator, 0 for gap/void.
        """
        masks = self.build_x(nx, ny, nz)
        material_map = np.zeros((nx, ny, nz), dtype=int)
        material_map[masks[0].astype(bool)] = self.fuel.id
        material_map[masks[1].astype(bool)] = self.cladding.id
        material_map[masks[2].astype(bool)] = self.moderator.id

        with h5py.File(filename, "w") as f:
            f.create_dataset("materials", data=material_map)

    def build_x(self, nx=100, ny=100, nz=10):
        """
        Construct stacked material masks (fuel, cladding, moderator)
        using vectorised analytic geometry instead of point-by-point queries.

        Returns:
            np.ndarray of shape (3, nx, ny, nz) with dtype float32
        """
        xmax = self.pitch / 2.0
        ymax = self.pitch / 2.0
        zmax = self.clad_radius

        x = np.linspace(-xmax, xmax, nx)
        y = np.linspace(-ymax, ymax, ny)

        r_sq = x[:, None] ** 2 + y[None, :] ** 2

        fuel_2d = r_sq < self.fuel_radius**2
        clad_2d = (r_sq >= self.gap_radius**2) & (r_sq < self.clad_radius**2)
        mod_2d = r_sq >= self.clad_radius**2

        fuel = np.broadcast_to(fuel_2d[:, :, None], (nx, ny, nz))
        clad = np.broadcast_to(clad_2d[:, :, None], (nx, ny, nz))
        mod = np.broadcast_to(mod_2d[:, :, None], (nx, ny, nz))

        return np.stack([fuel, clad, mod], axis=0).astype(np.float32)

    def save_masks(self, nx=1000, ny=1000, nz=1):
        """
        Save input features (masks) to a .npy file
        """
        x = self.build_x(nx, ny, nz)
        filename = f"npy/pincell_{self.instance_id}_masks.npy"
        np.save(os.path.join(self.output_dir, filename), x)

    def plot_material_masks(self, nx=1000, ny=1000, nz=1):
        """
        Generates and plots material masks
        """
        x = self.build_x(nx, ny, nz)

        z_idx = nz // 2

        f_slice = x[0, :, :, z_idx].T
        c_slice = x[1, :, :, z_idx].T
        m_slice = x[2, :, :, z_idx].T

        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        extent = [-self.pitch / 2, self.pitch / 2, -self.pitch / 2, self.pitch / 2]

        kwargs = {
            "origin": "lower",
            "extent": extent,
            "cmap": "gray_r",
            "vmin": 0,
            "vmax": 1,
        }

        axes[0].imshow(f_slice, **kwargs)
        axes[0].set_title("Fuel Mask (Channel 0)")

        axes[1].imshow(c_slice, **kwargs)
        axes[1].set_title("Cladding Mask (Channel 1)")

        axes[2].imshow(m_slice, **kwargs)
        axes[2].set_title("Moderator Mask (Channel 2)")

        composite = np.zeros((ny, nx, 3))
        composite[..., 0] = f_slice
        composite[..., 1] = c_slice
        composite[..., 2] = m_slice

        axes[3].imshow(composite, origin="lower", extent=extent)
        axes[3].set_title("Composite Tensor (RGB)")

        for ax in axes:
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("y [cm]")

        plt.tight_layout()
        plt.show()
