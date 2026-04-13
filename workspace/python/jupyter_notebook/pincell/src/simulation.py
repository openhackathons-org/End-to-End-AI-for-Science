import os

import numpy as np
import openmc

from src.utils import compute_homogenized_xs


class Simulation:

    def __init__(
        self,
        batches=100,
        inactive=10,
        particles=10000,
        mesh_dim=(20, 20, 2),
        mesh_z=(-10.0, 10.0),
    ):
        self.batches = batches
        self.inactive = inactive
        self.particles = particles
        self.mesh_dim = mesh_dim
        self.mesh_z = mesh_z

    def _create_settings(self):
        """
        Return OpenMC settings for k-eigenvalue simulation
        """
        settings = openmc.Settings()
        settings.run_mode = "eigenvalue"
        settings.batches = self.batches
        settings.inactive = self.inactive
        settings.particles = self.particles
        return settings

    def _create_tallies(self, pitch):
        """
        Return tallies object with a 3D mesh flux tally
        """
        tallies = openmc.Tallies()

        mesh = openmc.RegularMesh(mesh_id=1)
        mesh.dimension = list(self.mesh_dim)
        mesh.lower_left = [-pitch / 2.0, -pitch / 2.0, self.mesh_z[0]]
        mesh.upper_right = [pitch / 2.0, pitch / 2.0, self.mesh_z[1]]
        mesh_filter = openmc.MeshFilter(mesh)

        physics_tally = openmc.Tally(name="physics")
        physics_tally.filters = [mesh_filter]
        physics_tally.scores = ["flux", "absorption"]
        tallies.append(physics_tally)
        return tallies

    def _extract_results(self, statepoint_path, output_dir, tally_name="physics"):
        """
        Extract k-eff, homogenised cross-section, and save the
        per-cell sigma_a map from the statepoint.
        """
        with openmc.StatePoint(statepoint_path) as sp:
            k_eff = sp.keff.nominal_value
            tally = sp.get_tally(name=tally_name)
            mesh_shape = tally.filters[0].mesh.dimension

            flux = tally.get_values(scores=["flux"]).reshape(mesh_shape)
            absorption = tally.get_values(scores=["absorption"]).reshape(mesh_shape)

        safe_flux = np.where(flux > 0, flux, 1.0)
        sigma_a_map = np.where(flux > 0, absorption / safe_flux, 0.0)
        np.save(os.path.join(output_dir, "sigma_a_map.npy"), sigma_a_map)

        xs = compute_homogenized_xs(sigma_a_map, flux)

        return k_eff, xs

    def run(self, pincell):
        """
        Run OpenMC simulation for a given Pincell object.

        Args:
            pincell: Pincell geometry/materials

        Returns:
            dict: Input parameters and k-effective result.
        """
        output_dir = f"data/output/pincell_{pincell.instance_id}"
        os.makedirs(output_dir, exist_ok=True)

        settings = self._create_settings()
        tallies = self._create_tallies(pincell.pitch)

        model = openmc.Model(
            geometry=pincell.geometry,
            materials=pincell.materials,
            settings=settings,
            tallies=tallies,
        )

        statepoint_file = os.path.join(output_dir, f"statepoint.{settings.batches}.h5")
        model.export_to_xml(directory=output_dir)
        openmc.run(cwd=output_dir, openmc_exec="openmc")

        k_eff, xs = self._extract_results(statepoint_file, output_dir)

        return {
            "pincell_id": pincell.instance_id,
            "enrichment_wpct": pincell.enrichment,
            "pitch_cm": pincell.pitch,
            "fuel_radius_cm": pincell.fuel_radius,
            "gap_thickness_cm": pincell.gap_thickness,
            "clad_thickness_cm": pincell.clad_thickness,
            "k_effective": k_eff,
            "xs": xs,
            "statepoint_path": statepoint_file,
        }
