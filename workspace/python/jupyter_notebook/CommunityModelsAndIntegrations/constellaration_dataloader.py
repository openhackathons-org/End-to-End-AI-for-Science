import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from constellaration.geometry.surface_rz_fourier import (
    SurfaceRZFourier,
    evaluate_points_rz,
)


CHALLENGE_KEYS = {
    "challenge_1": ["metrics.max_elongation"],
    "challenge_2": [
        "metrics.minimum_normalized_magnetic_gradient_scale_length",
        "metrics.qi",
    ],
    "challenge_3": [
        "metrics.vacuum_well",
        "metrics.flux_compression_in_regions_of_bad_curvature",
        "metrics.qi",
    ],
}

INPUT_KEYS = [
    "boundary.json",
    "boundary.r_cos",
    "boundary.z_sin",
    "boundary.n_field_periods",
    "boundary.is_stellarator_symmetric",
]


class BasicConstellarationDataset(Dataset):
    def __init__(self, examples, input_keys, output_keys, device=None):
        self.examples = examples
        input_keys = [k for k in input_keys if k != "boundary.json"]
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        x_parts = [
            torch.tensor(ex[k], dtype=torch.float32).flatten()
            for k in self.input_keys
            if k in ex
        ]
        y_parts = [
            torch.tensor(ex[k], dtype=torch.float32).flatten()
            for k in self.output_keys
            if k in ex
        ]

        x = torch.cat(x_parts)
        y = torch.cat(y_parts)

        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


class FullConstellarationDataset(Dataset):
    def __init__(
        self, examples, input_keys, output_keys, device=None, grid_size=(64, 64, 64)
    ):
        self.examples = examples
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.device = device
        self.grid_size = grid_size

        # Setup coordinate grids
        Nt, Np, Nrho = grid_size
        theta = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, Np, endpoint=False)
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")  # (Nt, Np)
        self.theta_phi = np.stack((Theta, Phi), axis=-1)  # shape (Nt, Np, 2)

        # Radial coordinate for 3D volume
        rho = np.linspace(0, 1, Nrho)
        self.Rho = rho[None, None, :]  # shape (1, 1, Nrho)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Extract boundary JSON and parse geometry
        boundary_json = ex["boundary.json"]
        boundary = SurfaceRZFourier.model_validate_json(boundary_json)

        # Evaluate surface points in R-Z
        rz = evaluate_points_rz(boundary, self.theta_phi)  # shape (Nt, Np, 2)
        R, Z = rz[..., 0], rz[..., 1]

        # Build 3D volume using radial parameter rho ∈ [0, 1]
        Nt, Np, Nrho = self.grid_size
        Phi = self.theta_phi[..., 1]  # Extract phi from theta_phi

        # Broadcast cylindrical coordinates to 3D
        R_3d = R[:, :, None] * self.Rho  # shape (Nt, Np, Nrho)
        Z_3d = Z[:, :, None] * self.Rho  # shape (Nt, Np, Nrho)
        Phi_3d = Phi[:, :, None]  # shape (Nt, Np, 1), auto-broadcasted

        # Convert cylindrical to Cartesian
        X = R_3d * np.cos(Phi_3d)  # shape (Nt, Np, Nrho)
        Y = R_3d * np.sin(Phi_3d)  # shape (Nt, Np, Nrho)

        # Stack into volume tensor: [Nt, Np, Nrho, 3]
        vol = np.stack([X, Y, Z_3d], axis=3)
        vol = torch.from_numpy(vol).float()

        # Reshape for TFNO: [C, H, W, D] -> [C, H, W, D]
        # TFNO expects: [batch, channels, x, y, z]
        x = vol.permute(3, 0, 1, 2)  # [channels, Nt, Np, Nrho]

        # Extract output targets
        y_parts = [
            torch.tensor(ex[k], dtype=torch.float32).flatten()
            for k in self.output_keys
            if k in ex
        ]
        y = torch.cat(y_parts)

        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)

        return {"input_fields": x, "output_fields": y}


class ConstellarationDataLoader:
    def __init__(
        self,
        challenge: str = "challenge_1",
        val_size: float = 0.1,
        test_size: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
        dataset_type: str = "basic",
        grid_size: tuple = (64, 64, 64),
    ):
        assert challenge in CHALLENGE_KEYS, f"Unknown challenge: {challenge}"
        self.challenge = challenge
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.grid_size = grid_size

        self.input_keys = INPUT_KEYS
        self.output_keys = CHALLENGE_KEYS[challenge]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare(self):
        # Load full dataset
        ds = load_dataset("proxima-fusion/constellaration", split="train", num_proc=4)
        # ds = ds[:1000]

        # Select only boundary.* and metrics.* columns
        ds = ds.select_columns(
            [
                c
                for c in ds.column_names
                if c.startswith("boundary.") or c.startswith("metrics.")
            ]
        )
        # Convert to Python format for indexing
        ds.set_format("python")

        # Filter out samples with missing outputs OR inputs
        filtered_ds = ds.filter(
            lambda x: all(
                k in x and x[k] is not None for k in self.output_keys + self.input_keys
            )
        )

        # Split indices
        # indices = list(range(len(filtered_ds)))
        indices = list(range(1000))
        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.seed
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.seed,
        )
        # Index data
        train_data = [filtered_ds[i] for i in train_idx]
        val_data = [filtered_ds[i] for i in val_idx]
        test_data = [filtered_ds[i] for i in test_idx]

        # Create datasets
        if self.dataset_type == "basic":
            self.train_dataset = BasicConstellarationDataset(
                train_data, self.input_keys, self.output_keys, device=self.device
            )
            self.val_dataset = BasicConstellarationDataset(
                val_data, self.input_keys, self.output_keys, device=self.device
            )
            self.test_dataset = BasicConstellarationDataset(
                test_data, self.input_keys, self.output_keys, device=self.device
            )
        elif self.dataset_type == "full":
            self.train_dataset = FullConstellarationDataset(
                train_data,
                self.input_keys,
                self.output_keys,
                device=self.device,
                grid_size=self.grid_size,
            )
            self.val_dataset = FullConstellarationDataset(
                val_data,
                self.input_keys,
                self.output_keys,
                device=self.device,
                grid_size=self.grid_size,
            )
            self.test_dataset = FullConstellarationDataset(
                test_data,
                self.input_keys,
                self.output_keys,
                device=self.device,
                grid_size=self.grid_size,
            )

    def train_dataloader(self):
        return self.get_dataloaders()["train"]

    def val_dataloader(self):
        return self.get_dataloaders()["val"]

    def test_dataloader(self):
        return self.get_dataloaders()["test"]

    def get_dataloaders(self):
        return {
            "train": DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            ),
            "val": DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            ),
            "test": DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            ),
        }
