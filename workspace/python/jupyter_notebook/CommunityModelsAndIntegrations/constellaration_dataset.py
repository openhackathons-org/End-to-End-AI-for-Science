import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from constellaration.geometry.surface_rz_fourier import (
    SurfaceRZFourier,
    evaluate_points_rz,
)
from datasets import load_dataset


class ConstellarationDataset(Dataset):
    """Dataset for constellation plasma boundary data."""

    def __init__(self, data, grid_shape=(64, 64, 64), num_channels=7):
        """
        Initialize the dataset.

        Args:
            data: List of dataset items with boundary and metrics fields
            grid_shape: Spatial grid resolution (H, W, D)
            num_channels: Number of channels in output tensor
        """
        self.data = data
        self.grid_shape = grid_shape
        self.num_channels = num_channels

        # Pre-compute theta, phi grids
        Nt, Np, Nrho = grid_shape
        theta = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, Np, endpoint=False)
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
        self.theta_phi = np.stack((Theta, Phi), axis=-1)  # shape (Nt, Np, 2)

        # Pre-compute rho for 3D volume
        rho = np.linspace(0, 1, Nrho)
        self.Rho = rho[None, None, :]  # shape (1, 1, Nrho)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Generate a sample from boundary geometry."""
        item = self.data[idx]

        # Extract boundary JSON for geometry processing
        boundary_json = item["boundary.json"]

        # Parse boundary geometry
        boundary = SurfaceRZFourier.model_validate_json(boundary_json)

        # Evaluate surface points in R-Z
        rz = evaluate_points_rz(boundary, self.theta_phi)  # shape (Nt, Np, 2)
        R, Z = rz[..., 0], rz[..., 1]

        # Build 3D volume using radial parameter rho ∈ [0, 1]
        Nt, Np, Nrho = self.grid_shape
        Phi = self.theta_phi[..., 1]  # Extract phi from theta_phi

        # Broadcast cylindrical coordinates to 3D
        R_3d = R[:, :, None] * self.Rho  # shape (Nt, Np, Nrho)
        Z_3d = Z[:, :, None] * self.Rho  # shape (Nt, Np, Nrho)
        Phi_3d = Phi[:, :, None]  # shape (Nt, Np, 1), auto-broadcasted

        # Convert cylindrical to Cartesian
        X = R_3d * np.cos(Phi_3d)  # shape (Nt, Np, Nrho)
        Y = R_3d * np.sin(Phi_3d)  # shape (Nt, Np, Nrho)

        # Stack into volume tensor
        vol = np.stack([X, Y, Z_3d], axis=3)  # shape (Nt, Np, Nrho, 3)
        vol = torch.from_numpy(vol).float()  # shape (Nt, Np, Nrho, 3)

        # Pad to desired number of channels
        if self.num_channels > 3:
            # Repeat volume and add zeros for remaining channels
            vol_padded = torch.cat(
                [
                    vol,  # 3 channels
                    vol,  # 3 more channels (repeated)
                    torch.zeros(
                        Nt, Np, Nrho, self.num_channels - 6
                    ),  # remaining channels
                ],
                dim=-1,
            )
        else:
            vol_padded = vol

        # Add time dimension for TFNO compatibility: [B, T, H, W, D, C]
        # For now, we'll use a single time step
        vol_padded = vol_padded.unsqueeze(0)  # shape [1, H, W, D, C]

        # Create output dictionary with all fields
        output = {
            "input_fields": vol_padded,  # shape [1, H, W, D, C]
        }

        # Add all boundary fields (excluding the JSON)
        for key, value in item.items():
            if key.startswith("boundary.") and key != "boundary.json":
                output[key] = torch.tensor(value, dtype=torch.float32)

        # Add all metrics fields (excluding the JSON and id)
        for key, value in item.items():
            if key.startswith("metrics.") and key not in ["metrics.json", "metrics.id"]:
                output[key] = torch.tensor(value, dtype=torch.float32)

        return output


class ConstellarationDataModule:
    """Plain PyTorch DataModule for constellation dataset."""

    def __init__(
        self,
        dataset_name="proxima-fusion/constellaration",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        batch_size=4,
        num_workers=4,
        grid_shape=(64, 64, 64),
        num_channels=7,
        **kwargs,
    ):
        """
        Initialize the DataModule.

        Args:
            dataset_name: HuggingFace dataset name
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            grid_shape: Spatial grid resolution
            num_channels: Number of channels in output tensor
        """
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grid_shape = grid_shape
        self.num_channels = num_channels

        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Train, val, and test splits must sum to 1.0, got {total_split}"
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader_instance = None
        self.val_dataloader_instance = None
        self.test_dataloader_instance = None

    def setup(self):
        """Setup datasets and dataloaders for training, validation, and testing."""
        # Load the full dataset with all boundary and metrics columns
        ds = load_dataset(
            self.dataset_name,
            split="train",
            num_proc=4,
        )

        # Select only boundary and metrics columns
        ds = ds.select_columns(
            [
                c
                for c in ds.column_names
                if c.startswith("boundary.") or c.startswith("metrics.")
            ]
        )

        # Filter for 3 field periods (as in the example)
        ds = ds.filter(
            lambda x: x == 3,
            input_columns=["boundary.n_field_periods"],
            num_proc=4,
        )

        # Remove unnecessary columns
        ds = ds.remove_columns(
            [
                "boundary.n_field_periods",
                "boundary.is_stellarator_symmetric",  # all same value
                "boundary.r_sin",
                "boundary.z_cos",  # empty
                "boundary.json",
                "metrics.json",
                "metrics.id",  # not needed
            ]
        )

        # Convert to list of dictionaries for easier splitting
        data = [dict(item) for item in ds]
        total_size = len(data)

        # Calculate split sizes
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size  # Use remainder for test

        print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

        # Split the data
        train_data, val_data, test_data = random_split(
            data,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),  # For reproducibility
        )

        # Create datasets
        self.train_dataset = ConstellarationDataset(
            train_data, self.grid_shape, self.num_channels
        )
        self.val_dataset = ConstellarationDataset(
            val_data, self.grid_shape, self.num_channels
        )
        self.test_dataset = ConstellarationDataset(
            test_data, self.grid_shape, self.num_channels
        )

        # Create dataloaders
        self.train_dataloader_instance = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.val_dataloader_instance = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.test_dataloader_instance = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        print(
            f"Created datasets: Train={len(self.train_dataset)}, "
            f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}"
        )

    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataloader_instance is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.train_dataloader_instance

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataloader_instance is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.val_dataloader_instance

    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_dataloader_instance is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.test_dataloader_instance


# Example usage
if __name__ == "__main__":
    # Create and test the datamodule
    datamodule = ConstellarationDataModule(
        batch_size=2,
        num_workers=0,  # Set to 0 for debugging
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )

    # Setup datasets and dataloaders
    datamodule.setup()

    # Test a few samples
    print("Testing dataloaders...")

    # Test training dataloader
    train_loader = datamodule.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"Training batch {i}:")
        print(f"  input_fields shape: {batch['input_fields'].shape}")
        print(f"  Available keys: {list(batch.keys())}")
        print(f"  Sample boundary field: {batch['boundary.r_cos'].shape}")
        print(f"  Sample metric: {batch['metrics.aspect_ratio'].shape}")
        if i >= 1:  # Just test first batch
            break

    # Test validation dataloader
    val_loader = datamodule.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"Validation batch {i}:")
        print(f"  input_fields shape: {batch['input_fields'].shape}")
        print(f"  Number of fields: {len(batch)}")
        if i >= 1:  # Just test first batch
            break

    print("DataModule test completed successfully!")
