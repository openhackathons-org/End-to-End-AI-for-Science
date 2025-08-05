import torch

from torch.utils.data import DataLoader, Dataset


class DummyWellDataset(Dataset):
    """Dummy dataset for testing with the same structure as the real Well dataset."""

    def __init__(
        self,
        num_samples=10,
        input_time_steps=4,
        output_time_steps=1,
        spatial_size=64,
        num_fields=7,
        num_constant_scalars=2,
        num_boundary_conditions=3,
    ):
        self.num_samples = num_samples
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.spatial_size = spatial_size
        self.num_fields = num_fields
        self.num_constant_scalars = num_constant_scalars
        self.num_boundary_conditions = num_boundary_conditions

        # Mock metadata to match the real dataset structure
        self.metadata = type(
            "Metadata",
            (),
            {
                "field_names": {
                    0: ["density"],
                    1: [
                        "magnetic_field_x",
                        "magnetic_field_y",
                        "magnetic_field_z",
                        "velocity_x",
                        "velocity_y",
                        "velocity_z",
                    ],
                    2: [],
                }
            },
        )()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a dummy sample with the same structure as the real dataset."""
        # Generate random data with appropriate shapes
        sample = {
            "input_fields": torch.randn(
                self.input_time_steps,
                self.spatial_size,
                self.spatial_size,
                self.spatial_size,
                self.num_fields,
            ),
            "output_fields": torch.randn(
                self.output_time_steps,
                self.spatial_size,
                self.spatial_size,
                self.spatial_size,
                self.num_fields,
            ),
            "constant_scalars": torch.randn(self.num_constant_scalars),
            "boundary_conditions": torch.randn(self.num_boundary_conditions, 2),
            "space_grid": torch.randn(
                self.spatial_size, self.spatial_size, self.spatial_size, 3
            ),
            "input_time_grid": torch.linspace(0, 1, self.input_time_steps),
            "output_time_grid": torch.linspace(1, 2, self.output_time_steps),
        }

        return sample


class DummyWellDataModule:
    """Dummy datamodule for testing that mimics the WellDataModule interface."""

    def __init__(
        self,
        num_train_samples=8,
        num_val_samples=2,
        num_test_samples=2,
        batch_size=2,
        num_workers=0,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create datasets
        self.train_dataset = DummyWellDataset(num_samples=num_train_samples)
        self.val_dataset = DummyWellDataset(num_samples=num_val_samples)
        self.test_dataset = DummyWellDataset(num_samples=num_test_samples)

        # Create dataloaders
        self.train_dataloader_instance = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )

        self.val_dataloader_instance = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        self.test_dataloader_instance = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    def train_dataloader(self):
        return self.train_dataloader_instance

    def val_dataloader(self):
        return self.val_dataloader_instance

    def test_dataloader(self):
        return self.test_dataloader_instance

    def setup(self, stage=None):
        """Setup method to match the interface."""
        pass
