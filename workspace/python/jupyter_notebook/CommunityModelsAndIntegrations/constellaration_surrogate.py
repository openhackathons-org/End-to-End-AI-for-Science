import torch
import torch.nn as nn


class TFNOSurrogate(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_dim=3,
        hidden_dim=256,
        target_input_channels=28,
        target_output_channels=7,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.target_input_channels = target_input_channels
        self.target_output_channels = target_output_channels

        # Input projection: 3D conv to project from input_channels to target_input_channels
        # This projects the 3D volume to match the pretrained TFNO's expected input
        self.input_projection = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(hidden_dim // 2, target_input_channels, kernel_size=1),
        )

        # Output projection: 3D conv to project from target_output_channels to output_dim
        # This projects the TFNO output to our target output size
        self.output_projection = nn.Sequential(
            nn.Conv3d(target_output_channels, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(
                hidden_dim, 1, kernel_size=1
            ),  # Single channel for global pooling
        )

        # Global pooling layer to convert 3D output to 1D
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Final projection to output dimension
        self.final_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.surrogate = self.setup_model()  # pretrained TFNOPhysicsNeMoModel
        self.freeze_surrogate_weights()

    def setup_model(self):
        """Setup the pretrained TFNO model from The Well as a PhysicsNeMo model.
        Model input parameters are set from the well, and the original config is updated to reflect this."""
        import inspect

        from physicsnemo.models.meta import ModelMetaData
        from physicsnemo.models.module import Module
        from the_well.benchmark.models import TFNO

        well_model = TFNO.from_pretrained("polymathic-ai/TFNO-MHD_64")
        model_dict = well_model.__dict__

        signature = inspect.signature(TFNO)
        parameters = signature.parameters
        filtered_params = {k: model_dict[k] for k in parameters if k in model_dict}

        model = Module.from_torch(TFNO, meta=ModelMetaData(name="converted_tfno"))
        well_pretrained_model = model(**filtered_params)
        well_pretrained_model.inner_model.load_state_dict(
            well_model.state_dict(), strict=True
        )
        return well_pretrained_model

    def freeze_surrogate_weights(self):
        """Freeze all parameters in the pretrained TFNO surrogate model."""
        for param in self.surrogate.parameters():
            param.requires_grad = False

        # Also freeze the surrogate model itself to prevent any updates
        self.surrogate.eval()  # Set to evaluation mode

        print(
            f"Frozen {sum(p.numel() for p in self.surrogate.parameters())} parameters in pretrained TFNO model"
        )

    def forward(self, x):
        # Input: [batch_size, channels, height, width, depth] from FullConstellarationDataset
        # Expected: [batch_size, 3, 64, 64, 64] -> [batch_size, 28, 64, 64, 64]

        # Input projection: 3D conv to match pretrained TFNO input channels
        x_projected = self.input_projection(x)  # [batch_size, 28, 64, 64, 64]

        # Pass through the pretrained TFNO model
        tfno_output = self.surrogate(x_projected)  # [batch_size, 7, 64, 64, 64]

        # Output projection: 3D conv to prepare for global pooling
        tfno_projected = self.output_projection(
            tfno_output
        )  # [batch_size, 1, 64, 64, 64]

        # Global average pooling: [batch_size, 1, 64, 64, 64] -> [batch_size, 1, 1, 1, 1]
        pooled = self.global_pool(tfno_projected)  # [batch_size, 1, 1, 1, 1]

        # Flatten: [batch_size, 1, 1, 1, 1] -> [batch_size, 1]
        flattened = pooled.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch_size, 1]

        # Final projection to output dimension: [batch_size, 1] -> [batch_size, output_dim]
        output = self.final_projection(flattened)  # [batch_size, 3]

        return output


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing TFNO Surrogate with 3D Input ===")

    # Test with 3D input from FullConstellarationDataset
    batch_size = 2
    input_channels = 3
    spatial_size = 64

    # Create sample 3D input: [batch_size, channels, height, width, depth]
    sample_input = torch.randn(
        batch_size, input_channels, spatial_size, spatial_size, spatial_size
    )

    print(f"Sample input shape: {sample_input.shape}")

    # Test both versions
    print("\n--- Full Version ---")
    surrogate_full = TFNOSurrogate(input_channels=input_channels, output_dim=3)
    with torch.no_grad():
        output_full = surrogate_full(sample_input)
        print(f"Full surrogate output shape: {output_full.shape}")

    print("\n=== Model Architectures ===")
    print("Full TFNO Surrogate:")
    print(surrogate_full)
