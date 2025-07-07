import logging
import os
from typing import Dict, Optional, Tuple, Union

import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.sym.hydra import to_absolute_path
from the_well.data import WellDataModule
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from data_utils import DummyWellDataModule

logging.basicConfig(level=logging.INFO)


dtype = torch.float
torch.set_default_dtype(dtype)


class Trainer:
    """
    Modular trainer class for MHD (Magnetohydrodynamics) training using TFNO.

    This class encapsulates all training components including model setup,
    data loading, training/validation loops, and logging functionality.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the MHD trainer with configuration.

        Args:
            cfg: Hydra configuration object containing all training parameters
        """
        self.cfg = cfg
        self.dist = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.datamodule = None
        self.sampler_train = None
        self.log = None
        self.output_dir = None
        self.names = None
        self.loaded_epoch = 0
        self.dummy_data = None

        # Extract configuration parameters
        self._extract_config()

        # Initialize components
        self._setup_distributed()
        self._setup_logging()
        self._setup_data()
        self._empyt_registry()
        self.setup_model()
        self._setup_distributed_model()
        self._setup_optimizer()
        self._setup_loss()
        self._load_checkpoint()

    def _extract_config(self):
        """Extract and store configuration parameters."""
        self.model_params = self.cfg.model_params
        self.dataset_params = self.cfg.dataset_params
        self.dataloader_params = self.cfg.dataloader_params
        self.optimizer_params = self.cfg.optimizer_params
        self.train_params = self.cfg.train_params
        self.dummy_data = self.cfg.dummy_data

        self.load_ckpt = self.cfg.load_ckpt
        self.output_dir = self.cfg.output_dir
        self.training = self.cfg.training

        self.output_dir = to_absolute_path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_dir = self.dataset_params.well_base_path
        self.ckpt_path = self.train_params.ckpt_path

    def _empyt_registry(self):
        from physicsnemo.registry import ModelRegistry

        model_registry = ModelRegistry()
        model_registry.__clear_registry__()
        model_registry.__restore_registry__()

    def _setup_distributed(self):
        """Initialize distributed training setup."""
        DistributedManager.initialize()
        self.dist = DistributedManager()

    def _setup_logging(self):
        """Setup logging and monitoring."""
        self.log = PythonLogger(name="mhd_pino")
        self.log.file_logging()

        LaunchLogger.initialize(use_wandb=self.cfg.use_wandb)

    def _setup_data(self):
        """Setup datasets and dataloaders."""
        # Use dummy datamodule for testing
        if self.dummy_data:
            datamodule = DummyWellDataModule(
                num_train_samples=8,
                num_val_samples=2,
                num_test_samples=2,
                batch_size=self.dataset_params.batch_size,
                num_workers=self.dataloader_params.num_workers,
            )
            # For dummy data, we don't have normalization
            self.dset_norm = None
        else:
            print("Setting up datamodule...")
            print("Instantiating datamodule...")
            datamodule: WellDataModule = instantiate(
                self.dataset_params,
                world_size=self.dist.world_size,
                rank=self.dist.rank,
                data_workers=self.dataloader_params.num_workers,
            )
            print("Datamodule instantiated successfully!")
            # Get normalization from the dataset
            self.dset_norm = datamodule.train_dataset.norm
            print(f"Dataset length: {len(datamodule.train_dataset)}")
            print("Testing dataset access...")
            try:
                _ = datamodule.train_dataset[0]
                print("Successfully accessed first sample!")
            except Exception as e:
                print(f"Error accessing dataset: {e}")
                import traceback

                traceback.print_exc()

        self.datamodule = datamodule

    def setup_model(self):
        """Setup the TFNO model."""
        raise NotImplementedError("This method is not implemented!")

    def _setup_distributed_model(self):
        # Setup DistributedDataParallel if using multiple processes
        if self.dist.distributed:
            ddps = torch.cuda.Stream()
            with torch.cuda.stream(ddps):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                )
            torch.cuda.current_stream().wait_stream(ddps)

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            betas=self.optimizer_params.betas,
            lr=self.optimizer_params.lr,
            weight_decay=0.1,
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.optimizer_params.milestones,
            gamma=self.optimizer_params.gamma,
        )

    def _setup_loss(self):
        """Setup loss function - using simple MSE loss."""
        self.loss_fn = torch.nn.MSELoss()

    def _load_checkpoint(self):
        """Load model from checkpoint if specified."""
        if self.load_ckpt:
            self.loaded_epoch = load_checkpoint(
                self.ckpt_path,
                self.model,
                self.optimizer,
                self.scheduler,
                device=self.dist.device,
            )

    def normalize(self, batch):
        """Normalize batch using dataset normalization if available."""
        if hasattr(self, "dset_norm") and self.dset_norm:
            batch["input_fields"] = self.dset_norm.normalize_flattened(
                batch["input_fields"], "variable"
            )
            if "constant_fields" in batch:
                batch["constant_fields"] = self.dset_norm.normalize_flattened(
                    batch["constant_fields"], "constant"
                )
        return batch

    def denormalize(self, batch):
        """Denormalize batch using dataset normalization if available."""
        if hasattr(self, "dset_norm") and self.dset_norm:
            batch = self.dset_norm.denormalize_flattened(batch, "variable")
            # if "constant_fields" in batch:
            #     batch["constant_fields"] = self.dset_norm.denormalize_flattened(
            #         batch["constant_fields"], "constant"
            #     )

        return batch

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform forward pass through the model.

        Args:
            batch: Dictionary containing input data with keys like 'input_fields', 'output_fields', etc.

        Returns:
            Model predictions
        """
        # Normalize the batch
        normalized_batch = self.normalize(batch.copy())

        # Extract input fields from normalized batch
        inputs = normalized_batch[
            "input_fields"
        ]  # Shape: (batch, time, x, y, z, fields)

        # Rearrange for model: (batch, time, x, y, z, fields) -> (batch, (time fields), x, y, z)
        model_inputs = rearrange(inputs, "B Ti Lx Ly Lz F -> B (Ti F) Lx Ly Lz")

        # Forward pass through model
        model_outputs = self.model(model_inputs)

        # Rearrange back: (batch, (time fields), x, y, z) -> (batch, time, x, y, z, fields)
        outputs = rearrange(
            model_outputs,
            "B (Tp F) Lx Ly Lz -> B Tp Lx Ly Lz F",
            F=self.model_params.out_dim,
        )

        # Denormalize outputs
        outputs = self.denormalize(outputs)

        return outputs

    def _training_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing input and output data

        Returns:
            Tuple of (loss, loss_dict)
        """
        self.optimizer.zero_grad()

        # Forward pass (includes normalization)
        pred = self._forward_pass(batch)

        # Get target outputs (already denormalized in _forward_pass)
        targets = batch["output_fields"]

        # Compute loss
        loss = self.loss_fn(pred, targets)
        loss_dict = {"loss": loss.item()}

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss, loss_dict

    def _validation_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single validation step.

        Args:
            batch: Dictionary containing input and output data

        Returns:
            Tuple of (loss, loss_dict)
        """
        with torch.no_grad():
            # Forward pass (includes normalization)
            pred = self._forward_pass(batch)
            targets = batch["output_fields"]
            loss = self.loss_fn(pred, targets)
            loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing training metrics
        """
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(self.datamodule.train_dataloader()),
            epoch_alert_freq=1,
        ) as log:
            # if self.dist.distributed:
            #     self.sampler_train.set_epoch(epoch)

            self.model.train()
            dataloader = self.datamodule.train_dataloader()
            print(f"Starting training epoch {epoch} with {len(dataloader)} batches")
            for i, batch in enumerate(dataloader):
                # Move batch to device
                batch = {
                    k: v.type(torch.FloatTensor).to(self.dist.device)
                    for k, v in batch.items()
                }

                loss, loss_dict = self._training_step(batch)
                log.log_minibatch(loss_dict)

            log.log_epoch({"Learning Rate": self.optimizer.param_groups[0]["lr"]})
            self.scheduler.step()

            return {"learning_rate": self.optimizer.param_groups[0]["lr"]}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing validation metrics
        """
        with LaunchLogger("valid", epoch=epoch) as log:
            self.model.eval()

            with torch.no_grad():
                for i, batch in enumerate(self.datamodule.val_dataloader()):
                    # Move batch to device
                    batch = {
                        k: v.type(dtype).to(self.dist.device) for k, v in batch.items()
                    }

                    # Forward pass (includes normalization)
                    pred = self._forward_pass(batch)
                    targets = batch["output_fields"]
                    loss = self.loss_fn(pred, targets)
                    loss_dict = {"loss": loss.item()}

                    log.log_minibatch(loss_dict)

            return loss_dict

    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
        """
        if epoch % self.train_params.ckpt_freq == 0 and self.dist.rank == 0:
            save_checkpoint(
                self.ckpt_path, self.model, self.optimizer, self.scheduler, epoch=epoch
            )

    def train(
        self, num_epochs: Optional[int] = None
    ) -> Union[Dict[str, float], Dict[str, float]]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train for (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.train_params.epochs

        start_epoch = max(1, self.loaded_epoch + 1)

        for epoch in range(start_epoch, num_epochs + 1):
            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_metrics = self.validate_epoch(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch)

        return train_metrics, val_metrics

    def evaluate(self, dataloader=None) -> Dict[str, float]:
        """
        Evaluate the model on a given dataloader.

        Args:
            dataloader: Dataloader to evaluate on (uses validation dataloader if None)

        Returns:
            Dictionary containing evaluation metrics
        """
        if dataloader is None:
            dataloader = self.datamodule.val_dataloader()

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {
                    k: v.type(dtype).to(self.dist.device) for k, v in batch.items()
                }

                # Forward pass (includes normalization)
                pred = self._forward_pass(batch)
                targets = batch["output_fields"]
                loss = self.loss_fn(pred, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"evaluation_loss": avg_loss}

    def get_model(self):
        """Get the trained model."""
        return self.model

    def get_optimizer(self):
        """Get the optimizer."""
        return self.optimizer

    def get_scheduler(self):
        """Get the scheduler."""
        return self.scheduler


# @hydra.main(
#     version_base="1.3", config_path="config", config_name="pnm_model_well_data.yaml"
# )
# def main(cfg: DictConfig) -> None:
#     """Training for the MHD problem.

#     This training script demonstrates how to set up a data-driven model for a 2D MHD flow
#     using Tensor Factorized Fourier Neural Operators (TFNO) and acts as a benchmark for this type of operator.
#     Training data is generated in-situ via the MHD data loader from PhysicsNeMo. The model is trained
#     over multiple epochs with validation and checkpointing.
#     """

#     # Create trainer instance
#     trainer = Trainer(cfg)

#     # Start training
#     trainer.train()


# if __name__ == "__main__":
#     main()
