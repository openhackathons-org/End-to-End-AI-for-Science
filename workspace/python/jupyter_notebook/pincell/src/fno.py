from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openmc
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from physicsnemo.models.fno import FNO


def build_model(cfg, device: torch.device) -> FNO:
    """
    Factory function to create an FNO model from config
    """
    return FNO(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        decoder_layers=cfg.decoder_layers,
        decoder_layer_size=cfg.decoder_layer_size,
        dimension=cfg.dimension,
        latent_channels=cfg.latent_channels,
        num_fno_layers=cfg.num_fno_layers,
        num_fno_modes=cfg.num_fno_modes,
    ).to(device)


class Preprocessor:
    """
    Handles the conversion of raw material masks and scalar parameters
    into the multi-channel tensor format required by the FNO.

    Input:
        - Mask: Integer tensor $$(B, H, W)$$
        - Enrichment: Scalar tensor $$(B)$$
    Output:
        - Tensor $$(B, C_{in}, H, W)$$ where $$C_{in}=4$$
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, mask, enrichment):
        mask = mask.to(self.device).long()
        enrichment = enrichment.to(self.device).float()
        materials = F.one_hot(mask, num_classes=3).permute(0, 3, 1, 2).float()
        B, _, H, W = materials.shape
        enrich_field = enrichment.view(B, 1, 1, 1).expand(B, 1, H, W)

        return torch.cat([materials, enrich_field], dim=1)


class PincellDataset(Dataset):
    """
    Custom torch.utils.data.Dataset to load Pincell data

    Notes:
    - Matches inputs from `input_dir` with outputs in `output_dir`
    - Returns raw flux values. Normalisation handled by Trainer
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        num_samples=None,
        resolution=1000,
        tally_name="physics",
        metadata_path="data/metadata.csv",
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.resolution = resolution
        self.tally_name = tally_name

        self.metadata = pd.read_csv(metadata_path, index_col="pincell_id")
        self.indices = self._discover_indices(limit=num_samples)

    def _discover_indices(self, limit=None):
        """
        Return sorted simulation indices
        """
        folders = Path(self.output_dir).glob("pincell_*")
        indices = []

        for folder in folders:
            idx = self._parse_simulation_idx(folder)

            if folder.is_dir() and self._input_exists(idx):
                indices.append(idx)

        indices.sort()
        return indices[:limit] if limit else indices

    @staticmethod
    def _parse_simulation_idx(folder):
        """
        Parse integer index
        """
        suffix = folder.name.rsplit("_", 1)[-1]
        return int(suffix)

    def _input_exists(self, idx):
        """
        Check if the expected input file exists for idx
        """
        return (Path(self.input_dir) / f"pincell_{idx}_masks.npy").exists()

    def _statepoint_path(self, sim_idx):
        """
        Return statepoint path for sim_idx
        """
        return os.path.join(self.output_dir, f"pincell_{sim_idx}", "statepoint.100.h5")

    def _cached_output_path(self, sim_idx, score):
        return os.path.join(
            self.output_dir, f"pincell_{sim_idx}", f"{score}_{self.resolution}.npy"
        )

    def _load_input(self, path):
        mask = torch.from_numpy(np.load(path)).float().squeeze(-1)[None]
        mask = F.interpolate(
            mask, size=(self.resolution, self.resolution), mode="nearest"
        )
        return torch.argmax(mask[0], dim=0)

    def _load_tally_score(self, tally, score: str, mesh_shape: tuple) -> torch.Tensor:
        """
        Extract and resize a single score from a tally.

        Args:
            tally: OpenMC tally object
            score: Score name (e.g., 'flux', 'absorption')
            mesh_shape: Shape of the mesh filter

        Returns:
            Tensor of shape (resolution, resolution)
        """
        field = tally.get_values(scores=[score]).reshape(mesh_shape)[:, :, 0].T
        field = torch.from_numpy(field).float()

        if field.shape[0] != self.resolution:
            field = F.interpolate(
                field[None, None],
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        return field

    def _load_output(self, sim_idx):
        """
        Load flux and per-cell sigma_a fields, using cached .npy files
        when available. On first access, extracts from the statepoint HDF5,
        computes sigma_a = absorption / flux, and caches both to .npy.

        Returns:
            tuple: (flux, sigma_a) tensors at target resolution
        """
        flux_cache = self._cached_output_path(sim_idx, "flux")
        sigma_a_cache = self._cached_output_path(sim_idx, "sigma_a")

        if os.path.exists(flux_cache) and os.path.exists(sigma_a_cache):
            flux = torch.from_numpy(np.load(flux_cache))
            sigma_a = torch.from_numpy(np.load(sigma_a_cache))
            return flux, sigma_a

        with openmc.StatePoint(self._statepoint_path(sim_idx)) as sp:
            tally = sp.get_tally(name=self.tally_name)
            mesh_shape = tally.filters[0].mesh.dimension

            flux = self._load_tally_score(tally, "flux", mesh_shape)
            absorption = self._load_tally_score(tally, "absorption", mesh_shape)

        safe_flux = torch.where(flux > 0, flux, torch.ones_like(flux))
        sigma_a = torch.where(flux > 0, absorption / safe_flux, torch.zeros_like(flux))

        np.save(flux_cache, flux.numpy())
        np.save(sigma_a_cache, sigma_a.numpy())

        return flux, sigma_a

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (mask, enrichment, flux, sigma_a, xs)
                - mask: Material mask (H, W) as LongTensor
                - enrichment: Fuel enrichment as scalar tensor
                - flux: Neutron flux field (H, W)
                - sigma_a: Per-cell macroscopic absorption XS (H, W)
                - xs: Homogenised absorption XS (scalar)
        """
        sim_idx = self.indices[idx]
        input_path = os.path.join(self.input_dir, f"pincell_{sim_idx}_masks.npy")

        mask = self._load_input(input_path)
        flux, sigma_a = self._load_output(sim_idx)

        enrichment = self.metadata.loc[sim_idx, "enrichment_wpct"]
        xs = self.metadata.loc[sim_idx, "xs"]

        return (
            mask.long(),
            torch.tensor(enrichment, dtype=torch.float32),
            flux.float(),
            sigma_a.float(),
            xs,
        )


class Trainer:
    """
    Trainer with per-channel log-space standardisation.

    Both flux and sigma_a are transformed as:
        normalised = (log(field + eps) - log_mean) / log_std

    Statistics are computed independently per output channel so that
    each field is well-scaled for training.
    """

    EPS = 1e-10

    def __init__(self, cfg, train_dataset, val_dataset):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = Preprocessor(self.device)

        self.writer = SummaryWriter(log_dir="runs/fno_experiment")
        self.model = build_model(cfg, self.device)

        self.loss_fun = MSELoss(reduction="mean")
        self.optimiser = Adam(self.model.parameters(), lr=self.cfg.initial_lr)

        self.log_mean = None
        self.log_std = None

        self._setup_data(train_dataset, val_dataset)

    def _setup_data(self, train_dataset, val_dataset):
        self._compute_statistics(train_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def _compute_statistics(self, training_dataset):
        """
        Compute per-channel mean and std of log-transformed targets.
        Channel 0 = flux, Channel 1 = sigma_a.
        """
        all_log_flux = []
        all_log_sigma_a = []

        for i in tqdm(range(len(training_dataset)), desc="Computing statistics"):
            _, _, flux, sigma_a, _ = training_dataset[i]
            all_log_flux.append(torch.log(flux + self.EPS).flatten())
            all_log_sigma_a.append(torch.log(sigma_a + self.EPS).flatten())

        all_log_flux = torch.cat(all_log_flux)
        all_log_sigma_a = torch.cat(all_log_sigma_a)

        self.log_mean = torch.tensor(
            [all_log_flux.mean().item(), all_log_sigma_a.mean().item()]
        )
        self.log_std = torch.tensor(
            [all_log_flux.std().item(), all_log_sigma_a.std().item()]
        )

        print(
            f"Log-space statistics:\n"
            f"  flux     — mean={self.log_mean[0]:.4f}, std={self.log_std[0]:.4f}\n"
            f"  sigma_a  — mean={self.log_mean[1]:.4f}, std={self.log_std[1]:.4f}"
        )

    def _normalise(self, target: torch.Tensor) -> torch.Tensor:
        """
        Per-channel log-space standardisation.
        target: (B, C, H, W) where C matches len(log_mean).
        """
        mean = self.log_mean.to(target.device).view(1, -1, 1, 1)
        std = self.log_std.to(target.device).view(1, -1, 1, 1)
        return (torch.log(target + self.EPS) - mean) / std

    def _denormalise(self, normalised: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _normalise.
        """
        mean = self.log_mean.to(normalised.device)
        std = self.log_std.to(normalised.device)
        if normalised.ndim == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        return torch.exp(normalised * std + mean) - self.EPS

    def _build_target(self, batch):
        """
        Stack flux and sigma_a into a (B, 2, H, W) target tensor.
        """
        mask, enrichment, flux, sigma_a, _ = batch
        flux = flux.to(self.device)
        sigma_a = sigma_a.to(self.device)
        if flux.ndim == 3:
            flux = flux.unsqueeze(1)
        if sigma_a.ndim == 3:
            sigma_a = sigma_a.unsqueeze(1)
        return mask, enrichment, torch.cat([flux, sigma_a], dim=1)

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                mask, enrichment, target = self._build_target(batch)
                inputs = self.preprocessor(mask, enrichment)
                prediction = self.model(inputs)
                loss = self.loss_fun(prediction, self._normalise(target))
                total_val_loss += loss.item()

        return total_val_loss / len(self.val_loader)

    def train(self):
        epochs = self.cfg.epochs
        print(f"Starting training on {self.device}...")

        train_history = []
        val_history = []

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            total_train_loss = 0

            for batch in self.train_loader:
                mask, enrichment, target = self._build_target(batch)
                inputs = self.preprocessor(mask, enrichment)

                self.optimiser.zero_grad()
                prediction = self.model(inputs)

                loss = self.loss_fun(prediction, self._normalise(target))
                loss.backward()
                self.optimiser.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            train_history.append(avg_train_loss)
            val_history.append(avg_val_loss)

            self.writer.add_scalars(
                "Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch
            )

            if (epoch + 1) % 10 == 0:
                tqdm.write(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Validation Loss: {avg_val_loss:.6f}"
                )

        self.writer.close()
        self._save_checkpoint()
        self.plot_loss_history(train_history, val_history)

    def _save_checkpoint(self):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "log_mean": self.log_mean.tolist(),
            "log_std": self.log_std.tolist(),
            "eps": self.EPS,
        }
        torch.save(ckpt, "fno_flux_model.pth")
        print("Checkpoint saved to fno_flux_model.pth")

    def plot_loss_history(self, train_history, val_history):
        """
        Plot training and validation loss
        """
        epochs = len(train_history)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_history, label="Training Loss")
        plt.plot(range(1, epochs + 1), val_history, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss (normalised log-space)")
        plt.legend()
        plt.grid(True)


class Inference:
    """
    Loads a trained multi-output FNO model to predict both the
    neutron flux field and the macroscopic absorption cross-section map.

    Predictions are made in normalised log-space per channel, then
    transformed back to physical values using saved statistics.
    """

    def __init__(self, cfg, model_path="fno_flux_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = Preprocessor(self.device)
        self.model = build_model(cfg, self.device)

        self.log_mean = None
        self.log_std = None
        self.eps = 1e-10

        self._load_checkpoint(model_path)

    def _load_checkpoint(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        raw_mean = ckpt["log_mean"]
        raw_std = ckpt["log_std"]
        self.log_mean = torch.tensor(raw_mean) if not isinstance(raw_mean, torch.Tensor) else raw_mean
        self.log_std = torch.tensor(raw_std) if not isinstance(raw_std, torch.Tensor) else raw_std
        self.eps = ckpt.get("eps", 1e-10)

        self.model.eval()
        print(f"Model loaded from {model_path}")
        print(f"Normalisation stats (per channel): log_mean={self.log_mean.tolist()}, log_std={self.log_std.tolist()}")

    def _denormalise(self, normalised: torch.Tensor) -> torch.Tensor:
        mean = self.log_mean.to(normalised.device)
        std = self.log_std.to(normalised.device)
        if normalised.ndim == 4:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        return torch.clamp(torch.exp(normalised * std + mean) - self.eps, min=0.0)

    def predict(self, mask, enrichment):
        """
        Predict flux and sigma_a fields for a given geometry.

        Returns:
            tuple: (flux, sigma_a) tensors, each of shape (H, W)
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        with torch.no_grad():
            inputs = self.preprocessor(mask, enrichment)
            normalised_pred = self.model(inputs)
            pred = self._denormalise(normalised_pred)

        pred = pred.squeeze(0).cpu()
        return pred[0], pred[1]

    def predict_from_dataset(self, dataset, idx):
        mask, enrichment, gt_flux, gt_sigma_a, _ = dataset[idx]
        flux_pred, sigma_a_pred = self.predict(mask, enrichment)

        return {
            "flux_pred": flux_pred,
            "sigma_a_pred": sigma_a_pred,
            "flux_gt": gt_flux,
            "sigma_a_gt": gt_sigma_a,
            "mask": mask,
            "enrichment": enrichment.item(),
        }

    def evaluate(self, dataset, num_samples=None):
        """
        Evaluate model on a dataset. Reports per-channel relative L2
        errors and MSE for both flux and sigma_a.
        """
        num_samples = num_samples or len(dataset)
        num_samples = min(num_samples, len(dataset))

        flux_rel, sigma_a_rel = [], []

        for idx in tqdm(range(num_samples), desc="Evaluating"):
            result = self.predict_from_dataset(dataset, idx)

            for key_pred, key_gt, err_list in [
                ("flux_pred", "flux_gt", flux_rel),
                ("sigma_a_pred", "sigma_a_gt", sigma_a_rel),
            ]:
                pred, gt = result[key_pred], result[key_gt]
                rel = (torch.norm(pred - gt) / (torch.norm(gt) + 1e-10)).item()
                err_list.append(rel)

        return {
            "flux_mean_rel_error": np.mean(flux_rel),
            "sigma_a_mean_rel_error": np.mean(sigma_a_rel),
            "num_samples": num_samples,
        }

    def plot_comparison(self, dataset, idx, save_path=None):
        """
        Plot predicted vs ground truth for both output channels.
        """
        result = self.predict_from_dataset(dataset, idx)

        fields = [
            ("Flux (GT)", result["flux_gt"], "hot"),
            ("Flux (Pred)", result["flux_pred"], "hot"),
            (r"$\Sigma_a$ (GT)", result["sigma_a_gt"], "viridis"),
            (r"$\Sigma_a$ (Pred)", result["sigma_a_pred"], "viridis"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for ax, (title, field, cmap) in zip(axes, fields):
            im = ax.imshow(field.numpy(), cmap=cmap, origin="lower")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        flux_rel = torch.norm(result["flux_pred"] - result["flux_gt"]) / (
            torch.norm(result["flux_gt"]) + 1e-10
        )
        fig.suptitle(f"Sample {idx} | Flux Relative L2 Error: {flux_rel:.4f}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved comparison plot to {save_path}")
        else:
            plt.show()

        plt.close()
