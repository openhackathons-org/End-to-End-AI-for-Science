# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import plotly
import torch
import wandb
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.sym.hydra import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
import time 

from dataloaders import Dedalus2DDataset, MHDDataloaderVecPot
from losses import LossMHDVecPot_PhysicsNeMo
from tfno import TFNO
from utils.plot_utils import plot_predictions_mhd, plot_predictions_mhd_plotly

dtype = torch.float # dtype = torch.double
torch.set_default_dtype(dtype)


@hydra.main(
    version_base="1.3", config_path="config", config_name="eval_mhd_vec_pot_tfno_Re100.yaml"
)
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="mhd_pino")
    log.file_logging()

    wandb_dir = cfg.wandb_params.wandb_dir
    wandb_project = cfg.wandb_params.wandb_project
    wandb_group = cfg.wandb_params.wandb_group

    initialize_wandb(
        project=wandb_project,
        entity="pino_evaluation",
        mode="offline",
        group=wandb_group,
        config=dict(cfg),
        results_dir=wandb_dir,
    )

    LaunchLogger.initialize(use_wandb=cfg.use_wandb)  # PhysicsNeMo launch logger

    # Load config file parameters
    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
    test_loader_params = cfg.test_loader_params
    loss_params = cfg.loss_params
    optimizer_params = cfg.optimizer_params
    wandb_params = cfg.wandb_params

    load_ckpt = cfg.load_ckpt
    output_dir = cfg.output_dir
    use_wandb = cfg.use_wandb
    test_params = cfg.test

    output_dir = to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = dataset_params.data_dir
    wandb_dir = wandb_params.wandb_dir
    wandb_project = wandb_params.wandb_group
    wandb_group = wandb_params.wandb_project

    # Construct dataloaders
    dataset_test = Dedalus2DDataset(
        data_dir,
        output_names=dataset_params.output_names,
        field_names=dataset_params.field_names,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )
    mhd_dataloader_test = MHDDataloaderVecPot(
        dataset_test,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
    )

    dataloader_test, sampler_test = mhd_dataloader_test.create_dataloader(
        batch_size=test_loader_params.batch_size,
        shuffle=test_loader_params.shuffle,
        num_workers=test_loader_params.num_workers,
        pin_memory=test_loader_params.pin_memory,
        distributed=dist.distributed,
    )

    # define FNO model
    model = TFNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=model_params.modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
        rank=model_params.rank,
        factorization=model_params.factorization,
        fixed_rank_modes=model_params.fixed_rank_modes,
        decomposition_kwargs=model_params.decomposition_kwargs,
    ).to(dist.device)

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                # the local rank of this process on
                # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Construct optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        betas=optimizer_params.betas,
        lr=optimizer_params.lr,
        weight_decay=0.1,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=optimizer_params.milestones, gamma=optimizer_params.gamma
    )

    # Construct Loss class
    mhd_loss = LossMHDVecPot_PhysicsNeMo(**loss_params)

    # Load model from checkpoint (if exists)
    if load_checkpoint:
        loaded_epoch = load_checkpoint(
            test_params.ckpt_path, model, optimizer, scheduler, device=dist.device
        )

    # Eval Loop
    names = dataset_params.fields
    input_norm = torch.tensor(model_params.input_norm).to(dist.device)
    output_norm = torch.tensor(model_params.output_norm).to(dist.device)

    with LaunchLogger("test") as log:
        # Val loop
        model.eval()
        test_loss_dict = {}
        plot_count = 0
        plot_dict = {name: {} for name in names}
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(dataloader_test):
                inputs = inputs.type(dtype).to(dist.device)
                outputs = outputs.type(dtype).to(dist.device)
                start_time = time.time()
                # Compute Predictions
                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )
                end_time = time.time()
                print(f"Inference Time: {end_time-start_time}")
                # Compute Loss
                loss, loss_dict = mhd_loss(
                    pred, outputs, inputs, return_loss_dict=True
                )

                log.log_minibatch(loss_dict)

                # Get prediction plots to log for wandb
                for j, _ in enumerate(pred):
                    # Make plots for each field
                    for index, name in enumerate(names):
                        # Generate figure
                        if use_wandb:
                            figs = plot_predictions_mhd_plotly(
                                pred[j].cpu(),
                                outputs[j].cpu(),
                                inputs[j].cpu(),
                                index=index,
                                name=name,
                            )
                            # Add figure to plot dict
                            plot_dict[name] = {
                                f"{plot_type}-{plot_count}": wandb.Html(
                                    plotly.io.to_html(fig)
                                )
                                for plot_type, fig in zip(
                                    wandb_params.wandb_plot_types, figs
                                )
                            }

                    plot_count += 1

                # Get prediction plots and save images locally
                for j, _ in enumerate(pred):
                    # Generate figure
                    plot_predictions_mhd(
                        pred[j].cpu(),
                        outputs[j].cpu(),
                        inputs[j].cpu(),
                        names=names,
                        save_path=os.path.join(
                            output_dir,
                            "MHD_" + cfg.derivative + "_" + str(dist.rank),
                        ),
                        save_suffix=i,
                    )

        if use_wandb:
            wandb.log({"plots": plot_dict})


if __name__ == "__main__":
    main()
