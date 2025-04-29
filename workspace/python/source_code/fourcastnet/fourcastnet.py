# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Script to train Fourcastnet on ERA5
# Ref: https://arxiv.org/abs/2202.11214

import physicsnemo

from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.utils.io import GridValidatorPlotter

from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch
from src.loss import LpLoss


@physicsnemo.sym.main(config_path="conf", config_name="config_FCN")
def run(cfg: PhysicsNeMoConfig) -> None:

    # load training/ test data
    channels = list(range(cfg.custom.n_channels))
    train_dataset = ERA5HDF5GridDataset(
        cfg.custom.training_data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
    )
    test_dataset = ERA5HDF5GridDataset(
        cfg.custom.test_data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
        n_samples_per_year=20,
    )

    # define input/output keys
    input_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.invar_keys]
    output_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.outvar_keys]

    # make list of nodes to unroll graph on
    model = FourcastNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        img_shape=test_dataset.img_shape,
        patch_size=cfg.arch.afno.patch_size,
        embed_dim=cfg.arch.afno.embed_dim,
        depth=cfg.arch.afno.depth,
        num_blocks=cfg.arch.afno.num_blocks,
    )
    nodes = [model.make_node(name="FCN")]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        loss=LpLoss(),
        num_workers=cfg.custom.num_workers.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        num_workers=cfg.custom.num_workers.validation,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
