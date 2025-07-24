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
import warnings

import numpy as np
from sympy import Symbol, Eq, And, Or

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    ns = NavierStokes() # Define the Navier-Stokes equations
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5

    x, y = Symbol("x"), Symbol("y")
    channel = Channel2D((channel_length[0], channel_width[0]), (channel_length[1], channel_width[1]))
    rec = Rectangle((chip_pos, channel_width[0]), (chip_pos + chip_width, channel_width[0] + chip_height))
    geo = channel - rec
    
    inlet = Line((channel_length[0], channel_width[0]), (channel_length[0], channel_width[1]), normal=1)
    outlet = Line((channel_length[1], channel_width[0]), (channel_length[1], channel_width[1]), normal=1)
    x_pos = Symbol("x_pos")
    integral_line = Line((x_pos, channel_width[0]), (x_pos, channel_width[1]), 1)
    x_pos_range = {x_pos: lambda batch_size: np.full((batch_size, 1), np.random.uniform(channel_length[0], channel_length[1]))}

    domain = Domain()

    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={}, #inlet boundary condition 
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={}, #outlet boundary condition
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={}, #no slip boundary condition
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={}, #PDE constraint
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    file_path = "examples_sym/examples/chip_2d/openfoam/2D_chip_fluid0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] -= 2.5  # normalize pos
        openfoam_var["y"] -= 0.5
        openfoam_invar_numpy = {key: value for key, value in openfoam_var.items() if key in ["x", "y"]}
        openfoam_outvar_numpy = {key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]}
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
