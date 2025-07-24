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
from sympy import Symbol, sin

import physicsnemo.sym
from physicsnemo.sym.hydra import to_yaml, to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.hydra.utils import compose
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io import ValidatorPlotter

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

class WaveEquation1D(PDE):
    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        # Define coordinates
        x = Symbol("x")
        t = Symbol("t")

        # Define input variables
        input_variables = {"x": x, "t": t}

        # Define the wave function
        u = Function("u")(*input_variables)

        # Set wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # Define the wave equation
        self.equations = {}
        self.equations["wave_equation"] = # wave equation

@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Initialize wave equation with wave speed
    c = # c value
    we = WaveEquation1D(c=c)
    
    # Create neural network
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]
    
    # Define geometry and time range
    x, t_symbol = Symbol("x"), Symbol("t")
    L = float(np.pi)
    geo = Line1D(0, L)
    time_range = {t_symbol: (0, 2 * L)}
    
    # Create domain
    domain = Domain()
    
    # Add initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={}, #initial condition
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")
    
    # Add boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={}, #boundary condition
        lambda_weighting={"u": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Add interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={}, #PDE constraint
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")
    
    # Add validation data
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = # add formula of exact solution
    invar_numpy = {"x": X, "t": T}
    outvar_numpy = {"u": u}
    
    # Create validator
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=128,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator)
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
