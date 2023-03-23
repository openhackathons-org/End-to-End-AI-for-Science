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

import numpy as np
import os 
from sympy import Symbol, Eq, Abs, sin, cos

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.geometry.primitives_2d import Rectangle as rect
from modulus.models.fully_connected import FullyConnectedArch
from modulus.key import Key
from modulus.node import Node
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)

def read_wf_data(velocity_scale,pressure_scale):
    path = "/workspace/python/source_code/navier_stokes/data_lat.npy"
    print(path)
    ic = np.load(path).astype(np.float32)
    
    Pa_to_kgm3 = 0.10197
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(-0.720, 0.719, ic[0].shape[0]),
        np.linspace(-0.720, 0.719, ic[0].shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_x.astype(np.float32).flatten(),axis=-1)
    invar["y"] = np.expand_dims(mesh_y.astype(np.float32).flatten(),axis=-1)
    invar["t"] = np.full_like(invar["x"], 0)
    outvar = {}
    outvar["u"] = np.expand_dims((ic[0]/velocity_scale).flatten(),axis=-1)
    outvar["v"] = np.expand_dims((ic[1]/velocity_scale).flatten(),axis=-1)
    outvar["p"] = np.expand_dims((ic[2]*Pa_to_kgm3/pressure_scale).flatten(),axis=-1)
    
    return invar, outvar


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    
    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")
    
    # make geometry for problem
    length = (-0.720, 0.720)
    height  = (-0.720, 0.720)
    box_bounds = {x: length, y: height}

    # define geometry
    rec = rect(
        (length[0], height[0]),
        (length[1], height[1])
    )
    
    # Scaling and Nondimensionalizing the Problem
    
    #############
    # Real Params
    #############
    fluid_kinematic_viscosity = 1.655e-5  # m**2/s 
    fluid_density = 1.1614  # kg/m**3
    fluid_specific_heat = 1005  # J/(kg K)
    fluid_conductivity = 0.0261  # W/(m K)

    ################
    # Non dim params for normalisation 
    ################
    # Diameter of Earth : 12742000 m over range of 1.440
    length_scale = 12742000/1.440 
    # 60 hrs to 1 timestep- every inference frame is a 6 hour prediction (s)
    time_scale = 60*60*60 
    # Calcuale velocity & pressure scale 
    velocity_scale = length_scale / time_scale  # m/s
    pressure_scale = fluid_density * ((length_scale / time_scale) ** 2)  # kg / (m s**2)
    # Density scale
    density_scale = 1.1614  # kg/m3


    ##############################
    # Nondimensionalization Params for NavierStokes fn
    ##############################
    # fluid params
    nd_fluid_kinematic_viscosity = fluid_kinematic_viscosity / (
        length_scale ** 2 / time_scale
    )
    nd_fluid_density = fluid_density / density_scale
    
    
    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}

    # make navier stokes equations
    ns = NavierStokes(nu=nd_fluid_kinematic_viscosity, rho=nd_fluid_density, dim=2, time=True)


    # make network 
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        periodicity={"x": length, "y" : height}, 
        layer_size=256,
    )

    # make nodes to unroll graph on
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_net")]

    # make initial condition domain
    navier = Domain("navier_stokes")
    
    # make initial condition 
    ic_invar,ic_outvar = read_wf_data(velocity_scale,pressure_scale)

    ic = PointwiseConstraint.from_numpy(
            nodes,
            ic_invar,
            ic_outvar,
            batch_size=cfg.batch_size.initial_condition,
        )
    navier.add_constraint(ic, name="ic")
    
    # make interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        bounds=box_bounds,
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    navier.add_constraint(interior, name="interior")

    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
        vtk_obj = VTKUniformGrid(
            bounds=[(-0.720, 0.720), (-0.360, 0.360)],
            npoints=[1440,720],
            export_map={"u": ["u", "v"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y"},
            output_names=["u", "v", "p"],
            requires_grad=False,
            invar={"t": np.full([720 *1440, 1], specific_time)},
            batch_size=100000,
        )
        navier.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))

    slv = Solver(cfg, navier)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
