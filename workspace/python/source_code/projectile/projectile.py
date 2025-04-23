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
from sympy import Symbol, sin, cos, pi,  Eq
import torch
import physicsnemo
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D,Point1D
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
        PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from projectile_eqn import ProjectileEquation
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    #Creating Nodes and Domain
    pe = ProjectileEquation()
    projectile_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("x"),Key("y")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = pe.make_nodes() + [projectile_net.make_node(name="projectile_network")]

    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    #Creating Geometry and adding constraint
    geo = Point1D(0)
    
    #make domain
    projectile_domain = Domain()

    #add constraint to solver
    v_o = 40.0
    theta = np.pi/3
    time_range = {t :(0.0,5.0)}
    

    #initial condition
    # Set boundary to be only left boundary
    IC = PointwiseBoundaryConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"x": 0.0,"y":0.0, "x__t":v_o*cos(theta), "y__t":v_o*sin(theta)},
            batch_size = cfg.batch_size.initial_x,
            parameterization = {t:0.0}
    )
    projectile_domain.add_constraint(IC,"IC")

    #interior
    interior = PointwiseBoundaryConstraint(
            nodes = nodes,
            geometry = geo,
            outvar = {"ode_x":0.0,"ode_y":-9.81},
            batch_size = cfg.batch_size.interior,
            parameterization = time_range,
    )   
    projectile_domain.add_constraint(interior,"interior")

    
    # Setup validator
    delta_T = 0.01
    t_val = np.arange(0.,5.,delta_T)
    T_val = np.expand_dims(t_val.flatten(), axis = -1)
    X_val =  v_o*np.cos(theta)*T_val
    Y_val =  v_o*np.sin(theta)*T_val - 0.5*9.81*(T_val**2)
    
    invar_numpy = {"t": T_val}
    outvar_numpy = {"x":X_val, "y": Y_val}
    
    validator = PointwiseValidator(
            nodes=nodes,
            invar=invar_numpy,
            true_outvar=outvar_numpy,
            batch_size=128,
            plotter = ValidatorPlotter(),
    )
    projectile_domain.add_validator(validator)
    
    
    # Setup Inferencer
    t_infe = np.arange(0,8,0.001)
    T_infe = np.expand_dims(t_infe.flatten(), axis = -1)
    invar_infe = {"t":T_infe}

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_infe,
        output_names=["x","y"],
        batch_size=128,
        plotter=InferencerPlotter(),
    )
    projectile_domain.add_inferencer(grid_inference, "inferencer_data")

    #make solver
    slv = Solver(cfg, projectile_domain)

    #start solve
    slv.solve()
    
if __name__ == "__main__":
    run()

































        
