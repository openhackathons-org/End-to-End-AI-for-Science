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

"""Defines the FCN architecture"""
import logging
import torch
from torch import Tensor
from typing import List, Tuple, Dict

from modulus.sym.models.afno.afno import AFNONet
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key


class FourcastNetArch(Arch):
    "Defines the FourcastNet architecture"

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        img_shape: Tuple[int, int],
        detach_keys: List[Key] = [],
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
    ) -> None:
        """Fourcastnet model. This is a simple wrapper for Modulus' AFNO model.
        The only difference is that FourcastNet needs multi-step training. This class
        allows the model to auto-regressively predict multiple timesteps

        Parameters (Same as AFNO)
        ----------
        input_keys : List[Key]
            Input key list. The key dimension size should equal the variables channel dim.
        output_keys : List[Key]
            Output key list. The key dimension size should equal the variables channel dim.
        img_shape : Tuple[int, int]
            Input image dimensions (height, width)
        detach_keys : List[Key], optional
            List of keys to detach gradients, by default []
        patch_size : int, optional
            Size of image patchs, by default 16
        embed_dim : int, optional
            Embedded channel size, by default 256
        depth : int, optional
            Number of AFNO layers, by default 4
        num_blocks : int, optional
            Number of blocks in the frequency weight matrices, by default 4
        """
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
        )

        # get number of timesteps steps to unroll
        assert (
            len(self.input_keys) == 1
        ), "Error, FourcastNet only accepts one input variable (x_t0)"
        self.n_tsteps = len(self.output_keys)
        logging.info(f"Unrolling FourcastNet over {self.n_tsteps} timesteps")

        # get number of input/output channels
        in_channels = self.input_keys[0].size
        out_channels = self.output_keys[0].size

        # intialise AFNO kernel
        self._impl = AFNONet(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=(patch_size, patch_size),
            img_size=img_shape,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # prepare input tensor
        x = self.prepare_input(
            input_variables=in_vars,
            mask=self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )

        # unroll model over multiple timesteps
        ys = []
        for t in range(self.n_tsteps):
            x = self._impl(x)
            ys.append(x)
        y = torch.cat(ys, dim=1)

        # prepare output dict
        return self.prepare_output(
            output_tensor=y,
            output_var=self.output_key_dict,
            dim=1,
            output_scales=self.output_scales,
        )
