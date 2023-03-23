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

import torch
from typing import Dict

Tensor = torch.Tensor


class LpLoss(torch.nn.Module):
    def __init__(
        self,
        d: float = 2.0,
        p: float = 2.0,
    ):
        """Relative Lp loss normalized seperately in the batch dimension.
        Expects inputs of the shape [B, C, ...]

        Parameters
        ----------
        p : float, optional
            Norm power, by default 2.0
        """
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert p > 0.0
        self.p = p

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> float:
        num_examples = x.size()[0]

        xv = x.reshape(num_examples, -1)
        yv = y.reshape(num_examples, -1)

        diff_norms = torch.linalg.norm(xv - yv, ord=self.p, dim=1)
        y_norms = torch.linalg.norm(yv, ord=self.p, dim=1)

        return torch.mean(diff_norms / y_norms)

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, float]:
        losses = {}
        for key, value in pred_outvar.items():
            losses[key] = self._rel(pred_outvar[key], true_outvar[key])
        return losses
