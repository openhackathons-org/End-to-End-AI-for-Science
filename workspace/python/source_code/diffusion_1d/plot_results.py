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
import matplotlib.pyplot as plt

network_dir = "./outputs/diffusion_bar/validators/"
data_1 = np.load(network_dir + "Val1.npz", allow_pickle=True)
data_2 = np.load(network_dir + "Val2.npz", allow_pickle=True)
data_1 = np.atleast_1d(data_1.f.arr_0)[0]
data_2 = np.atleast_1d(data_2.f.arr_0)[0]

plt.plot(data_1["x"][:, 0], data_1["pred_u_1"][:, 0], "--", label="u_1_pred")
plt.plot(data_2["x"][:, 0], data_2["pred_u_2"][:, 0], "--", label="u_2_pred")
plt.plot(data_1["x"][:, 0], data_1["true_u_1"][:, 0], label="u_1_true")
plt.plot(data_2["x"][:, 0], data_2["true_u_2"][:, 0], label="u_2_true")

plt.legend()
plt.savefig("image_diffusion_problem_bootcamp")
