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

import gdown
import os
import subprocess

## FCN Dataset 
url = 'https://drive.google.com/uc?id=1mSN6eLqPYEo9d9pBjSGzQ-ocLd8itP0P&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/fourcastnet/dataset.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)

## FCN Pre-trained 
url = 'https://drive.google.com/uc?id=1oSkK69LGP3DfU2tlH5iaejOh94VNsMDu&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/../jupyter_notebook/FourCastNet/pre_trained.zip' 
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)

## NS Data
url = 'https://drive.google.com/uc?id=1IXEGbM3NOO6Dig1sxG1stHubwb09-D2N&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/navier_stokes/dataset.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)

# Darcy Flow Dataset
id = "1aRCdIMgAwUrgGuUCifnoKgwT5-c9YlAT"
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/../jupyter_notebook/Operators/datasets/'
gdown.download(id=id,output=output,quiet=False,proxy=None)

subprocess.run(['unzip',output+"darcy_processed.zip",'-d',output])
## Downloading Modulus Core Dataset 
url = 'https://drive.google.com/uc?id=12B_B8QIHI25VwNfNDB-L8ffsPKtC40dL&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/core/dataset.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)


## Downloading DoMINO Dataset 
url = 'https://drive.google.com/file/d/1dB_-ggzFLPJYAXLUccaGGT_nvk03rYgb/view?usp=sharing'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/domino/dataset.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)

