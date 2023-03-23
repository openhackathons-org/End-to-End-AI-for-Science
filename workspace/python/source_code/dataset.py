# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gdown
import os

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

## FCN for Omniverse-P1
url = 'https://drive.google.com/uc?id=16YqSnstqoSJdgBzerbzYIkYagwS12lK3&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/FCN.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)


## FCN for Omniverse-P2 
url = 'https://drive.google.com/uc?id=1lSSx8eKfqCcHAbDvXTeUMoZGHfVQe-HG&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+ '/FCN/dataset.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)


## Download and Install Omniverse
url = 'https://drive.google.com/uc?id=1DugS2IbHhBPyCE-EuZczLHBZnlnFViIm&export=download'
output = str(os.path.realpath(os.path.dirname(__file__)))+'/ov.zip'
gdown.cached_download(url, output, quiet=False,proxy=None,postprocess=gdown.extractall)
os.remove(output)
