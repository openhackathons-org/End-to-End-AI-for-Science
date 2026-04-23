#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
###############################################################################

# This example script is used as the entrypoint for our Docker container.
# The following command launches the JuptyerLab server.

echo "starting jupyter in the background"
jupyter lab \
        --ip 0.0.0.0                               `# Bind to all network interfaces` \
        --allow-root                               `# Allow running as the root user` \
        --no-browser                               `# Do not attempt to launch a browser` \
        --NotebookApp.base_url="/lab"              `# Set a base URL for the lab` \
        --NotebookApp.token="$JUPYTER_TOKEN"       `# Optionally require a token for access` \
        --NotebookApp.password=""                  `# Do not require password to access the course` \
        --notebook-dir="/dli/workshop"                 `# Set notebook directory`
