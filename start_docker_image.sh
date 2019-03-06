#!/bin/bash

# DEFAULT VALUES
GPUIDS="0"
NAME="paperspace_GPU"


NV_GPU="$GPUIDS" nvidia-docker run -it --rm \
        -p 5584:5584 \
        --name "$NAME""$GPUIDS" \
        -v /home/andyser/code/subt_related/paper_code:/root/ \
        andyser/frustum-pointnets bash
