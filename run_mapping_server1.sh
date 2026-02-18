#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 -m rayfronts.mapping_server_rosnode \
    --config-name low_memory dataset=ros2isaacsim
