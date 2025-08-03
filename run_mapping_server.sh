#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 -m rayfronts.mapping_server_rosnode \
	dataset=ros2isaacsim \
	mapping=semantic_ray_frontiers_map \
	mapping.vox_size=0.5 \
	dataset.rgb_resolution=[224,224] \
	dataset.depth_resolution=[224,224] \
	dataset.frame_skip=10 \
	mapping.max_rays_per_frame=10000 \
	mapping.vox_accum_period=4 \
	mapping.occ_observ_weight=100 \
	mapping.max_occ_cnt=100 \
	mapping.ray_accum_period=8 \
	mapping.ray_accum_phase=4

