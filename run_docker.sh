#!/bin/bash

docker run -it --rm \
	--name rayfronts_container \
	--gpus all \
	--network host \
	--ipc host \
	--privileged \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=1 \
	-v ~/RAVEN/RayFronts:/workspace/RayFronts \
	-w /workspace/RayFronts \
	rayfronts:desktop
