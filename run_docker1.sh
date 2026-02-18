#!/bin/bash

docker run -it \
	--gpus all \
	--network host \
	--ipc host \
	--privileged \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=1 \
	-e DISPLAY="${DISPLAY:-:0}" \
	-e QT_X11_NO_MITSHM=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v ~/work/RayFronts:/workspace/RayFronts \
	-w /workspace/RayFronts \
	rayfronts:desktop-models
