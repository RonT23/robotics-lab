#!/bin/bash
xhost +local:docker
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/ros_docker/external:/root/catkin_ws/src/external \
  --net=host \
  --privileged \
  ros-docker
