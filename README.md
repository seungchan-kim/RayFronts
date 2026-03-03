## Introduction
This is a forked code repository of RayFronts, which serves as a perception and representation backbone of RAVEN. For more comprehensive information about the original RayFronts codes, please go to <a href="https://github.com/RayFronts/RayFronts">this repository</a>. This README contains minimal information about setting RayFronts environment for RAVEN. 

## Docker Setup

We strongly recommend setting up a docker image of RayFronts for RAVEN. 

Build the image with:

    docker build -f docker/desktop.Dockerfile . -t rayfronts:desktop

After building the `rayfronts:desktop` docker image, run

    ./run_docker.sh

Then, build CPP extension one time by running `./compile.sh`