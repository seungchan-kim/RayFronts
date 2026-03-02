<h1 align="center">RayFronts: Open-Set Semantic Ray Frontiers <br/>
  for Online Scene Understanding and Exploration</h1>

<p align="center">
  <a href="https://oasisartisan.github.io/"><strong>Omar Alama</strong></a>
  .
  <a href="https://avigyanbh.github.io/"><strong>Avigyan Bhattacharya</strong></a>
  ·
  <a href="https://purenothingness24.github.io/"><strong>Haoyang He</strong></a>
  ·
  <a href="https://seungchan-kim.github.io/"><strong>Seungchan Kim</strong></a>
  <br>
  <a href="https://haleqiu.github.io/"><strong>Yuheng Qiu</strong></a>
  .
  <a href="https://theairlab.org/team/wenshan/"><strong>Wenshan Wang</strong></a>
  ·
  <a href="https://cherieho.com/"><strong>Cherie Ho</strong></a>
  ·
  <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
  ·
  <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
</p>

  <h3 align="center"><a href="https://arxiv.org/abs/2504.06994">Paper</a> | <a href="https://RayFronts.github.io/">Project Page</a> | <a href="https://www.youtube.com/watch?v=fFSKUBHx5gA">Video</a></h3>
  <div align="center"></div>


## Introduction
This is a forked code repository of RayFronts, which serves as a perception and representation backbone of RAVEN. For more comprehensive information about the original RayFronts codes, please go to <a href="https://github.com/RayFronts/RayFronts">this repository</a>. This README contains minimal information about setting RayFronts environment for RAVEN. 

## Docker Setup

We strongly recommend setting up a docker image of RayFronts for RAVEN. 

Build the image with:

    docker build -f docker/desktop.Dockerfile . -t rayfronts:desktop

After building the `rayfronts:desktop` docker image, run

    ./run_docker.sh

Then, build CPP extension one time by running `./compile.sh`