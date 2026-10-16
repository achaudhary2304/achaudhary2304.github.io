---
title: "ERC 2025"
excerpt: "Autonomous navigation and software work for the European Rover Challenge 2025."
collection: portfolio
---

## Project Overview

The rover had to follow a path of ArUco markers and traverse the Mars yard, identifying objects along the way and producing a mission report. The [technical handbook](https://github.com/husarion/erc2025/blob/main/TECHNICAL_HANDBOOK.md) describes the full challenge.

## Scope of Work

I worked mainly on autonomous navigation: detecting ArUco markers and planning how to traverse the environment without a prior map.

## Team Structure

I led a group of 7–8 sophomores who were getting started with robotics and ROS. I helped them build the system, test ideas, and solve problems together.

## Key Technical Achievements

- **Multi-Camera Integration**: Merged data from four ZED cameras using RTAB-Map to create a combined point cloud.
- **Computer Vision Optimization**: Optimized ArUco marker detection.
- **Point Cloud Filtering**: Processed point cloud data to distinguish steep hills, traversable hills, and rocks.
- **Navigation System**: Optimized mapping and navigation using Nav2 and slam_toolbox.
- **Networking**: Set up Husarnet, a peer-to-peer VPN, to transfer ROS 2 topics between the remote rover and our local machines. I also found bugs in the networking code.

## Acknowledgments

Credit to the juniors who worked with me and tested many of my ideas.
