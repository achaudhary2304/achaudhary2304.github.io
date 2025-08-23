---
title: "ERC 2025"
excerpt: "Work Done in ERC2025"
collection: portfolio
---


## Project Overview

The Robot has to follow the path of the Aruco Markers and traverse the whole marsyard. In the marsyard there would be objects sprinkled randomly which we need to identify and make a report of the whole mission. This statement is simpler to the actual problem statement. You can find more details as [[Link](https://github.com/husarion/erc2025/blob/main/TECHNICAL_HANDBOOK.md)].

## Scope of Work

I am working on mainly the Autonomous part of ERC where we have to detect the aruco markers and create how to traverse the whole thing without any prior map.

## Team Structure

I am leading a group of 7-8 sophomores which recently started working on Robotics and ROS and things like these and have very limited experience of what to do and all that. My job was to help them build this system and guide them and work with them to solve this problem.

## Key Technical Achievements

Some of the things I worked upon:

- **Multi-Camera Integration**: Merging 4 Zed Cameras using Rtabmap and creating a combined pointcloud
- **Computer Vision Optimization**: Optimizing the Aruco Marker Detection
- **Pointcloud Filtering**: Modifying the pointcloud data so as to differntiate between steep hills,traversable hills and rocks 
- **Navigation System**: Optimizing the Mapping and the Navigation part using Nav2 and Slam_toolbox.This is the part where I am still working on right now.
- **Networking**: Beacuse the competition is remote we need to transfer some of the ROS2 topics to our local machine,so I worked on setting up Husarnet which is a peer to peer VPN for connecting Ros2 Nodes over a network. I also found some bugs in their code ;)


## Acknowledgments

Credit to the juniors who worked with me and tested a lot of ideas I had.