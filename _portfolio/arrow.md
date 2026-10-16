---
title: "Arrow Detection"
excerpt: "Arrow detection and 3D pose estimation using the ZED 2i camera."
collection: portfolio
---

## Project Overview

For a task at the International Rover Challenge (IRC), the rover had to approach arrow markers placed around the environment and navigate in the indicated direction.

## Challenges Faced

- **Variable Lighting Conditions**: Arrows could be far away, with changing lighting and backgrounds, making classical computer vision techniques such as feature matching unsuitable.
- **Lack of Training Data**: No available dataset was large enough to train the model.
- **3D Pose Estimation**: The model’s bounding boxes needed to be converted into arrow poses in world coordinates.

## Solution Approach

1. **Custom Dataset Creation**: We created our own dataset by photographing arrows under different conditions.
2. **Using the ZED 2i Camera**: We were already using the camera for visual odometry. I adapted its object detection integration to use our arrow detection model, obtaining detections and usable 3D poses for the arrows.

## Areas for Improvement

- I used Roboflow for augmentation and model training because of the limited time available.
- I trained the model for 25 epochs; further training and optimization could improve performance.
