---
title: "Arrow Detection"
excerpt: "Arrow Detection using Zed 2i"
collection: portfolio
---
## Project Overview

This was the result to one of the tasks of IRC (International Rover Challenge) where there would be arrows randomly placed in the environment and the rover has to go near the marker and according to the arrow's direction move accordingly.

## Challenges Faced

- **Variable Lighting Conditions**: The lighting in which the arrow was not fixed and the arrow could be very far and the background was also not fixed (so approaches like feature matching, and other classical computer vision techniques were ruled out)

- **Lack of Training Data**: There was not a single dataset which was large enough to train the model

- **3D Pose Estimation**: Even if we have the model's bounding box how do we get its 3D pose in the world

## Solution Approach

### How Did I Solve Them

1. **Custom Dataset Creation**: We first created a huge dataset ourself where we took photos of arrows in different conditions

2. **Leveraging Zed 2i Capabilities**: We were using the Zed 2i camera for visual odometry and noticed that it had a mode for human detection and things like that. We changed some code in that so that it would use our arrow detection model. This resulted in detecting arrows and also giving an acceptable 3D pose of the arrow.

## Areas for Improvement

### What I Could Have Done Better

- I used Roboflow from augmentations to training the model due to the limited time I had at the time
- I also did not optimize the model to its limit and just trained it for 25 epochs




