---
title: "MCB State Detector Using OpenCV"
excerpt: "A Python script that uses OpenCV to detect whether a miniature circuit breaker is on or off."
collection: portfolio
---

## Design Choices

The script assumes that the miniature circuit breaker (MCB) has a white body and a colored switch. First, I mask out white and gray regions to isolate the colored parts of the image. I then find contours and select the largest one to filter out smaller objects such as screws.

Using the mean color of that region, I define a color range for the switch. I create a bounding box around the MCB and check whether most of the switch color lies in its upper or lower half to determine whether it is on or off.

I use the HSV color space for greater robustness to shadows and changing lighting conditions.

[View the code and demo images](https://github.com/achaudhary2304/mcb_status).
