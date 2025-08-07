---
title: "MCB State Detector using OpenCV "
excerpt: "A simple python script which just uses openCV to tell whether a MCB(Miniature Circuit Breaker) is on or off"
collection: portfolio
---
  

Design Choices

The script assumes that the MCB would have a white body and can have any random color as the MCB,so first I create a mask to filter out all the white and the grey regions of the image,so that I can have the non white region of the image,Then I take contours of the masked image,as there can be other non white things in the image such as screws and all that I take the largest contour so as to filter out the MCB only.Then I use the mean color of that area to create the range for the color to get the color of the mcb and it's loose range .Then I create a bounding box type thing for the MCB itself and see where does most of the color lie in the upper part or the lower part and tell whether it is on or off.

I use HSV colorspace as it is more robust to shadows and different lighting conditions

Code and some demo images at [Link](https://github.com/achaudhary2304/mcb_status)
