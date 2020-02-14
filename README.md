# Facial Landmark Detection

## About

A simple python application for detection and calculating numerical values for facial landmark using opencv.

## Requirements

FL_Detect requires the following libraries to function:
- opencv-python-headless
- imutils
- numpy
- scipy

## Getting Started

Install the any required packages, example installation shown below:
````
sudo pip install opencv-python-headless
````
Start the script using the following:
````
Python main.py
````

## Program Workflow

- An initial baseline image needs to be taken for normalization, center your face and press "o" when ready
- After a 2 second delay faicial landmarks will be streamed into the "data" datastructure
- Press "q" to end landmark streaming and close application