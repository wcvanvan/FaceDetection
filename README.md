# Picture Classification

The forward part of a CNN, using the model trained by library "libfacedetect" (from my teacher)
capable of distinguishing human and scenery pictures and giving the reliability (possibility) of the test

This simple CNN model contains 3 convolutional layers, 2 max-pooling layers and 1 fully-connected layer.

## Features
The core of this project is a self-made matrix class and matrix multiplication algorithm

matrix class possesses soft-copy function, performant in the scene of image copying

matrix multiplication imitates algorithm from GotoBLAS

im2col algorithm from caffe is implemented to turn convolution operation into matrix multiplication

## How to use
The input image should be resize to 3x128x128 (channel, height, width), which will be turned into 2 numbers in the end, one indicates the possibility that the picture is a human face, the other indicates the possibility of scenery.





