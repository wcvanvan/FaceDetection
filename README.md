# Picture Classification

The forward part of a CNN, using the model trained by library "libfacedetect" (written by my teacher)

capable of distinguishing human and scenery pictures and giving the reliability (possibility) of the test

This simple CNN model contains 3 convolutional layers, 2 max-pooling layers and 1 fully-connected layer.

## Features
The core of this project is a self-made matrix class, matrix multiplication and high speed convolution

+ matrix class possesses soft-copy function, performant in the scene of image copying

+ matrix multiplication imitates algorithm from GotoBLAS, including matrix partitioning, simd (cross platfrom achieved)

+ im2col algorithm from caffe is implemented to turn convolution operation into matrix multiplication. After im2col, we only need to do multiplying a 1 x K matrix with a K x N matrix, which I specifically do optimization. Finally this kind of multiplication reaches the 
same speed with OpenBLAS

## How to use
Check out main.cpp, and change the image path

The program will proffer two numbers after running, one indicates the possibility that the picture is a human face, the other indicates the possibility of scenery.





