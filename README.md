# Picture Classification

function: capable of distinguishing human and background pictures, and give the possibilities of both


# General structure

This simple CNN model contains 3 convolutional layers, 2 max-pooling layers and 1 fully-connected layer.

The input image should be resize to 3x128x128 (channel, height, width), which will turn into 2 numbers in the end,  between which one indicates the possibility that the picture is a human face, the other indicates the possibility of background.

# Convolutional layer

In convolutional layer, the size of convolutional kernel can be numbers other than 3

I used im2col algorithm from caffe to turn convolution operation into matrix multiplication

after implementing im2col, all I need is to multiply a 1 x K matrix with a K x N matrix

## Matrix multiplication

![mul](https://github.com/wcvanvan/PictureClassification/blob/main/samples/illustration/mul.JPG)

To do the multiplication of 1 x K with K x N

I utilized the idea of SIMD and partitioning the matrix B into smaller blocks

I used the idea of macros to achieve crossed-platform SIMD all using the names in SSE

~~~c++
#pragma once
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include <arm_neon.h>
#define __m128 float32x4_t
#define	_mm_load_ps vld1q_f32
#define _mm_store_ps vst1q_f32
#define _mm_mul_ps vmulq_f32
#define _mm_set1_ps vdupq_n_f32
#define _mm_add_ps vaddq_f32
#else
#include <immintrin.h>
#endif
~~~

In SSE, there are totally 16 registers. Each register holds 4 single precision floating-point numbers.  In matrix B, I put K * 40 numbers into 1 group, and I do the multiplication group by group. If there less than 40 numbers left, numbers in 1 group will be adjusted.

![partition](https://github.com/wcvanvan/PictureClassification/blob/main/samples/illustration/partition.JPG)

As the picture shown below, I calculate multiplication result of 1 group row by row and eventually add them up.

I use element a in Mat A to multiply row a in Mat B, element b to multiply row b and so on, and they will be put into an array. I manage to avoid the number of registers to go above 16.

![mul_order](https://github.com/wcvanvan/PictureClassification/blob/main/samples/illustration/mul_order.JPG)

There could be memory alignment issues in SIMD when N cannot be divided by 4. It could be solved by "filling" the gap by 0, but the input image must be 128 x 128, so it won't cause trouble



## Comparison with openblas

My algorithm is 3 times slower than openblas when doing multiplication of 1 x 10^4 with 10^4 x 10^5. But it's of the same speed when O3 is on



# ReLu

To turn the numbers smaller than 0 to 0. I try to find a continuous segment of numbers that are all smaller than 0, and use fill(start, end, 0) to turn them all into 0

I tested the time spent by set number to be 0 one by one, and fill(). The result shows that fill() is one time faster than the former way when the number of numbers to be modified is 10000



# Max-Pooling Layer

I simply use the idea of comparison to find the max in every 2x2 block, it takes 3 times of comparison to obtain one number in the result of max-pooling. I cannot find better solution



# Fully-Connected Layer

I also used SIMD to accelerate this layer in order to calculate the dot product betweent two vectors of 4 numbers simultaneously



# Test on ARM

The self-made Universal Intrinsics runs properly on ARM. But on ARM it's about 8 seconds slower than on X86

