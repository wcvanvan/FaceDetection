#pragma once
#include "convolution.hpp"
#include "face_binary_cls.hpp"
#include "matrix.hpp"
#include <iostream>

using namespace std;

int main() {
	float *array = new float[15];
	for (size_t i = 0; i < 15; i++)
	{
		array[i] = i;
	}
	Matrix<float> data_im(3, 5, 1, array);
	float* answer = new float[18];
	im2col(data_im, conv_params[3], answer);
	for (size_t i = 0; i < 18; i++)
	{
		cout << answer[i] << " ";
	}
}