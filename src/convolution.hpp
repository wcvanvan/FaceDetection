#pragma once
#include <opencv2/opencv.hpp>
#include "matrix.hpp"
#include "face_binary_cls.hpp"
#include <cstring>
#include <algorithm>

bool im2col(Matrix<float>& data_im, conv_param& param, float* data_col);

void convolution(Matrix<float>& data_im, conv_param& param, Matrix<float>& result_matrix);

void fully_connect(Matrix<float>& data_im, fc_param& param, Matrix<float>& result_matrix);

void convolution(Matrix<float>& data_im, conv_param& param, Matrix<float>& result_matrix) {
	const size_t output_h = (data_im.rowsROI + 2 * param.pad - param.kernel_size) / param.stride + 1;
	const size_t output_w = (data_im.colsROI + 2 * param.pad - param.kernel_size) / param.stride + 1;
	const size_t output_area = output_h * output_w;
	const int kernel_area = param.kernel_size * param.kernel_size;
	auto* data_col = new float[output_w * output_h * kernel_area * param.in_channels];
	im2col(data_im, param, data_col);
	auto data_col_mat = new Matrix<float>(kernel_area * param.in_channels,
		output_w * output_h, 1, data_col);
	float* data_ptr = result_matrix.data_start;
	for (int out_channel = 0; out_channel < param.out_channels; ++out_channel) {
		float* p_conv_core_weight = &param.p_weight[out_channel * kernel_area * param.in_channels];
		
		auto conv_core_mat = new Matrix<float>(1, kernel_area * param.in_channels, 1, p_conv_core_weight);
		conv_core_mat->isStaticData = true; // don't need to delete the data during destruction
		float* bias = new float[output_area];
		fill(bias, bias + output_area, param.p_bias[out_channel]);
		auto channel_mat = new Matrix<float>(1, output_w * output_h, 1, bias);
		*channel_mat = *channel_mat + *conv_core_mat * *data_col_mat;
		memcpy(data_ptr, channel_mat->data, output_area * sizeof(float));
		data_ptr += output_area;
		delete conv_core_mat;
		delete channel_mat;
	}
	delete data_col_mat;
}

bool im2col(Matrix<float>& data_im, conv_param& param, float* data_col) {
	if (data_im.data_start == nullptr) {
		return false;
	}
	const int output_h = (data_im.rowsROI + 2 * param.pad - param.kernel_size) / param.stride + 1;
	const int output_w = (data_im.colsROI + 2 * param.pad - param.kernel_size) / param.stride + 1;
	for (int channel = 0; channel < param.in_channels; ++channel) {
		for (int kernel_row = 0; kernel_row < param.kernel_size; ++kernel_row) {
			for (int kernel_col = 0; kernel_col < param.kernel_size; ++kernel_col) {
				//cout << kernel_row << " " << kernel_col << endl;
				int input_row = kernel_row - param.pad;
				for (int output_row = 0; output_row < output_h; ++output_row) {
					// 该行超界
					if (input_row < 0 || input_row >= data_im.rowsROI) {
						for (int output_col = 0; output_col < output_w; ++output_col) {
							*data_col = 0;
							data_col++;
						}
					}
					else {
						int input_col = kernel_col - param.pad;
						for (int output_col = 0; output_col < output_w; ++output_col) {
							// 该列超界
							if (input_col < 0 || input_col >= data_im.colsROI) {
								*data_col = 0;
								data_col++;
							}
							else {
								*data_col = data_im.data_start[channel * data_im.rowsROI * data_im.colsROI +
									input_row * data_im.colsROI + input_col];
								data_col++;
							}
							input_col += param.stride;
						}
					}
					input_row += param.stride;
				}
			}
		}
	}
	return true;
}
