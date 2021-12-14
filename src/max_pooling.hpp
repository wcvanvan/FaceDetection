#pragma once

#include "matrix.hpp"

bool max_pooling(Matrix<float>& matrix_in, Matrix<float>& result_matrix);

/**
 * take the max term of every 4x4 block
 */
bool max_pooling(Matrix<float>& matrix_in, Matrix<float>& result_matrix) {
	size_t matrix_area = matrix_in.colsROI * matrix_in.rowsROI;
	size_t output_w = matrix_in.colsROI / 2, output_h = matrix_in.rowsROI / 2;
	float* p_matrix = result_matrix.data_start;
	for (size_t output_channel = 0; output_channel < matrix_in.channels; output_channel++)
	{
		for (int output_row = 0; output_row < output_h; ++output_row) {
			size_t input_row = output_row * 2;
			for (int output_col = 0; output_col < output_w; ++output_col) {
				size_t input_col = output_col * 2;
				float numbers[3];
				numbers[0] = matrix_in.data_start[output_channel * matrix_area + input_row * matrix_in.colsROI + input_col + 1];
				numbers[1] = matrix_in.data_start[output_channel * matrix_area + (input_row + 1) * matrix_in.colsROI + input_col];
				numbers[2] = matrix_in.data_start[output_channel * matrix_area + (input_row + 1) * matrix_in.colsROI + input_col + 1];
				float max = matrix_in.data_start[output_channel * matrix_area + input_row * matrix_in.colsROI + input_col];
				max = (max > numbers[0]) ? max : numbers[0];
				max = (max > numbers[1]) ? max : numbers[1];
				max = (max > numbers[2]) ? max : numbers[2];
				*p_matrix = max;
				p_matrix++;
			}
		}
	}
	return true;
}

