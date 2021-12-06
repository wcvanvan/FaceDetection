#pragma once

#include "matrix.hpp"

bool max_pooling(Matrix<float> &matrix_in, Matrix<float> &result_matrix);

/**
 * take the max term of every 4x4 block
 */
bool max_pooling(Matrix<float> &matrix_in, Matrix<float> &result_matrix) {
    size_t output_w = matrix_in.rowsROI / 2, output_h = matrix_in.colsROI / 2;
    float *matrix_in_ptr = result_matrix.data;
    for (int output_row = 0; output_row < output_h; ++output_row) {
        size_t input_row = output_row * 2;
        for (int output_col = 0; output_col < output_w; ++output_col) {
            size_t input_col = output_col * 2;
            float max = matrix_in.data_start[input_row * matrix_in.colsROI + input_col];
            max = (max > matrix_in.data_start[input_row * matrix_in.colsROI + input_col + 1]) ? max
                                                                                              : matrix_in.data_start[
                          input_row * matrix_in.colsROI +
                          input_col + 1];
            max = (max > matrix_in.data_start[(input_row + 1) * matrix_in.colsROI + input_col]) ? max
                                                                                                : matrix_in.data_start[
                          (input_row + 1) * matrix_in.colsROI +
                          input_col];
            max = (max > matrix_in.data_start[(input_row + 1) * matrix_in.colsROI + input_col + 1]) ? max
                                                                                                    : matrix_in.data_start[
                          (input_row + 1) * matrix_in.colsROI + input_col + 1];
            *matrix_in_ptr = max;
            matrix_in_ptr++;
            output_col += 2;
        }
        input_row += 2;
    }
    return true;
}

