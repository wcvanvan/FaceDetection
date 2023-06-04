#pragma once
#include <cmath>
#include "matrix.hpp"
#include "face_binary_cls.hpp"
#include "neon_replace.hpp"

bool fully_connect(const Matrix<float>& matrix_in, const Matrix<float>& result_matrix, fc_param param);

bool fully_connect(const Matrix<float>& matrix_in, const Matrix<float>& result_matrix, fc_param param) {
	auto *result = new float[4 * param.out_features];
	for (int core_index = 0; core_index < param.out_features; ++core_index) {
		__m128 sum_vector = _mm_set1_ps(param.p_bias[core_index]);
		for (int vector_index = 0; vector_index < param.in_features / 4; ++vector_index) {
			__m128 vector_matrix_in = _mm_load_ps(&matrix_in.data_start[vector_index * 4]);
			__m128 vector_weight = _mm_load_ps(&param.p_weight[vector_index * 4 + core_index * param.in_features]);
			__m128 mul = _mm_mul_ps(vector_matrix_in, vector_weight);
			sum_vector = _mm_add_ps(sum_vector, mul);
		}
		_mm_store_ps(&result[core_index * 4], sum_vector);
	}
	for (int i = 0; i < 4 * param.out_features; ++i) {
		result_matrix.data_start[i / 4] += result[i];
	}
    delete[] result;
	// softmax
    float softmax_sum = 0.0f;
    auto * fc_result = new float[param.out_features];
    for (int i = 0; i < param.out_features; ++i) {
        fc_result[i] = exp(result_matrix.data_start[i]);
        softmax_sum += fc_result[i];
    }
    for (int i = 0; i < param.out_features; ++i) {
        result_matrix.data_start[i] = fc_result[i] / softmax_sum;
    }
    delete[] fc_result;
	return true;
}