#pragma once
#include <immintrin.h>
#include "matrix.hpp"
#include "face_binary_cls.hpp"
bool fully_connect(const Matrix<float>& matrix_in, fc_param param, const Matrix<float>& result_matrix);

bool fully_connect(const Matrix<float>& matrix_in, fc_param param, const Matrix<float>& result_matrix) {
    float result[8];
    fill(result, &result[3], param.p_bias[0]);
    fill(&result[4], &result[7], param.p_bias[1]);
    for (int core_index = 0; core_index < param.out_features; ++core_index) {
        __m128 sum_vector = _mm_set_ps(0.,0.,0.,0.);
        for (int vector_index = 0; vector_index < param.in_features/4; ++vector_index) {
            __m128 vector_matrix_in = _mm_load_ps(&matrix_in.data_start[vector_index*4]);
            __m128 vector_weight = _mm_load_ps(&param.p_weight[vector_index*4 + core_index * param.in_features]);
            __m128 mul = _mm_mul_ps(vector_matrix_in, vector_weight);
            sum_vector = _mm_add_ps(sum_vector, mul);
        }
        _mm_store_ps(&result[core_index*4], sum_vector);
    }
    for (int i = 0; i < 8; ++i) {
        result_matrix.data_start[i/4] += result[i];
    }
}