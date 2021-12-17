#pragma once
#include <cstring>
#include <iostream>
#include "neon_replace.hpp"

#define A(i, j) A[(i)*K + (j)]
#define B(i, j) B[(i)*N + (j)]
#define C(i, j) C[(i)*N + (j)]

using namespace std;

void mul_1xk(const float* A, const float* B, float* C, int M, int N, int K);

void addDot_4x4(const float* A, float* B, float* C, int M, int N, int K);

void addDot_8x8(const float* A, float* B, float* C, int M, int N, int K);

void mul_1xk(const float* A, const float* B, float* C, int M, int N, int K) {
	int vectors_in_array = 10;
	int original_vectors_in_array = vectors_in_array;
	for (int vector_array_count = 0;
		vector_array_count < (int)std::ceil((float)N / ((float)original_vectors_in_array * 4)); ++vector_array_count) {
		if (vector_array_count == N / (vectors_in_array * 4)) {
			vectors_in_array = (N % (vectors_in_array * 4)) / 4;
		}
		__m128* vector_array = new __m128[vectors_in_array];
		memset(vector_array, 0, sizeof(__m128) * vectors_in_array);
		for (int core_col = 0; core_col < K; ++core_col) {
			__m128 core_value_scalar = _mm_set1_ps(A[core_col]);
			for (int vector_count = 0; vector_count < vectors_in_array; ++vector_count) {
				__m128 vector = _mm_load_ps(&B[core_col * N + vector_count * 4 +
					vector_array_count * 4 * original_vectors_in_array]);
				__m128 mul_result = _mm_mul_ps(core_value_scalar, vector);
				vector_array[vector_count] = _mm_add_ps(vector_array[vector_count], mul_result);
			}
		}
		for (int vector_in_array = 0; vector_in_array < vectors_in_array; ++vector_in_array) {
			_mm_store_ps(&C[vector_in_array * 4 + vector_array_count * 4 * original_vectors_in_array],
				vector_array[vector_in_array]);
		}
	}
}

void addDot_4x4(const float* A, float* B, float* C, int M, int N, int K) {
	float c_00_reg, c_01_reg, c_02_reg, c_03_reg, c_10_reg, c_11_reg,
		c_12_reg, c_13_reg, c_20_reg, c_21_reg, c_22_reg, c_23_reg, c_30_reg,
		c_31_reg, c_32_reg, c_33_reg, a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
		b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

	float* b_p0_ptr = &B(0, 0), * b_p1_ptr = &B(0, 1), * b_p2_ptr = &B(0, 2),
		* b_p3_ptr = &B(0, 3);

	c_00_reg = 0.0;
	c_01_reg = 0.0;
	c_02_reg = 0.0;
	c_03_reg = 0.0;
	c_10_reg = 0.0;
	c_11_reg = 0.0;
	c_12_reg = 0.0;
	c_13_reg = 0.0;
	c_20_reg = 0.0;
	c_21_reg = 0.0;
	c_22_reg = 0.0;
	c_23_reg = 0.0;
	c_30_reg = 0.0;
	c_31_reg = 0.0;
	c_32_reg = 0.0;
	c_33_reg = 0.0;

	for (int p = 0; p < K; p++) {
		a_0p_reg = A(0, p);
		a_1p_reg = A(1, p);
		a_2p_reg = A(2, p);
		a_3p_reg = A(3, p);

		b_p0_reg = *b_p0_ptr;
		b_p0_ptr += N;
		b_p1_reg = *b_p1_ptr;
		b_p1_ptr += N;
		b_p2_reg = *b_p2_ptr;
		b_p2_ptr += N;
		b_p3_reg = *b_p3_ptr;
		b_p3_ptr += N;

		/* First row and second rows */
		c_00_reg += a_0p_reg * b_p0_reg;
		c_10_reg += a_1p_reg * b_p0_reg;

		c_01_reg += a_0p_reg * b_p1_reg;
		c_11_reg += a_1p_reg * b_p1_reg;

		c_02_reg += a_0p_reg * b_p2_reg;
		c_12_reg += a_1p_reg * b_p2_reg;

		c_03_reg += a_0p_reg * b_p3_reg;
		c_13_reg += a_1p_reg * b_p3_reg;

		/* Third and fourth rows */
		c_20_reg += a_2p_reg * b_p0_reg;
		c_30_reg += a_3p_reg * b_p0_reg;

		c_21_reg += a_2p_reg * b_p1_reg;
		c_31_reg += a_3p_reg * b_p1_reg;

		c_22_reg += a_2p_reg * b_p2_reg;
		c_32_reg += a_3p_reg * b_p2_reg;

		c_23_reg += a_2p_reg * b_p3_reg;
		c_33_reg += a_3p_reg * b_p3_reg;
	}

	C(0, 0) += c_00_reg;
	C(0, 1) += c_01_reg;
	C(0, 2) += c_02_reg;
	C(0, 3) += c_03_reg;
	C(1, 0) += c_10_reg;
	C(1, 1) += c_11_reg;
	C(1, 2) += c_12_reg;
	C(1, 3) += c_13_reg;
	C(2, 0) += c_20_reg;
	C(2, 1) += c_21_reg;
	C(2, 2) += c_22_reg;
	C(2, 3) += c_23_reg;
	C(3, 0) += c_30_reg;
	C(3, 1) += c_31_reg;
	C(3, 2) += c_32_reg;
	C(3, 3) += c_33_reg;
}


void addDot_8x8(const float* A, float* B, float* C, int M, int N, int K) {
	float c_00_reg, c_01_reg, c_02_reg, c_03_reg, c_04_reg, c_05_reg,
		c_06_reg, c_07_reg, c_10_reg, c_11_reg, c_12_reg, c_13_reg, c_14_reg,
		c_15_reg, c_16_reg, c_17_reg, c_20_reg, c_21_reg, c_22_reg, c_23_reg,
		c_24_reg, c_25_reg, c_26_reg, c_27_reg, c_30_reg, c_31_reg, c_32_reg,
		c_33_reg, c_34_reg, c_35_reg, c_36_reg, c_37_reg, c_40_reg, c_41_reg,
		c_42_reg, c_43_reg, c_44_reg, c_45_reg, c_46_reg, c_47_reg, c_50_reg,
		c_51_reg, c_52_reg, c_53_reg, c_54_reg, c_55_reg, c_56_reg, c_57_reg,
		c_60_reg, c_61_reg, c_62_reg, c_63_reg, c_64_reg, c_65_reg, c_66_reg,
		c_67_reg, c_70_reg, c_71_reg, c_72_reg, c_73_reg, c_74_reg, c_75_reg,
		c_76_reg, c_77_reg, a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg,
		a_5p_reg, a_6p_reg, a_7p_reg, b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg,
		b_p4_reg, b_p5_reg, b_p6_reg, b_p7_reg;

	float* b_p0_ptr = &B(0, 0), * b_p1_ptr = &B(0, 1), * b_p2_ptr = &B(0, 2),
		* b_p3_ptr = &B(0, 3), * b_p4_ptr = &B(0, 4), * b_p5_ptr = &B(0, 5),
		* b_p6_ptr = &B(0, 6), * b_p7_ptr = &B(0, 7);

	c_00_reg = c_01_reg = c_02_reg = c_03_reg = c_04_reg = c_05_reg = c_06_reg =
		c_07_reg = c_10_reg = c_11_reg = c_12_reg = c_13_reg = c_14_reg =
		c_15_reg = c_16_reg = c_17_reg = c_20_reg = c_21_reg = c_22_reg =
		c_23_reg = c_24_reg = c_25_reg = c_26_reg = c_27_reg = c_30_reg =
		c_31_reg = c_32_reg = c_33_reg = c_34_reg = c_35_reg =
		c_36_reg = c_37_reg = c_40_reg = c_41_reg = c_42_reg =
		c_43_reg = c_44_reg = c_45_reg = c_46_reg = c_47_reg =
		c_50_reg = c_51_reg = c_52_reg = c_53_reg =
		c_54_reg = c_55_reg = c_56_reg = c_57_reg =
		c_60_reg = c_61_reg = c_62_reg =
		c_63_reg = c_64_reg = c_65_reg =
		c_66_reg = c_67_reg = c_70_reg =
		c_71_reg = c_72_reg =
		c_73_reg = c_74_reg =
		c_75_reg = c_76_reg =
		c_77_reg = 0.0;

	for (int p = 0; p < K; p++) {
		a_0p_reg = A(0, p);
		a_1p_reg = A(1, p);
		a_2p_reg = A(2, p);
		a_3p_reg = A(3, p);
		a_4p_reg = A(4, p);
		a_5p_reg = A(5, p);
		a_6p_reg = A(6, p);
		a_7p_reg = A(7, p);

		b_p0_reg = *b_p0_ptr;
		b_p0_ptr += N;
		b_p1_reg = *b_p1_ptr;
		b_p1_ptr += N;
		b_p2_reg = *b_p2_ptr;
		b_p2_ptr += N;
		b_p3_reg = *b_p3_ptr;
		b_p3_ptr += N;
		b_p4_reg = *b_p4_ptr;
		b_p4_ptr += N;
		b_p5_reg = *b_p5_ptr;
		b_p5_ptr += N;
		b_p6_reg = *b_p6_ptr;
		b_p6_ptr += N;
		b_p7_reg = *b_p7_ptr;
		b_p7_ptr += N;

		/* First row and second rows */
		c_00_reg += a_0p_reg * b_p0_reg;
		c_10_reg += a_1p_reg * b_p0_reg;

		c_01_reg += a_0p_reg * b_p1_reg;
		c_11_reg += a_1p_reg * b_p1_reg;

		c_02_reg += a_0p_reg * b_p2_reg;
		c_12_reg += a_1p_reg * b_p2_reg;

		c_03_reg += a_0p_reg * b_p3_reg;
		c_13_reg += a_1p_reg * b_p3_reg;

		c_04_reg += a_0p_reg * b_p4_reg;
		c_14_reg += a_1p_reg * b_p4_reg;

		c_05_reg += a_0p_reg * b_p5_reg;
		c_15_reg += a_1p_reg * b_p5_reg;

		c_06_reg += a_0p_reg * b_p6_reg;
		c_16_reg += a_1p_reg * b_p6_reg;

		c_07_reg += a_0p_reg * b_p7_reg;
		c_17_reg += a_1p_reg * b_p7_reg;

		/* Third and fourth rows */
		c_20_reg += a_2p_reg * b_p0_reg;
		c_30_reg += a_3p_reg * b_p0_reg;

		c_21_reg += a_2p_reg * b_p1_reg;
		c_31_reg += a_3p_reg * b_p1_reg;

		c_22_reg += a_2p_reg * b_p2_reg;
		c_32_reg += a_3p_reg * b_p2_reg;

		c_23_reg += a_2p_reg * b_p3_reg;
		c_33_reg += a_3p_reg * b_p3_reg;

		c_24_reg += a_2p_reg * b_p4_reg;
		c_34_reg += a_3p_reg * b_p4_reg;

		c_25_reg += a_2p_reg * b_p5_reg;
		c_35_reg += a_3p_reg * b_p5_reg;

		c_26_reg += a_2p_reg * b_p6_reg;
		c_36_reg += a_3p_reg * b_p6_reg;

		c_27_reg += a_2p_reg * b_p7_reg;
		c_37_reg += a_3p_reg * b_p7_reg;

		/* Fifth and sixth rows */
		c_40_reg += a_4p_reg * b_p0_reg;
		c_50_reg += a_5p_reg * b_p0_reg;

		c_41_reg += a_4p_reg * b_p1_reg;
		c_51_reg += a_5p_reg * b_p1_reg;

		c_42_reg += a_4p_reg * b_p2_reg;
		c_52_reg += a_5p_reg * b_p2_reg;

		c_43_reg += a_4p_reg * b_p3_reg;
		c_53_reg += a_5p_reg * b_p3_reg;

		c_44_reg += a_4p_reg * b_p4_reg;
		c_54_reg += a_5p_reg * b_p4_reg;

		c_45_reg += a_4p_reg * b_p5_reg;
		c_55_reg += a_5p_reg * b_p5_reg;

		c_46_reg += a_4p_reg * b_p6_reg;
		c_56_reg += a_5p_reg * b_p6_reg;

		c_47_reg += a_4p_reg * b_p7_reg;
		c_57_reg += a_5p_reg * b_p7_reg;

		/* Seventh and eighth rows */
		c_60_reg += a_6p_reg * b_p0_reg;
		c_70_reg += a_7p_reg * b_p0_reg;

		c_61_reg += a_6p_reg * b_p1_reg;
		c_71_reg += a_7p_reg * b_p1_reg;

		c_62_reg += a_6p_reg * b_p2_reg;
		c_72_reg += a_7p_reg * b_p2_reg;

		c_63_reg += a_6p_reg * b_p3_reg;
		c_73_reg += a_7p_reg * b_p3_reg;

		c_64_reg += a_6p_reg * b_p4_reg;
		c_74_reg += a_7p_reg * b_p4_reg;

		c_65_reg += a_6p_reg * b_p5_reg;
		c_75_reg += a_7p_reg * b_p5_reg;

		c_66_reg += a_6p_reg * b_p6_reg;
		c_76_reg += a_7p_reg * b_p6_reg;

		c_67_reg += a_6p_reg * b_p7_reg;
		c_77_reg += a_7p_reg * b_p7_reg;
	}

	C(0, 0) += c_00_reg;
	C(0, 1) += c_01_reg;
	C(0, 2) += c_02_reg;
	C(0, 3) += c_03_reg;
	C(0, 4) += c_04_reg;
	C(0, 5) += c_05_reg;
	C(0, 6) += c_06_reg;
	C(0, 7) += c_07_reg;

	C(1, 0) += c_10_reg;
	C(1, 1) += c_11_reg;
	C(1, 2) += c_12_reg;
	C(1, 3) += c_13_reg;
	C(1, 4) += c_14_reg;
	C(1, 5) += c_15_reg;
	C(1, 6) += c_16_reg;
	C(1, 7) += c_17_reg;

	C(2, 0) += c_20_reg;
	C(2, 1) += c_21_reg;
	C(2, 2) += c_22_reg;
	C(2, 3) += c_23_reg;
	C(2, 4) += c_24_reg;
	C(2, 5) += c_25_reg;
	C(2, 6) += c_26_reg;
	C(2, 7) += c_27_reg;

	C(3, 0) += c_30_reg;
	C(3, 1) += c_31_reg;
	C(3, 2) += c_32_reg;
	C(3, 3) += c_33_reg;
	C(3, 4) += c_34_reg;
	C(3, 5) += c_35_reg;
	C(3, 6) += c_36_reg;
	C(3, 7) += c_37_reg;

	C(4, 0) += c_40_reg;
	C(4, 1) += c_41_reg;
	C(4, 2) += c_42_reg;
	C(4, 3) += c_43_reg;
	C(4, 4) += c_44_reg;
	C(4, 5) += c_45_reg;
	C(4, 6) += c_46_reg;
	C(4, 7) += c_47_reg;

	C(5, 0) += c_50_reg;
	C(5, 1) += c_51_reg;
	C(5, 2) += c_52_reg;
	C(5, 3) += c_53_reg;
	C(5, 4) += c_54_reg;
	C(5, 5) += c_55_reg;
	C(5, 6) += c_56_reg;
	C(5, 7) += c_57_reg;

	C(6, 0) += c_60_reg;
	C(6, 1) += c_61_reg;
	C(6, 2) += c_62_reg;
	C(6, 3) += c_63_reg;
	C(6, 4) += c_64_reg;
	C(6, 5) += c_65_reg;
	C(6, 6) += c_66_reg;
	C(6, 7) += c_67_reg;

	C(7, 0) += c_70_reg;
	C(7, 1) += c_71_reg;
	C(7, 2) += c_72_reg;
	C(7, 3) += c_73_reg;
	C(7, 4) += c_74_reg;
	C(7, 5) += c_75_reg;
	C(7, 6) += c_76_reg;
	C(7, 7) += c_77_reg;
}