#pragma once
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#include <arm_neon.h>
#define __m128 float32x4_t
#define	_mm_load_ps vld1q_f32
#define _mm_store_ps vst1q_f32
#define _mm_mul_ps vmulq_f32
#define _mm_load1_ps vdupq_n_f32
#define _mm_add_ps vaddq_f32
#else
#include <immintrin.h>
#endif