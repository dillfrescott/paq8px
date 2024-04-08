#pragma once

#include "SystemDefines.hpp"

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
int dotProductSimdAvx2(const short* const t, const short* const w, int n) {
  __m256i sum = _mm256_setzero_si256();

  while ((n -= 16) >= 0) {
    __m256i tmp = _mm256_madd_epi16(*(__m256i*) & t[n], *(__m256i*) & w[n]);
    tmp = _mm256_srai_epi32(tmp, 8);
    sum = _mm256_add_epi32(sum, tmp);
  }

  __m128i lo = _mm256_extractf128_si256(sum, 0);
  __m128i hi = _mm256_extractf128_si256(sum, 1);

  __m128i newSum = _mm_hadd_epi32(lo, hi);
  newSum = _mm_add_epi32(newSum, _mm_srli_si128(newSum, 8));
  newSum = _mm_add_epi32(newSum, _mm_srli_si128(newSum, 4));
  return _mm_cvtsi128_si32(newSum);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void trainSimdAvx2(const short* const t, short* const w, int n, const int e) {
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i err = _mm256_set1_epi16(short(e));

  while ((n -= 16) >= 0) {
    __m256i tmp = _mm256_adds_epi16(*(__m256i*) & t[n], *(__m256i*) & t[n]);
    tmp = _mm256_mulhi_epi16(tmp, err);
    tmp = _mm256_adds_epi16(tmp, one);
    tmp = _mm256_srai_epi16(tmp, 1);
    tmp = _mm256_adds_epi16(tmp, *reinterpret_cast<__m256i*>(&w[n]));
    *reinterpret_cast<__m256i*>(&w[n]) = tmp;
  }
}

#endif // X64_SIMD_AVAILABLE