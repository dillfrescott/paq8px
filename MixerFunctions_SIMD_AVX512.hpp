#pragma once

#include "SystemDefines.hpp"

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512bw")))
#endif
int dotProductSimdAvx512(const short* const t, const short* const w, int n) {
  __m512i sum = _mm512_setzero_si512();

  while ((n -= 32) >= 0) {
    __m512i tmp = _mm512_madd_epi16(*(__m512i*)&t[n], *(__m512i*)&w[n]);
    tmp = _mm512_srai_epi32(tmp, 8);
    sum = _mm512_add_epi32(sum, tmp);
  }

  __m256i lo = _mm512_extracti64x4_epi64(sum, 0);
  __m256i hi = _mm512_extracti64x4_epi64(sum, 1);

  __m256i newSum1 = _mm256_add_epi32(lo, hi);
  __m128i newSum2 = _mm_add_epi32(_mm256_extractf128_si256(newSum1, 0), _mm256_extractf128_si256(newSum1, 1));
  newSum2 = _mm_add_epi32(newSum2, _mm_srli_si128(newSum2, 8));
  newSum2 = _mm_add_epi32(newSum2, _mm_srli_si128(newSum2, 4));
  return _mm_cvtsi128_si32(newSum2);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512bw")))
#endif
void trainSimdAvx512(const short* const t, short* const w, int n, const int e) {
  const __m512i one = _mm512_set1_epi16(1);
  const __m512i err = _mm512_set1_epi16(short(e));

  while ((n -= 32) >= 0) {
    __m512i tmp = _mm512_adds_epi16(*(__m512i*)&t[n], *(__m512i*)&t[n]);
    tmp = _mm512_mulhi_epi16(tmp, err);
    tmp = _mm512_adds_epi16(tmp, one);
    tmp = _mm512_srai_epi16(tmp, 1);
    tmp = _mm512_adds_epi16(tmp, *reinterpret_cast<__m512i*>(&w[n]));
    *reinterpret_cast<__m512i*>(&w[n]) = tmp;
  }
}

#endif // X64_SIMD_AVAILABLE