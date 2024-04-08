#pragma once

#include "SystemDefines.hpp"

#ifdef ARM_NEON_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
static inline int32x4_t _mm_mulhi_epi16(int32x4_t a, int32x4_t b) {
  int32x4_t rl = vmull_s16(vget_low_s16(vreinterpretq_s16_s32(a)), vget_low_s16(vreinterpretq_s16_s32(b)));
  int32x4_t rh = vmull_s16(vget_high_s16(vreinterpretq_s16_s32(a)), vget_high_s16(vreinterpretq_s16_s32(b)));
  uint16x8x2_t r = vuzpq_u16(vreinterpretq_u16_s32(rl), vreinterpretq_u16_s32(rh));
  return vreinterpretq_s32_u16(r.val[1]);
}

static inline int32x4_t _mm_madd_epi16(int32x4_t a, int32x4_t b) {
  int32x4_t pl = vmull_s16(vget_low_s16(vreinterpretq_s16_s32(a)), vget_low_s16(vreinterpretq_s16_s32(b)));
  int32x4_t ph = vmull_s16(vget_high_s16(vreinterpretq_s16_s32(a)), vget_high_s16(vreinterpretq_s16_s32(b)));
  int32x2_t rl = vpadd_s32(vget_low_s32(pl), vget_high_s32(pl));
  int32x2_t rh = vpadd_s32(vget_low_s32(ph), vget_high_s32(ph));
  return vcombine_s32(rl, rh);
}
#endif

int dotProductSimdNeon(const short* const t, const short* const w, int n) {
  int32x4_t sum = vdupq_n_s32(0);

  while ((n -= 8) >= 0) {
    int32x4_t tmp = _mm_madd_epi16(*(int32x4_t*)&t[n], *(int32x4_t*)&w[n]);
    tmp = vshrq_n_s32(tmp, 8);
    sum = vaddq_s32(sum, tmp);
  }

  sum = vaddq_s32(sum, vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(sum), vdupq_n_s8(0), 8)));
  sum = vaddq_s32(sum, vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(sum), vdupq_n_s8(0), 4)));
  return vgetq_lane_s32(sum, 0);
}

void trainSimdNeon(const short* const t, short* const w, int n, const int e) {
  const int32x4_t one = vreinterpretq_s32_s16(vdupq_n_s16(1));
  const int32x4_t err = vreinterpretq_s32_s16(vdupq_n_s16(short(e)));

  while ((n -= 8) >= 0) {
    int32x4_t tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(*(int32x4_t*)&t[n]), vreinterpretq_s16_s32(*(int32x4_t*)&t[n])));
    tmp = _mm_mulhi_epi16(tmp, err);
    tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(tmp), vreinterpretq_s16_s32(one)));
    tmp = vreinterpretq_s32_s16(vshrq_n_s16(vreinterpretq_s16_s32(tmp), (1)));
    tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(tmp), vreinterpretq_s16_s32(*reinterpret_cast<int32x4_t*>(&w[n]))));
    *reinterpret_cast<int32x4_t*>(&w[n]) = tmp;
  }
}

#endif // ARM_NEON_AVAILABLE