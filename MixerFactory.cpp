#include "MixerFactory.hpp"

MixerFactory::MixerFactory(const Shared* const sh) : shared(sh)
{}

Mixer* MixerFactory::createMixer(const int n, const int m, const int s, const int promoted) const {
  const SIMDType chosenSimd = shared->chosenSimd;
  if (chosenSimd == SIMDType::SIMD_NONE) {
    return new SIMDMixer<SIMDType::SIMD_NONE>(shared, n, m, s, promoted);
  }
#ifdef X64_SIMD_AVAILABLE
  else if (chosenSimd == SIMDType::SIMD_SSE2) {
    return new SIMDMixer<SIMDType::SIMD_SSE2>(shared, n, m, s, promoted);
  }
  else if (chosenSimd == SIMDType::SIMD_AVX2) {
    return new SIMDMixer<SIMDType::SIMD_AVX2>(shared, n, m, s, promoted);
  }
  else if (chosenSimd == SIMDType::SIMD_AVX512) {
    return new SIMDMixer<SIMDType::SIMD_AVX512>(shared, n, m, s, promoted);
  }
#endif
#ifdef ARM_NEON_AVAILABLE
  else if (chosenSimd == SIMDType::SIMD_NEON) {
    return new SIMDMixer<SIMDType::SIMD_NEON>(shared, n, m, s, promoted);
  }
#endif
  assert(false);
  return nullptr;
}
