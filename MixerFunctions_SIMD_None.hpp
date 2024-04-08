#pragma once

int dotProductSimdNone(const short* const t, const short* const w, int n) {
  int sum = 0;
  while ((n -= 2) >= 0) {
    sum += (t[n] * w[n] + t[n + 1] * w[n + 1]) >> 8;
  }
  return sum;
}

void trainSimdNone(const short* const t, short* const w, int n, const int err) {
  while ((n -= 1) >= 0) {
    int wt = w[n] + ((((t[n] * err * 2) >> 16) + 1) >> 1);
    if (wt < -32768) {
      wt = -32768;
    }
    else if (wt > 32767) {
      wt = 32767;
    }
    *reinterpret_cast<short*>(&w[n]) = wt;
  }
}
