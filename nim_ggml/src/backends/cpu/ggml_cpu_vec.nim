# ggml_cpu_vec.nim

import ggml_base

{.emit: """
#include <immintrin.h>
#include <cmath>
""".}

proc ggml_vec_add_f32*(n: cint, z: ptr float32, x: ptr float32, y: ptr float32) {.exportc.} =
  {.emit: """
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < `n`; i += 8) {
        __m256 vx = _mm256_loadu_ps(`x` + i);
        __m256 vy = _mm256_loadu_ps(`y` + i);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(`z` + i, vz);
    }
#endif
    for (; i < `n`; ++i) {
        `z`[i] = `x`[i] + `y`[i];
    }
  """.}

proc ggml_vec_mul_f32*(n: cint, z: ptr float32, x: ptr float32, y: ptr float32) {.exportc.} =
  {.emit: """
    for (int i = 0; i < `n`; ++i) {
        `z`[i] = `x`[i] * `y`[i];
    }
  """.}

# Dot products, etc. to be ported similarly
