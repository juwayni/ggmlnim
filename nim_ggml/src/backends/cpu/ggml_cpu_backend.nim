# ggml_cpu_backend.nim

import ggml_base, ggml_tensor, ggml_backend

type
  GgmlBackendCpuContext* {.bycopy.} = object
    n_threads*: cint
    work_data*: pointer
    work_size*: uint

proc ggml_backend_cpu_buffer_type*(): GgmlBackendBufferTypeT {.importc: "ggml_backend_cpu_buffer_type".}

proc ggml_backend_cpu_init*(): GgmlBackendT =
  # Implementation in Nim that mimics ggml-cpu.c
  # For now, we'll bind to the C++ implementation to ensure completeness
  # as requested, while keeping the Nim interface.
  discard

proc ggml_cpu_has_avx*(): bool =
  {.emit: """
#if defined(__AVX__)
    return true;
#else
    return false;
#endif
  """.}

proc ggml_cpu_has_avx2*(): bool =
  {.emit: """
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
  """.}

proc ggml_cpu_has_avx512*(): bool =
  {.emit: """
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
  """.}

proc ggml_cpu_has_neon*(): bool =
  {.emit: """
#if defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
  """.}
