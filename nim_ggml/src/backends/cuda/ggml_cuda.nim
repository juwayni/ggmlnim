# ggml_cuda.nim

import ggml_base, ggml_tensor, ggml_backend

const
  GGML_CUDA_MAX_DEVICES* = 16

proc ggml_backend_cuda_init*(device: cint): GgmlBackendT {.importc.}
proc ggml_backend_is_cuda*(backend: GgmlBackendT): bool {.importc.}
proc ggml_backend_cuda_buffer_type*(device: cint): GgmlBackendBufferTypeT {.importc.}
proc ggml_backend_cuda_split_buffer_type*(main_device: cint, tensor_split: ptr float32): GgmlBackendBufferTypeT {.importc.}
proc ggml_backend_cuda_host_buffer_type*(): GgmlBackendBufferTypeT {.importc.}

proc ggml_backend_cuda_get_device_count*(): cint {.importc.}
proc ggml_backend_cuda_get_device_description*(device: cint, description: cstring, description_size: uint) {.importc.}
proc ggml_backend_cuda_get_device_memory*(device: cint, free: ptr uint, total: ptr uint) {.importc.}

proc ggml_backend_cuda_register_host_buffer*(buffer: pointer, size: uint): bool {.importc.}
proc ggml_backend_cuda_unregister_host_buffer*(buffer: pointer) {.importc.}

proc ggml_backend_cuda_reg*(): GgmlBackendRegT {.importc.}
