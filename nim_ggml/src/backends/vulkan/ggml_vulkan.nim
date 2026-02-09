# ggml_vulkan.nim

import ggml_base, ggml_tensor, ggml_backend

const
  GGML_VULKAN_MAX_DEVICES* = 16

proc ggml_backend_vulkan_init*(device: cint): GgmlBackendT {.importc.}
proc ggml_backend_is_vulkan*(backend: GgmlBackendT): bool {.importc.}
proc ggml_backend_vulkan_buffer_type*(device: cint): GgmlBackendBufferTypeT {.importc.}
proc ggml_backend_vulkan_get_device_count*(): cint {.importc.}
proc ggml_backend_vulkan_get_device_description*(device: cint, description: cstring, description_size: uint) {.importc.}
proc ggml_backend_vulkan_get_device_memory*(device: cint, free: ptr uint, total: ptr uint) {.importc.}
proc ggml_backend_vulkan_reg*(): GgmlBackendRegT {.importc.}
