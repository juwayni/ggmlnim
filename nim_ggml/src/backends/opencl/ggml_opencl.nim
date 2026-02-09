# ggml_opencl.nim

import ggml_base, ggml_tensor, ggml_backend

proc ggml_backend_opencl_init*(): GgmlBackendT {.importc.}
proc ggml_backend_is_opencl*(backend: GgmlBackendT): bool {.importc.}
proc ggml_backend_opencl_buffer_type*(): GgmlBackendBufferTypeT {.importc.}
proc ggml_backend_opencl_host_buffer_type*(): GgmlBackendBufferTypeT {.importc.}
proc ggml_backend_opencl_reg*(): GgmlBackendRegT {.importc.}
