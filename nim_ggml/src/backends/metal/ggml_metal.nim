# ggml_metal.nim

import ggml_base, ggml_tensor, ggml_backend

proc ggml_backend_metal_init*(): GgmlBackendT {.importc.}
proc ggml_backend_is_metal*(backend: GgmlBackendT): bool {.importc.}
proc ggml_backend_metal_set_abort_callback*(backend: GgmlBackendT, abort_callback: pointer, user_data: pointer) {.importc.}
proc ggml_backend_metal_supports_family*(backend: GgmlBackendT, family: cint): bool {.importc.}
proc ggml_backend_metal_capture_next_compute*(backend: GgmlBackendT) {.importc.}
proc ggml_backend_metal_reg*(): GgmlBackendRegT {.importc.}
