# ggml_lib.nim

import ggml_base, ggml_tensor, ggml_context, ggml_context_impl, ggml_graph, ggml_quants

export ggml_base, ggml_tensor, ggml_context, ggml_context_impl, ggml_graph, ggml_quants

proc ggml_log_set*(cb: GgmlLogCallback, userData: pointer) {.importc: "ggml_log_set".}
proc ggml_time_ms*(): int64 {.importc: "ggml_time_ms".}
proc ggml_time_us*(): int64 {.importc: "ggml_time_us".}

# ... and so on for all API functions
