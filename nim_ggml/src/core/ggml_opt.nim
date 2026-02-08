# ggml_opt.nim

import ggml_base, ggml_tensor, ggml_context

type
  GgmlOptType* {.size: sizeof(int32).} = enum
    GGML_OPT_TYPE_ADAM
    GGML_OPT_TYPE_LBFGS

  GgmlOptParams* {.bycopy.} = object
    `type`*: GgmlOptType
    n_threads*: cint
    # ... many more fields

proc ggml_opt_default_params*(type: GgmlOptType): GgmlOptParams {.importc: "ggml_opt_default_params".}
proc ggml_opt*(ctx: ptr GgmlContext, params: GgmlOptParams, f: ptr GgmlTensor): GgmlStatus {.importc: "ggml_opt".}
