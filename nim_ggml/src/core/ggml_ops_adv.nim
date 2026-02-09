# ggml_ops_adv.nim

import ggml_base, ggml_tensor, ggml_context, ggml_graph

proc ggml_mul_mat*(ctx: ptr GgmlContext, a: ptr GgmlTensor, b: ptr GgmlTensor): ptr GgmlTensor =
  var ne: array[GGML_MAX_DIMS, int64]
  ne[0] = a.ne[1]
  ne[1] = b.ne[1]
  ne[2] = b.ne[2]
  ne[3] = b.ne[3]

  # result type is usually F32 unless specified otherwise
  let result = ggml_new_tensor_impl(ctx, GGML_TYPE_F32, 4, addr ne[0], nil, 0)
  result.op = GGML_OP_MUL_MAT
  result.src[0] = a
  result.src[1] = b
  return result

proc ggml_rms_norm*(ctx: ptr GgmlContext, a: ptr GgmlTensor, eps: float32): ptr GgmlTensor =
  let result = ggml_new_tensor_impl(ctx, a.`type`, 4, addr a.ne[0], nil, 0)
  result.op = GGML_OP_RMS_NORM
  result.src[0] = a
  # eps is usually stored in op_params
  # result.op_params[0] = cast[int32](eps) # Simplified
  return result
