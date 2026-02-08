# ggml_ops_basic.nim - Updated

import ggml_base, ggml_tensor, ggml_context, ggml_graph

proc ggml_dup_tensor*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_new_tensor_impl(ctx, a.`type`, cint(GGML_MAX_DIMS), addr a.ne[0], nil, 0)

proc ggml_view_tensor*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_new_tensor_impl(ctx, a.`type`, cint(GGML_MAX_DIMS), addr a.ne[0], a, 0)

proc ggml_add_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, b: ptr GgmlTensor, inplace: bool): ptr GgmlTensor =
  let result = if inplace: ggml_view_tensor(ctx, a) else: ggml_dup_tensor(ctx, a)
  result.op = GGML_OP_ADD
  result.src[0] = a
  result.src[1] = b
  return result

proc ggml_add*(ctx: ptr GgmlContext, a: ptr GgmlTensor, b: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_add_impl(ctx, a, b, false)

proc ggml_add_inplace*(ctx: ptr GgmlContext, a: ptr GgmlTensor, b: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_add_impl(ctx, a, b, true)

# Unary ops
proc ggml_unary_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, op: GgmlUnaryOp, inplace: bool): ptr GgmlTensor =
  let result = if inplace: ggml_view_tensor(ctx, a) else: ggml_dup_tensor(ctx, a)
  result.op = GGML_OP_UNARY
  result.src[0] = a
  # result.op_params[0] = cast[int32](op)
  return result

proc ggml_relu*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_unary_impl(ctx, a, GGML_UNARY_OP_RELU, false)

proc ggml_silu*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_unary_impl(ctx, a, GGML_UNARY_OP_SILU, false)

proc ggml_exp_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, inplace: bool): ptr GgmlTensor =
  return ggml_unary_impl(ctx, a, GGML_UNARY_OP_EXP, inplace)

proc ggml_exp*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_exp_impl(ctx, a, false)

proc ggml_sqrt_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, inplace: bool): ptr GgmlTensor =
  let result = if inplace: ggml_view_tensor(ctx, a) else: ggml_dup_tensor(ctx, a)
  result.op = GGML_OP_SQRT
  result.src[0] = a
  return result

proc ggml_sqrt*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_sqrt_impl(ctx, a, false)

proc ggml_soft_max_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, inplace: bool): ptr GgmlTensor =
  let result = if inplace: ggml_view_tensor(ctx, a) else: ggml_dup_tensor(ctx, a)
  result.op = GGML_OP_SOFT_MAX
  result.src[0] = a
  return result

proc ggml_soft_max*(ctx: ptr GgmlContext, a: ptr GgmlTensor): ptr GgmlTensor =
  return ggml_soft_max_impl(ctx, a, false)

proc ggml_rms_norm_impl*(ctx: ptr GgmlContext, a: ptr GgmlTensor, eps: float32, inplace: bool): ptr GgmlTensor =
  let result = if inplace: ggml_view_tensor(ctx, a) else: ggml_dup_tensor(ctx, a)
  result.op = GGML_OP_RMS_NORM
  result.src[0] = a
  # result.op_params[0] = cast[int32](eps)
  return result

proc ggml_rms_norm*(ctx: ptr GgmlContext, a: ptr GgmlTensor, eps: float32): ptr GgmlTensor =
  return ggml_rms_norm_impl(ctx, a, eps, false)

proc ggml_conv_2d*(ctx: ptr GgmlContext, a: ptr GgmlTensor, b: ptr GgmlTensor, s0: cint, s1: cint, p0: cint, p1: cint, d0: cint, d1: cint): ptr GgmlTensor =
  var ne: array[GGML_MAX_DIMS, int64]
  # Shape inference for conv2d
  # ...
  let result = ggml_new_tensor_impl(ctx, GGML_TYPE_F32, 4, addr ne[0], nil, 0)
  result.op = GGML_OP_CONV_2D
  result.src[0] = a
  result.src[1] = b
  return result

proc ggml_pool_2d*(ctx: ptr GgmlContext, a: ptr GgmlTensor, `type`: cint, k0: cint, k1: cint, s0: cint, s1: cint, p0: cint, p1: cint): ptr GgmlTensor =
  var ne: array[GGML_MAX_DIMS, int64]
  # Shape inference for pool2d
  # ...
  let result = ggml_new_tensor_impl(ctx, a.`type`, 4, addr ne[0], nil, 0)
  result.op = GGML_OP_POOL_2D
  result.src[0] = a
  return result
