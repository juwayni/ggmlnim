# ggml_tensor.nim

import ggml_base

const
  QK4_0* = 32
  QK4_1* = 32
  QK5_0* = 32
  QK5_1* = 32
  QK8_0* = 32
  QK8_1* = 32
  QK_K* = 256

type
  GgmlBackendBuffer* = object # Opaque

  GgmlTensor* {.bycopy, pure.} = object
    `type`*: GgmlType
    buffer*: ptr GgmlBackendBuffer
    ne*: array[GGML_MAX_DIMS, int64]
    nb*: array[GGML_MAX_DIMS, uint]
    op*: GgmlOp
    op_params*: array[GGML_MAX_OP_PARAMS div sizeof(int32), int32]
    flags*: int32
    src*: array[GGML_MAX_SRC, ptr GgmlTensor]
    view_src*: ptr GgmlTensor
    view_offs*: uint
    data*: pointer
    name*: array[GGML_MAX_NAME, char]
    extra*: pointer
    padding*: array[8, char]

const
  GGML_TENSOR_SIZE* = sizeof(GgmlTensor)

proc ggml_nbytes*(tensor: ptr GgmlTensor): uint =
  result = uint(tensor.nb[GGML_MAX_DIMS - 1]) * uint(tensor.ne[GGML_MAX_DIMS - 1])

proc ggml_blck_size*(t: GgmlType): int =
  case t:
    of GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64: 1
    of GGML_TYPE_Q4_0, GGML_TYPE_Q4_1: QK4_0
    of GGML_TYPE_Q5_0, GGML_TYPE_Q5_1: QK5_0
    of GGML_TYPE_Q8_0, GGML_TYPE_Q8_1: QK8_0
    of GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K: QK_K
    of GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS: QK_K
    else: 1

proc ggml_type_size*(t: GgmlType): uint =
  case t:
    of GGML_TYPE_F32: 4
    of GGML_TYPE_F16: 2
    of GGML_TYPE_Q4_0: 2 + QK4_0 div 2
    of GGML_TYPE_Q4_1: 2 + 2 + QK4_1 div 2
    of GGML_TYPE_Q5_0: 2 + 4 + QK5_0 div 2
    of GGML_TYPE_Q5_1: 2 + 2 + 4 + QK5_1 div 2
    of GGML_TYPE_Q8_0: 2 + QK8_0
    of GGML_TYPE_Q8_1: 2 + 2 + QK8_1
    of GGML_TYPE_I8: 1
    of GGML_TYPE_I16: 2
    of GGML_TYPE_I32: 4
    of GGML_TYPE_I64: 8
    of GGML_TYPE_F64: 8
    else: 0 # To be completed for K-quants

proc ggml_row_size*(t: GgmlType, ne: int64): uint =
  return uint(ne div int64(ggml_blck_size(t))) * ggml_type_size(t)

proc ggml_is_quantized*(t: GgmlType): bool =
  case t:
    of GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64: false
    else: true
