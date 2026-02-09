# ggml_base.nim - Core enums, constants, and basic types

import std/os

const
  GGML_MAX_DIMS* = 4
  GGML_MAX_PARAMS* = 2048
  GGML_MAX_CONTEXTS* = 64
  GGML_MAX_SRC* = 10
  GGML_MAX_NAME* = 64
  GGML_MAX_OP_PARAMS* = 64
  GGML_DEFAULT_GRAPH_SIZE* = 2048

type
  GgmlStatus* {.size: sizeof(int32).} = enum
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED = -1
    GGML_STATUS_SUCCESS = 0
    GGML_STATUS_ABORTED = 1

  GgmlType* {.size: sizeof(int32).} = enum
    GGML_TYPE_F32     = 0
    GGML_TYPE_F16     = 1
    GGML_TYPE_Q4_0    = 2
    GGML_TYPE_Q4_1    = 3
    GGML_TYPE_Q5_0    = 6
    GGML_TYPE_Q5_1    = 7
    GGML_TYPE_Q8_0    = 8
    GGML_TYPE_Q8_1    = 9
    GGML_TYPE_Q2_K    = 10
    GGML_TYPE_Q3_K    = 11
    GGML_TYPE_Q4_K    = 12
    GGML_TYPE_Q5_K    = 13
    GGML_TYPE_Q6_K    = 14
    GGML_TYPE_Q8_K    = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S   = 19
    GGML_TYPE_IQ4_NL  = 20
    GGML_TYPE_IQ3_S   = 21
    GGML_TYPE_IQ2_S   = 22
    GGML_TYPE_IQ4_XS  = 23
    GGML_TYPE_I8      = 24
    GGML_TYPE_I16     = 25
    GGML_TYPE_I32     = 26
    GGML_TYPE_I64     = 27
    GGML_TYPE_F64     = 28
    GGML_TYPE_IQ1_M   = 29
    GGML_TYPE_BF16    = 30
    GGML_TYPE_TQ1_0   = 34
    GGML_TYPE_TQ2_0   = 35
    GGML_TYPE_MXFP4   = 39
    GGML_TYPE_COUNT   = 40

  GgmlPrec* {.size: sizeof(int32).} = enum
    GGML_PREC_DEFAULT = 0
    GGML_PREC_F32     = 10

  GgmlFtype* {.size: sizeof(int32).} = enum
    GGML_FTYPE_UNKNOWN        = -1
    GGML_FTYPE_ALL_F32        = 0
    GGML_FTYPE_MOSTLY_F16     = 1
    GGML_FTYPE_MOSTLY_Q4_0    = 2
    GGML_FTYPE_MOSTLY_Q4_1    = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0    = 7
    GGML_FTYPE_MOSTLY_Q5_0    = 8
    GGML_FTYPE_MOSTLY_Q5_1    = 9
    GGML_FTYPE_MOSTLY_Q2_K    = 10
    GGML_FTYPE_MOSTLY_Q3_K    = 11
    GGML_FTYPE_MOSTLY_Q4_K    = 12
    GGML_FTYPE_MOSTLY_Q5_K    = 13
    GGML_FTYPE_MOSTLY_Q6_K    = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS  = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S   = 18
    GGML_FTYPE_MOSTLY_IQ4_NL  = 19
    GGML_FTYPE_MOSTLY_IQ3_S   = 20
    GGML_FTYPE_MOSTLY_IQ2_S   = 21
    GGML_FTYPE_MOSTLY_IQ4_XS  = 22
    GGML_FTYPE_MOSTLY_IQ1_M   = 23
    GGML_FTYPE_MOSTLY_BF16    = 24
    GGML_FTYPE_MOSTLY_MXFP4   = 25

  GgmlOp* {.size: sizeof(int32).} = enum
    GGML_OP_NONE = 0
    GGML_OP_DUP
    GGML_OP_ADD
    GGML_OP_ADD_ID
    GGML_OP_ADD1
    GGML_OP_ACC
    GGML_OP_SUB
    GGML_OP_MUL
    GGML_OP_DIV
    GGML_OP_SQR
    GGML_OP_SQRT
    GGML_OP_LOG
    GGML_OP_SIN
    GGML_OP_COS
    GGML_OP_SUM
    GGML_OP_SUM_ROWS
    GGML_OP_CUMSUM
    GGML_OP_MEAN
    GGML_OP_ARGMAX
    GGML_OP_COUNT_EQUAL
    GGML_OP_REPEAT
    GGML_OP_REPEAT_BACK
    GGML_OP_CONCAT
    GGML_OP_SILU_BACK
    GGML_OP_NORM
    GGML_OP_RMS_NORM
    GGML_OP_RMS_NORM_BACK
    GGML_OP_GROUP_NORM
    GGML_OP_L2_NORM
    GGML_OP_MUL_MAT
    GGML_OP_MUL_MAT_ID
    GGML_OP_OUT_PROD
    GGML_OP_SCALE
    GGML_OP_SET
    GGML_OP_CPY
    GGML_OP_CONT
    GGML_OP_RESHAPE
    GGML_OP_VIEW
    GGML_OP_PERMUTE
    GGML_OP_TRANSPOSE
    GGML_OP_GET_ROWS
    GGML_OP_GET_ROWS_BACK
    GGML_OP_SET_ROWS
    GGML_OP_DIAG
    GGML_OP_DIAG_MASK_INF
    GGML_OP_DIAG_MASK_ZERO
    GGML_OP_SOFT_MAX
    GGML_OP_SOFT_MAX_BACK
    GGML_OP_ROPE
    GGML_OP_ROPE_BACK
    GGML_OP_CLAMP
    GGML_OP_CONV_TRANSPOSE_1D
    GGML_OP_IM2COL
    GGML_OP_IM2COL_BACK
    GGML_OP_IM2COL_3D
    GGML_OP_CONV_2D
    GGML_OP_CONV_3D
    GGML_OP_CONV_2D_DW
    GGML_OP_CONV_TRANSPOSE_2D
    GGML_OP_POOL_1D
    GGML_OP_POOL_2D
    GGML_OP_POOL_2D_BACK
    GGML_OP_UPSCALE
    GGML_OP_PAD
    GGML_OP_PAD_REFLECT_1D
    GGML_OP_ROLL
    GGML_OP_ARANGE
    GGML_OP_TIMESTEP_EMBEDDING
    GGML_OP_ARGSORT
    GGML_OP_TOP_K
    GGML_OP_LEAKY_RELU
    GGML_OP_TRI
    GGML_OP_FILL
    GGML_OP_FLASH_ATTN_EXT
    GGML_OP_FLASH_ATTN_BACK
    GGML_OP_SSM_CONV
    GGML_OP_SSM_SCAN
    GGML_OP_WIN_PART
    GGML_OP_WIN_UNPART
    GGML_OP_GET_REL_POS
    GGML_OP_ADD_REL_POS
    GGML_OP_RWKV_WKV6
    GGML_OP_GATED_LINEAR_ATTN
    GGML_OP_RWKV_WKV7
    GGML_OP_SOLVE_TRI
    GGML_OP_UNARY
    GGML_OP_MAP_CUSTOM1
    GGML_OP_MAP_CUSTOM2
    GGML_OP_MAP_CUSTOM3
    GGML_OP_CUSTOM
    GGML_OP_CROSS_ENTROPY_LOSS
    GGML_OP_CROSS_ENTROPY_LOSS_BACK
    GGML_OP_OPT_STEP_ADAMW
    GGML_OP_OPT_STEP_SGD
    GGML_OP_GLU
    GGML_OP_COUNT

  GgmlUnaryOp* {.size: sizeof(int32).} = enum
    GGML_UNARY_OP_ABS
    GGML_UNARY_OP_SGN
    GGML_UNARY_OP_NEG
    GGML_UNARY_OP_STEP
    GGML_UNARY_OP_TANH
    GGML_UNARY_OP_ELU
    GGML_UNARY_OP_RELU
    GGML_UNARY_OP_SIGMOID
    GGML_UNARY_OP_GELU
    GGML_UNARY_OP_GELU_QUICK
    GGML_UNARY_OP_SILU
    GGML_UNARY_OP_HARDSWISH
    GGML_UNARY_OP_HARDSIGMOID
    GGML_UNARY_OP_EXP
    GGML_UNARY_OP_EXPM1
    GGML_UNARY_OP_SOFTPLUS
    GGML_UNARY_OP_GELU_ERF
    GGML_UNARY_OP_XIELU
    GGML_UNARY_OP_FLOOR
    GGML_UNARY_OP_CEIL
    GGML_UNARY_OP_ROUND
    GGML_UNARY_OP_TRUNC
    GGML_UNARY_OP_COUNT

  GgmlGluOp* {.size: sizeof(int32).} = enum
    GGML_GLU_OP_REGLU
    GGML_GLU_OP_GEGLU
    GGML_GLU_OP_SWIGLU
    GGML_GLU_OP_SWIGLU_OAI
    GGML_GLU_OP_GEGLU_ERF
    GGML_GLU_OP_GEGLU_QUICK
    GGML_GLU_OP_COUNT

  GgmlObjectType* {.size: sizeof(int32).} = enum
    GGML_OBJECT_TYPE_TENSOR
    GGML_OBJECT_TYPE_GRAPH
    GGML_OBJECT_TYPE_WORK_BUFFER

  GgmlLogLevel* {.size: sizeof(int32).} = enum
    GGML_LOG_LEVEL_NONE  = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO  = 2
    GGML_LOG_LEVEL_WARN  = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT  = 5

  GgmlTensorFlag* {.size: sizeof(int32).} = enum
    GGML_TENSOR_FLAG_INPUT   =  1
    GGML_TENSOR_FLAG_OUTPUT  =  2
    GGML_TENSOR_FLAG_PARAM   =  4
    GGML_TENSOR_FLAG_LOSS    =  8
    GGML_TENSOR_FLAG_COMPUTE = 16

  GgmlTriType* {.size: sizeof(int32).} = enum
    GGML_TRI_TYPE_UPPER_DIAG = 0
    GGML_TRI_TYPE_UPPER      = 1
    GGML_TRI_TYPE_LOWER_DIAG = 2
    GGML_TRI_TYPE_LOWER      = 3

  GgmlAbortCallbackT* = proc (errorMessage: cstring) {.cdecl.}
  GgmlLogCallback* = proc (level: GgmlLogLevel, text: cstring, userData: pointer) {.cdecl.}

  GgmlFp16t* = uint16
  GgmlBf16t* {.bycopy.} = object
    bits*: uint16

  GgmlGuid* = array[16, uint8]
  GgmlGuidT* = GgmlGuid

const
  GGML_ROPE_TYPE_NORMAL* = 0
  GGML_ROPE_TYPE_NEOX*   = 2
  GGML_ROPE_TYPE_MROPE*  = 8
  GGML_ROPE_TYPE_VISION* = 24
  GGML_ROPE_TYPE_IMROPE* = 40

  GGML_MROPE_SECTIONS* = 4

template GGML_PAD*(x, n: untyped): untyped =
  ((x + n - 1) and not (n - 1))

proc ggml_fp16_to_fp32*(h: uint16): float32 =
  # Native conversion or software fallback
  {.emit: """
    float f;
    uint16_t x = `h`;
    uint32_t exp = (x & 0x7c00) >> 10;
    uint32_t mant = (x & 0x03ff);
    if (exp == 0) f = ldexp((float)mant, -24);
    else if (exp != 31) f = ldexp((float)mant + 1024, (int)exp - 25);
    else f = mant == 0 ? INFINITY : NAN;
    return x & 0x8000 ? -f : f;
  """.}

proc ggml_fp32_to_fp16*(f: float32): uint16 =
  # Native conversion or software fallback
  {.emit: """
    float x = `f`;
    uint32_t b = *((uint32_t*)&x);
    uint32_t s = (b >> 16) & 0x8000;
    uint32_t e = (b >> 23) & 0xff;
    uint32_t m = b & 0x7fffff;
    if (e <= 112) return s;
    if (e >= 143) return s | 0x7c00 | (e == 255 ? (m != 0) : 0);
    return s | ((e - 112) << 10) | (m >> 13);
  """.}

# Memory alignment constants
# In C:
# #if UINTPTR_MAX == 0xFFFFFFFF
#     #define GGML_MEM_ALIGN 4
# #elif defined(__EMSCRIPTEN__)
#     #define GGML_MEM_ALIGN 8
# #else
#     #define GGML_MEM_ALIGN 16
# #endif
const
  GGML_MEM_ALIGN* = 16 # Default to 16 for 64-bit systems
