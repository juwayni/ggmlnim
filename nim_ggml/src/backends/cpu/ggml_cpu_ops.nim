# ggml_cpu_ops.nim

import ggml_base, ggml_tensor, ggml_ops, ggml_cpu_vec

proc ggml_compute_forward_add*(tensor: ptr GgmlTensor) =
  let src0 = tensor.src[0]
  let src1 = tensor.src[1]
  let dst = tensor

  # Simplified for F32
  if src0.`type` == GGML_TYPE_F32 and src1.`type` == GGML_TYPE_F32:
    let n = cint(ggml_nbytes(src0) div uint(sizeof(float32)))
    ggml_vec_add_f32(n, cast[ptr float32](dst.data), cast[ptr float32](src0.data), cast[ptr float32](src1.data))

proc ggml_compute_forward*(tensor: ptr GgmlTensor, n_threads: cint) =
  case tensor.op:
    of GGML_OP_ADD:
      ggml_compute_forward_add(tensor)
    of GGML_OP_MUL:
      # ggml_compute_forward_mul(tensor)
      discard
    else:
      discard
