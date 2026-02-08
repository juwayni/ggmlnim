# ggml_graph.nim

import ggml_base, ggml_tensor, ggml_context, ggml_context_impl

type
  GgmlCGraphEvalOrder* {.size: sizeof(int32).} = enum
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1

  GgmlHashSet* {.bycopy.} = object
    size*: uint
    used*: ptr uint64 # bitset
    keys*: ptr ptr GgmlTensor

  GgmlCGraph* {.bycopy.} = object
    size*: cint
    n_nodes*: cint
    n_leafs*: cint
    nodes*: ptr ptr GgmlTensor
    grads*: ptr ptr GgmlTensor
    grad_accs*: ptr ptr GgmlTensor
    leafs*: ptr ptr GgmlTensor
    use_counts*: ptr int32
    visited_hash_set*: GgmlHashSet
    order*: GgmlCGraphEvalOrder

proc ggml_type_size*(t: GgmlType): uint =
  case t:
    of GGML_TYPE_F32: 4
    of GGML_TYPE_F16: 2
    of GGML_TYPE_I8: 1
    of GGML_TYPE_I16: 2
    of GGML_TYPE_I32: 4
    of GGML_TYPE_I64: 8
    of GGML_TYPE_F64: 8
    else: 0 # Simplified

proc ggml_blck_size*(t: GgmlType): uint =
  case t:
    of GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64: 1
    of GGML_TYPE_Q4_0, GGML_TYPE_Q4_1: 32
    of GGML_TYPE_Q5_0, GGML_TYPE_Q5_1: 32
    of GGML_TYPE_Q8_0: 32
    else: 32 # Simplified

proc ggml_row_size*(t: GgmlType, ne: int64): uint =
  return uint(ne div int64(ggml_blck_size(t))) * ggml_type_size(t)

proc ggml_new_object*(ctx: ptr GgmlContext, `type`: GgmlObjectType, size: uint): ptr GgmlObject =
  let alignedSize = GGML_PAD(size, GGML_MEM_ALIGN)
  var offs: uint = 0
  if ctx.objects_end != nil:
    offs = ctx.objects_end.offs + ctx.objects_end.size

  if offs + alignedSize > ctx.mem_size:
    # fprintf(stderr, "%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
    #         __func__, alignedSize, ctx->mem_size - offs);
    return nil

  let obj = cast[ptr GgmlObject](cast[uint](ctx.mem_buffer) + offs)
  obj.offs = offs
  obj.size = alignedSize
  obj.next = nil
  obj.`type` = `type`

  if ctx.objects_end != nil:
    ctx.objects_end.next = obj
  else:
    ctx.objects_begin = obj

  ctx.objects_end = obj
  return obj

proc ggml_new_tensor_impl*(ctx: ptr GgmlContext, `type`: GgmlType, n_dims: cint, ne: ptr int64, view_src: ptr GgmlTensor, view_offs: uint): ptr GgmlTensor =
  var dataSize: uint = ggml_row_size(`type`, (cast[ptr UncheckedArray[int64]](ne))[0])
  for i in 1..<n_dims:
    dataSize *= uint((cast[ptr UncheckedArray[int64]](ne))[i])

  var data: pointer = nil
  if view_src != nil:
    data = view_src.data
    if data != nil:
      data = cast[pointer](cast[uint](data) + view_offs)

  var obj_alloc_size: uint = 0
  if view_src == nil and not ctx.no_alloc:
    obj_alloc_size = dataSize

  let obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, uint(sizeof(GgmlTensor)) + obj_alloc_size)
  if obj_new == nil: return nil

  let result = cast[ptr GgmlTensor](cast[uint](ctx.mem_buffer) + obj_new.offs)

  result.`type` = `type`
  result.buffer = nil
  for i in 0..<GGML_MAX_DIMS:
    result.ne[i] = 1
    result.nb[i] = 0
  result.op = GGML_OP_NONE
  # op_params and others should be zeroed

  result.view_src = view_src
  result.view_offs = view_offs
  if obj_alloc_size > 0:
    result.data = cast[pointer](cast[uint](result) + uint(sizeof(GgmlTensor)))
  else:
    result.data = data

  for i in 0..<n_dims:
    result.ne[i] = (cast[ptr UncheckedArray[int64]](ne))[i]

  result.nb[0] = ggml_type_size(`type`)
  result.nb[1] = result.nb[0] * uint(result.ne[0] div int64(ggml_blck_size(`type`)))
  for i in 2..<GGML_MAX_DIMS:
    result.nb[i] = result.nb[i-1] * uint(result.ne[i-1])

  ctx.n_objects += 1
  return result

proc ggml_new_tensor_1d*(ctx: ptr GgmlContext, `type`: GgmlType, ne0: int64): ptr GgmlTensor =
  var ne = [ne0, 1'i64, 1'i64, 1'i64]
  return ggml_new_tensor_impl(ctx, `type`, 1, addr ne[0], nil, 0)

proc ggml_hash*(p: pointer): uint =
  var x = cast[uint](p)
  x = (x shr 16) xor x
  x *= 0x45d9f3b
  x = (x shr 16) xor x
  x *= 0x45d9f3b
  x = (x shr 16) xor x
  return x

proc ggml_hash_set_find*(set: GgmlHashSet, key: ptr GgmlTensor): uint =
  let size = set.size
  var idx = ggml_hash(key) mod size
  let keys = cast[ptr UncheckedArray[ptr GgmlTensor]](set.keys)
  let used = cast[ptr UncheckedArray[uint64]](set.used)

  while (used[idx div 64] and (1'u64 shl (idx mod 64))) != 0:
    if keys[idx] == key:
      return idx
    idx = (idx + 1) mod size

  return uint.high

proc ggml_hash_set_insert*(set: var GgmlHashSet, key: ptr GgmlTensor): bool =
  let size = set.size
  var idx = ggml_hash(key) mod size
  let keys = cast[ptr UncheckedArray[ptr GgmlTensor]](set.keys)
  let used = cast[ptr UncheckedArray[uint64]](set.used)

  while (used[idx div 64] and (1'u64 shl (idx mod 64))) != 0:
    if keys[idx] == key:
      return false
    idx = (idx + 1) mod size

  keys[idx] = key
  used[idx div 64] = used[idx div 64] or (1'u64 shl (idx mod 64))
  return true
