# ggml_alloc.nim

import ggml_base, ggml_tensor, ggml_backend

type
  GgmlTallocr* {.bycopy.} = object
    buffer*: GgmlBackendBufferT
    base*: pointer
    alignment*: uint
    offset*: uint

  GgmlGallocr* = pointer # Opaque handle

proc ggml_tallocr_new*(buffer: GgmlBackendBufferT): GgmlTallocr =
  result.buffer = buffer
  # result.base = ggml_backend_buffer_get_base(buffer)
  # result.alignment = ggml_backend_buffer_get_alignment(buffer)
  result.offset = 0

proc ggml_tallocr_alloc*(talloc: ptr GgmlTallocr, tensor: ptr GgmlTensor): GgmlStatus =
  if tensor.data != nil:
    return GGML_STATUS_SUCCESS

  # size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
  let size = uint(0) # Placeholder for now, should call buffer interface

  # if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
  #    return GGML_STATUS_ALLOC_FAILED;
  # }

  tensor.data = cast[pointer](cast[uint](talloc.base) + talloc.offset)
  talloc.offset += size
  return GGML_STATUS_SUCCESS

# ... and so on for gallocr
