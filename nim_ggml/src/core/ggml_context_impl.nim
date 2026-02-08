# ggml_context_impl.nim

import ggml_base, ggml_context

proc posix_memalign*(memptr: ptr pointer, alignment: uint, size: uint): cint {.importc: "posix_memalign", header: "<stdlib.h>".}
proc free*(p: pointer) {.importc: "free", header: "<stdlib.h>".}

proc ggml_aligned_malloc*(size: uint): pointer =
  if size == 0: return nil
  let alignment: uint = 64
  var res: pointer
  let err = posix_memalign(addr res, alignment, size)
  if err != 0:
    return nil
  return res

proc ggml_init*(params: GgmlInitParams): ptr GgmlContext =
  var mem_size = params.mem_size
  var mem_buffer = params.mem_buffer
  var mem_buffer_owned = false

  if mem_buffer == nil and not params.no_alloc:
    mem_buffer = ggml_aligned_malloc(mem_size)
    mem_buffer_owned = true

  var ctx = cast[ptr GgmlContext](ggml_aligned_malloc(uint(sizeof(GgmlContext))))
  if ctx == nil: return nil

  ctx.mem_size = mem_size
  ctx.mem_buffer = mem_buffer
  ctx.mem_buffer_owned = mem_buffer_owned
  ctx.no_alloc = params.no_alloc
  ctx.n_objects = 0
  ctx.objects_begin = nil
  ctx.objects_end = nil

  return ctx

proc ggml_free*(ctx: ptr GgmlContext) =
  if ctx == nil: return
  if ctx.mem_buffer_owned:
    free(ctx.mem_buffer)
  free(ctx)
