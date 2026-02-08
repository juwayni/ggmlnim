# ggml_context.nim

import ggml_base

type
  GgmlObject* {.bycopy.} = object
    offs*: uint
    size*: uint
    next*: ptr GgmlObject
    `type`*: GgmlObjectType
    padding*: array[4, char]

  GgmlContext* {.bycopy.} = object
    mem_size*: uint
    mem_buffer*: pointer
    mem_buffer_owned*: bool
    no_alloc*: bool
    n_objects*: cint
    objects_begin*: ptr GgmlObject
    objects_end*: ptr GgmlObject

  GgmlInitParams* {.bycopy.} = object
    mem_size*: uint
    mem_buffer*: pointer
    no_alloc*: bool

  GgmlScratch* {.bycopy.} = object
    offs*: uint
    size*: uint
    data*: pointer

const
  GGML_OBJECT_SIZE* = sizeof(GgmlObject)
