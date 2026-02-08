# gguf.nim

import ggml_base

type
  GgufContext* = pointer # Opaque

  GgufType* {.size: sizeof(int32).} = enum
    GGUF_TYPE_UINT8   = 0
    GGUF_TYPE_INT8    = 1
    GGUF_TYPE_UINT16  = 2
    GGUF_TYPE_INT16   = 3
    GGUF_TYPE_UINT32  = 4
    GGUF_TYPE_INT32   = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL    = 7
    GGUF_TYPE_STRING  = 8
    GGUF_TYPE_ARRAY   = 9
    GGUF_TYPE_UINT64  = 10
    GGUF_TYPE_INT64   = 11
    GGUF_TYPE_FLOAT64 = 12

proc gguf_init_empty*(): GgufContext {.importc: "gguf_init_empty".}
proc gguf_init_from_file*(fname: cstring, params: pointer): GgufContext {.importc: "gguf_init_from_file".}
proc gguf_free*(ctx: GgufContext) {.importc: "gguf_free".}

proc gguf_get_key*(ctx: GgufContext, key_id: cint): cstring {.importc: "gguf_get_key".}
proc gguf_get_val_u8*(ctx: GgufContext, key_id: cint): uint8 {.importc: "gguf_get_val_u8".}
proc gguf_get_val_i8*(ctx: GgufContext, key_id: cint): int8 {.importc: "gguf_get_val_i8".}
proc gguf_get_val_u16*(ctx: GgufContext, key_id: cint): uint16 {.importc: "gguf_get_val_u16".}
proc gguf_get_val_i16*(ctx: GgufContext, key_id: cint): int16 {.importc: "gguf_get_val_i16".}
proc gguf_get_val_u32*(ctx: GgufContext, key_id: cint): uint32 {.importc: "gguf_get_val_u32".}
proc gguf_get_val_i32*(ctx: GgufContext, key_id: cint): int32 {.importc: "gguf_get_val_i32".}
proc gguf_get_val_f32*(ctx: GgufContext, key_id: cint): float32 {.importc: "gguf_get_val_f32".}
proc gguf_get_val_u64*(ctx: GgufContext, key_id: cint): uint64 {.importc: "gguf_get_val_u64".}
proc gguf_get_val_i64*(ctx: GgufContext, key_id: cint): int64 {.importc: "gguf_get_val_i64".}
proc gguf_get_val_f64*(ctx: GgufContext, key_id: cint): float64 {.importc: "gguf_get_val_f64".}
proc gguf_get_val_bool*(ctx: GgufContext, key_id: cint): bool {.importc: "gguf_get_val_bool".}
proc gguf_get_val_str*(ctx: GgufContext, key_id: cint): cstring {.importc: "gguf_get_val_str".}
proc gguf_get_val_data*(ctx: GgufContext, key_id: cint): pointer {.importc: "gguf_get_val_data".}
proc gguf_get_arr_n*(ctx: GgufContext, key_id: cint): cint {.importc: "gguf_get_arr_n".}
proc gguf_get_arr_data*(ctx: GgufContext, key_id: cint): pointer {.importc: "gguf_get_arr_data".}
proc gguf_get_n_kv*(ctx: GgufContext): cint {.importc: "gguf_get_n_kv".}
proc gguf_find_key*(ctx: GgufContext, key: cstring): cint {.importc: "gguf_find_key".}
