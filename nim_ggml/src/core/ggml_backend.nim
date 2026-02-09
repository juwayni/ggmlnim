# ggml_backend.nim

import ggml_base, ggml_tensor

type
  GgmlBackendBufferTypeT* = ptr GgmlBackendBufferType
  GgmlBackendBufferT* = ptr GgmlBackendBuffer
  GgmlBackendEventT* = pointer
  GgmlBackendT* = ptr GgmlBackend
  GgmlBackendGraphPlanT* = pointer
  GgmlBackendRegT* = pointer
  GgmlBackendDevT* = pointer

  GgmlBackendBufferUsage* {.size: sizeof(int32).} = enum
    GGML_BACKEND_BUFFER_USAGE_ANY = 0
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2

  GgmlBackendBufferTypeI* {.bycopy.} = object
    get_name*: proc (buft: GgmlBackendBufferTypeT): cstring {.cdecl.}
    alloc_buffer*: proc (buft: GgmlBackendBufferTypeT, size: uint): GgmlBackendBufferT {.cdecl.}
    get_alignment*: proc (buft: GgmlBackendBufferTypeT): uint {.cdecl.}
    get_max_size*: proc (buft: GgmlBackendBufferTypeT): uint {.cdecl.}
    get_alloc_size*: proc (buft: GgmlBackendBufferTypeT, tensor: ptr GgmlTensor): uint {.cdecl.}
    is_host*: proc (buft: GgmlBackendBufferTypeT): bool {.cdecl.}

  GgmlBackendBufferType* {.bycopy.} = object
    iface*: GgmlBackendBufferTypeI
    device*: GgmlBackendDevT
    context*: pointer

  GgmlBackendBufferI* {.bycopy.} = object
    free_buffer*: proc (buffer: GgmlBackendBufferT) {.cdecl.}
    get_base*: proc (buffer: GgmlBackendBufferT): pointer {.cdecl.}
    init_tensor*: proc (buffer: GgmlBackendBufferT, tensor: ptr GgmlTensor): GgmlStatus {.cdecl.}
    memset_tensor*: proc (buffer: GgmlBackendBufferT, tensor: ptr GgmlTensor, value: uint8, offset: uint, size: uint) {.cdecl.}
    set_tensor*: proc (buffer: GgmlBackendBufferT, tensor: ptr GgmlTensor, data: pointer, offset: uint, size: uint) {.cdecl.}
    get_tensor*: proc (buffer: GgmlBackendBufferT, tensor: ptr GgmlTensor, data: pointer, offset: uint, size: uint) {.cdecl.}
    cpy_tensor*: proc (buffer: GgmlBackendBufferT, src: ptr GgmlTensor, dst: ptr GgmlTensor): bool {.cdecl.}
    clear*: proc (buffer: GgmlBackendBufferT, value: uint8) {.cdecl.}
    reset*: proc (buffer: GgmlBackendBufferT) {.cdecl.}

  GgmlBackendBuffer* {.bycopy.} = object
    iface*: GgmlBackendBufferI
    buft*: GgmlBackendBufferTypeT
    context*: pointer
    size*: uint
    usage*: GgmlBackendBufferUsage

  GgmlBackendI* {.bycopy.} = object
    get_name*: proc (backend: GgmlBackendT): cstring {.cdecl.}
    free*: proc (backend: GgmlBackendT) {.cdecl.}
    set_tensor_async*: proc (backend: GgmlBackendT, tensor: ptr GgmlTensor, data: pointer, offset: uint, size: uint) {.cdecl.}
    get_tensor_async*: proc (backend: GgmlBackendT, tensor: ptr GgmlTensor, data: pointer, offset: uint, size: uint) {.cdecl.}
    cpy_tensor_async*: proc (backend_src: GgmlBackendT, backend_dst: GgmlBackendT, src: ptr GgmlTensor, dst: ptr GgmlTensor): bool {.cdecl.}
    synchronize*: proc (backend: GgmlBackendT) {.cdecl.}
    graph_plan_create*: proc (backend: GgmlBackendT, cgraph: pointer): pointer {.cdecl.}
    graph_plan_free*: proc (backend: GgmlBackendT, plan: pointer) {.cdecl.}
    graph_plan_compute*: proc (backend: GgmlBackendT, plan: pointer): GgmlStatus {.cdecl.}
    graph_compute*: proc (backend: GgmlBackendT, cgraph: pointer): GgmlStatus {.cdecl.}
    graph_compute_async*: proc (backend: GgmlBackendT, cgraph: pointer): GgmlStatus {.cdecl.}

  GgmlBackend* {.bycopy.} = object
    iface*: GgmlBackendI
    device*: GgmlBackendDevT
    context*: pointer

  GgmlBackendDevType* {.size: sizeof(int32).} = enum
    GGML_BACKEND_DEVICE_TYPE_CPU
    GGML_BACKEND_DEVICE_TYPE_GPU
    GGML_BACKEND_DEVICE_TYPE_IGPU
    GGML_BACKEND_DEVICE_TYPE_ACCEL

  GgmlBackendDevCaps* {.bycopy.} = object
    async*: bool
    host_buffer*: bool
    buffer_from_host_ptr*: bool
    events*: bool

  GgmlBackendDevProps* {.bycopy.} = object
    name*: cstring
    description*: cstring
    memory_free*: uint
    memory_total*: uint
    `type`*: GgmlBackendDevType
    device_id*: cstring
    caps*: GgmlBackendDevCaps

proc ggml_backend_buft_name*(buft: GgmlBackendBufferTypeT): cstring {.importc.}
proc ggml_backend_buft_alloc_buffer*(buft: GgmlBackendBufferTypeT, size: uint): GgmlBackendBufferT {.importc.}

proc ggml_backend_buffer_name*(buffer: GgmlBackendBufferT): cstring {.importc.}
proc ggml_backend_buffer_free*(buffer: GgmlBackendBufferT) {.importc.}
proc ggml_backend_buffer_get_base*(buffer: GgmlBackendBufferT): pointer {.importc.}
proc ggml_backend_buffer_get_size*(buffer: GgmlBackendBufferT): uint {.importc.}

proc ggml_backend_init_best*(): GgmlBackendT {.importc.}
proc ggml_backend_free*(backend: GgmlBackendT) {.importc.}

# Scheduler types
type
  GgmlBackendSchedT* = pointer
  GgmlBackendSchedEvalCallback* = proc (t: ptr GgmlTensor, ask: bool, userData: pointer): bool {.cdecl.}

proc ggml_backend_sched_new*(backends: ptr GgmlBackendT, bufts: ptr GgmlBackendBufferTypeT, n_backends: cint, graph_size: uint, parallel: bool): GgmlBackendSchedT {.importc.}
proc ggml_backend_sched_free*(sched: GgmlBackendSchedT) {.importc.}
proc ggml_backend_sched_reserve*(sched: GgmlBackendSchedT, measure_graph: pointer): bool {.importc.}
proc ggml_backend_sched_alloc_graph*(sched: GgmlBackendSchedT, graph: pointer): bool {.importc.}
proc ggml_backend_sched_graph_compute*(sched: GgmlBackendSchedT, graph: pointer): GgmlStatus {.importc.}
proc ggml_backend_sched_reset*(sched: GgmlBackendSchedT) {.importc.}

proc ggml_backend_sched_get_n_backends*(sched: GgmlBackendSchedT): cint {.importc.}
proc ggml_backend_sched_get_backend*(sched: GgmlBackendSchedT, i: cint): GgmlBackendT {.importc.}
proc ggml_backend_sched_set_eval_callback*(sched: GgmlBackendSchedT, callback: GgmlBackendSchedEvalCallback, userData: pointer) {.importc.}
