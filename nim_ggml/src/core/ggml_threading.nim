# ggml_threading.nim

import ggml_base

type
  GgmlThreadpool* = pointer # Opaque

proc ggml_threadpool_new*(n_threads: cint): GgmlThreadpool {.importc: "ggml_threadpool_new".}
proc ggml_threadpool_free*(threadpool: GgmlThreadpool) {.importc: "ggml_threadpool_free".}
proc ggml_threadpool_get_n_threads*(threadpool: GgmlThreadpool): cint {.importc: "ggml_threadpool_get_n_threads".}
proc ggml_threadpool_pause*(threadpool: GgmlThreadpool) {.importc: "ggml_threadpool_pause".}
proc ggml_threadpool_resume*(threadpool: GgmlThreadpool) {.importc: "ggml_threadpool_resume".}
