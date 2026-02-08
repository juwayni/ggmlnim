# test_nim_ggml.nim

import ../src/core/ggml_base
import ../src/core/ggml_tensor
import ../src/core/ggml_context
import ../src/core/ggml_context_impl
import ../src/core/ggml_graph
import ../src/core/ggml_ops
import ../src/core/ggml_graph_engine

proc test_full() =
  var params: GgmlInitParams
  params.mem_size = 16 * 1024 * 1024
  params.mem_buffer = nil
  params.no_alloc = false

  let ctx = ggml_init(params)
  if ctx == nil:
    echo "ggml_init failed"
    return

  echo "ggml_init success"

  let a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  let b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  let c = ggml_add(ctx, a, b)
  let d = ggml_mul(ctx, c, a)
  let e = ggml_relu(ctx, d)

  echo "Graph built: a + b -> c; c * a -> d; relu(d) -> e"
  echo "e.op: ", e.op

  ggml_free(ctx)
  echo "ggml_free success"

test_full()
