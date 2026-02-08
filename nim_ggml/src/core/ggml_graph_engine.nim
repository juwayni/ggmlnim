# ggml_graph_engine.nim

import ggml_base, ggml_tensor, ggml_context, ggml_graph

proc ggml_visit_parents*(cgraph: ptr GgmlCGraph, tensor: ptr GgmlTensor) =
  if tensor == nil: return

  if not ggml_hash_set_insert(cgraph.visited_hash_set, tensor):
    return

  for i in 0..<GGML_MAX_SRC:
    if tensor.src[i] != nil:
      ggml_visit_parents(cgraph, tensor.src[i])

  let nodes = cast[ptr UncheckedArray[ptr GgmlTensor]](cgraph.nodes)
  nodes[cgraph.n_nodes] = tensor
  cgraph.n_nodes += 1

proc ggml_build_forward_expand*(cgraph: ptr GgmlCGraph, tensor: ptr GgmlTensor) =
  ggml_visit_parents(cgraph, tensor)

proc ggml_graph_compute*(cgraph: ptr GgmlCGraph, n_threads: cint) =
  # Actual execution logic
  for i in 0..<cgraph.n_nodes:
    let node = (cast[ptr UncheckedArray[ptr GgmlTensor]](cgraph.nodes))[i]
    if node.op == GGML_OP_NONE: continue

    # Dispatch to CPU backend for now
    # ggml_compute_forward(node, n_threads)
    discard
