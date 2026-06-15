module {
  func.func @f_4_dynamic_NxCxHxW_N_linalg(%input: tensor<?x?x?x?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %od0 = tensor.dim %input, %c0 : tensor<?x?x?x?xf32>
    %init = tensor.empty(%od0) : tensor<?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?xf32>) -> tensor<?xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]
    } ins(%input : tensor<?x?x?x?xf32>) outs(%fill : tensor<?xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
}
