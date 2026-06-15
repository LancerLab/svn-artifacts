module {
  func.func @f_4_dynamic_Bx256x56x56_Bx256x56x56_linalg(%input: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%d0) : tensor<?x256x56x56xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<?x256x56x56xf32>) outs(%init : tensor<?x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<?x256x56x56xf32>
    return %r : tensor<?x256x56x56xf32>
  }
}
