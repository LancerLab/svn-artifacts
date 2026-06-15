module {
  func.func @f_3_dynamic_Nx256x56x56_Nx256_linalg(%input: tensor<?x256x56x56xf32>) -> tensor<?x256xf32> {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %od0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%od0) : tensor<?x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x256xf32>) -> tensor<?x256xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]
    } ins(%input : tensor<?x256x56x56xf32>) outs(%fill : tensor<?x256xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<?x256xf32>
    return %r : tensor<?x256xf32>
  }
}
