module {
  func.func @f_4_dynamic_Nx256x56x56_56x56_56x56_correct(%input: tensor<?x256x56x56xf32>, %gamma: tensor<56x56xf32>, %beta: tensor<56x56xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%d0) : tensor<?x256x56x56xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<?x256x56x56xf32>, tensor<56x56xf32>, tensor<56x56xf32>) outs(%init : tensor<?x256x56x56xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<?x256x56x56xf32>
    return %r : tensor<?x256x56x56xf32>
  }
}
