module {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_linalg(%input: tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xf32> {
    %init = tensor.empty() : tensor<64x256x56x56xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<64x256x56x56xf32>) outs(%init : tensor<64x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<64x256x56x56xf32>
    return %r : tensor<64x256x56x56xf32>
  }
}
