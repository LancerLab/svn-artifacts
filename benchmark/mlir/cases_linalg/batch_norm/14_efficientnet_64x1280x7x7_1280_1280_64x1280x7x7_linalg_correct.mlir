module {
  func.func @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_correct(%input: tensor<64x1280x7x7xf32>, %gamma: tensor<1280xf32>, %beta: tensor<1280xf32>) -> tensor<64x1280x7x7xf32> {
    %init = tensor.empty() : tensor<64x1280x7x7xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<64x1280x7x7xf32>, tensor<1280xf32>, tensor<1280xf32>) outs(%init : tensor<64x1280x7x7xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x1280x7x7xf32>
    return %r : tensor<64x1280x7x7xf32>
  }
}
