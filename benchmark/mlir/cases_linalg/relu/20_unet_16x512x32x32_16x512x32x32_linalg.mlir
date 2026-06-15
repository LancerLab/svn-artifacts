module {
  func.func @f_20_unet_16x512x32x32_16x512x32x32_linalg(%input: tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %init = tensor.empty() : tensor<16x512x32x32xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<16x512x32x32xf32>) outs(%init : tensor<16x512x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<16x512x32x32xf32>
    return %r : tensor<16x512x32x32xf32>
  }
}
