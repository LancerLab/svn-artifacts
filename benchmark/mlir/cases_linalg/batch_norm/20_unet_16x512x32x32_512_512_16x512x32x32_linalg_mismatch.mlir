module {
  func.func @f_20_unet_16x512x32x32_512_512_16x512x32x32_mismatch(%input: tensor<16x512x32x32xf32>, %gamma: tensor<511xf32>, %beta: tensor<512xf32>) -> tensor<16x512x32x32xf32> {
    %init = tensor.empty() : tensor<16x512x32x32xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<16x512x32x32xf32>, tensor<511xf32>, tensor<512xf32>) outs(%init : tensor<16x512x32x32xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x512x32x32xf32>
    return %r : tensor<16x512x32x32xf32>
  }
}
