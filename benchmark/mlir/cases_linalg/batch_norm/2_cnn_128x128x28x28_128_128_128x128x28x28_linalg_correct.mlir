module {
  func.func @f_2_cnn_128x128x28x28_128_128_128x128x28x28_correct(%input: tensor<128x128x28x28xf32>, %gamma: tensor<128xf32>, %beta: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %init = tensor.empty() : tensor<128x128x28x28xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<128x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<128x128x28x28xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<128x128x28x28xf32>
    return %r : tensor<128x128x28x28xf32>
  }
}
