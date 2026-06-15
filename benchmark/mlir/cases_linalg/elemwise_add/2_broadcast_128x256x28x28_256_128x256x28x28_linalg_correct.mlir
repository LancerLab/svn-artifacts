module {
  func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28_correct(%a: tensor<128x256x28x28xf32>, %b: tensor<256xf32>) -> tensor<128x256x28x28xf32> {
    %init = tensor.empty() : tensor<128x256x28x28xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%a, %b : tensor<128x256x28x28xf32>, tensor<256xf32>) outs(%init : tensor<128x256x28x28xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<128x256x28x28xf32>
    return %r : tensor<128x256x28x28xf32>
  }
}
