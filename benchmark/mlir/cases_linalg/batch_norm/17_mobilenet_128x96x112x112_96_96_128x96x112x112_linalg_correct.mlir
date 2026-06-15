module {
  func.func @f_17_mobilenet_128x96x112x112_96_96_128x96x112x112_correct(%input: tensor<128x96x112x112xf32>, %gamma: tensor<96xf32>, %beta: tensor<96xf32>) -> tensor<128x96x112x112xf32> {
    %init = tensor.empty() : tensor<128x96x112x112xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<128x96x112x112xf32>, tensor<96xf32>, tensor<96xf32>) outs(%init : tensor<128x96x112x112xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<128x96x112x112xf32>
    return %r : tensor<128x96x112x112xf32>
  }
}
