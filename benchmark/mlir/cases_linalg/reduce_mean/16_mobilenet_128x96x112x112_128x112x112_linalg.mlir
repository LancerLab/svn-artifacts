module {
  func.func @f_16_mobilenet_128x96x112x112_128x112x112_linalg(%input: tensor<128x96x112x112xf32>) -> tensor<128x112x112xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x112x112xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x112x112xf32>) -> tensor<128x112x112xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]
    } ins(%input : tensor<128x96x112x112xf32>) outs(%fill : tensor<128x112x112xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<128x112x112xf32>
    return %r : tensor<128x112x112xf32>
  }
}
