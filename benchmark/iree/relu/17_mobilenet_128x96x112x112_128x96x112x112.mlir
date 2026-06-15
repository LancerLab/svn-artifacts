module {
  func.func @f_17_mobilenet_128x96x112x112_128x96x112x112(%input: tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x96x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x96x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x96x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x96x112x112xf32>
    %out = tensor.empty() : tensor<128x96x112x112xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<128x96x112x112xf32>) outs(%out : tensor<128x96x112x112xf32>) {
    ^bb0(%in: f32, %init: f32):
      %zero = arith.constant 0.0 : f32
      %res = arith.maximumf %in, %zero : f32
      linalg.yield %res : f32
    } -> tensor<128x96x112x112xf32>
    return %result : tensor<128x96x112x112xf32>
  }
}
