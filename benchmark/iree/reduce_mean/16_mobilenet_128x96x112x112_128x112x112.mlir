module {
  func.func @f_16_mobilenet_128x96x112x112_128x112x112(%input: tensor<128x96x112x112xf32>) -> tensor<128x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x96x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x96x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x96x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x96x112x112xf32>
    %sum_t = tensor.empty() : tensor<128x112x112xf32>
    %zero_f = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<128x112x112xf32>) -> tensor<128x112x112xf32>
    %sum_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]
    } ins(%input : tensor<128x96x112x112xf32>) outs(%sum_init : tensor<128x112x112xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ns = arith.addf %acc, %in : f32
      linalg.yield %ns : f32
    } -> tensor<128x112x112xf32>
    %scale = arith.constant 0.010416666666666666 : f32
    %div_t = tensor.empty() : tensor<128x112x112xf32>
    %div_init = linalg.fill ins(%zero_f : f32) outs(%div_t : tensor<128x112x112xf32>) -> tensor<128x112x112xf32>
    %mean_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%sum_result : tensor<128x112x112xf32>) outs(%div_init : tensor<128x112x112xf32>) {
    ^bb0(%sv: f32, %init: f32):
      %res = arith.mulf %sv, %scale : f32
      linalg.yield %res : f32
    } -> tensor<128x112x112xf32>
    return %mean_result : tensor<128x112x112xf32>
  }
}
