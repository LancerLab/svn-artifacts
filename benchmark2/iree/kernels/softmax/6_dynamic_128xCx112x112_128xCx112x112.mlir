module {
  func.func @f_6_dynamic_128xCx112x112_128xCx112x112(%input: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x?x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x?x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x?x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x?x112x112xf32>
    %out = tensor.empty(%input_d1) : tensor<128x?x112x112xf32>
    %max_t = tensor.empty(%input_d1) : tensor<128x?x112xf32>
    %neg_inf = arith.constant -3.4028234663852886e+38 : f32
    %max_init = linalg.fill ins(%neg_inf : f32) outs(%max_t : tensor<128x?x112xf32>) -> tensor<128x?x112xf32>
    %max_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%input : tensor<128x?x112x112xf32>) outs(%max_init : tensor<128x?x112xf32>) {
    ^bb0(%in: f32, %cur: f32):
      %mx = arith.maximumf %in, %cur : f32
      linalg.yield %mx : f32
    } -> tensor<128x?x112xf32>
    %sum_t = tensor.empty(%input_d1) : tensor<128x?x112xf32>
    %zero_f = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<128x?x112xf32>) -> tensor<128x?x112xf32>
    %sum_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%input, %max_result : tensor<128x?x112x112xf32>, tensor<128x?x112xf32>) outs(%sum_init : tensor<128x?x112xf32>) {
    ^bb0(%in: f32, %mx: f32, %acc: f32):
      %sh  = arith.subf %in, %mx : f32
      %ex  = math.exp %sh : f32
      %nac = arith.addf %acc, %ex : f32
      linalg.yield %nac : f32
    } -> tensor<128x?x112xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %max_result, %sum_result : tensor<128x?x112x112xf32>, tensor<128x?x112xf32>, tensor<128x?x112xf32>) outs(%out : tensor<128x?x112x112xf32>) {
    ^bb0(%in: f32, %mx: f32, %sm: f32, %init: f32):
      %sh  = arith.subf %in, %mx : f32
      %ex  = math.exp %sh : f32
      %res = arith.divf %ex, %sm : f32
      linalg.yield %res : f32
    } -> tensor<128x?x112x112xf32>
    return %result : tensor<128x?x112x112xf32>
  }
}
