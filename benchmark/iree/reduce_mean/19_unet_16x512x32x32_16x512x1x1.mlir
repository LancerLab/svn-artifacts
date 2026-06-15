module {
  func.func @f_19_unet_16x512x32x32_16x512x1x1(%input: tensor<16x512x32x32xf32>) -> tensor<16x512x1x1xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x32x32xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x32x32xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x32x32xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x512x32x32xf32>
    %sum_t = tensor.empty() : tensor<16x512xf32>
    %zero_f = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<16x512xf32>) -> tensor<16x512xf32>
    %sum_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%input : tensor<16x512x32x32xf32>) outs(%sum_init : tensor<16x512xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ns = arith.addf %acc, %in : f32
      linalg.yield %ns : f32
    } -> tensor<16x512xf32>
    %scale = arith.constant 0.0009765625 : f32
    %div_t = tensor.empty() : tensor<16x512xf32>
    %div_init = linalg.fill ins(%zero_f : f32) outs(%div_t : tensor<16x512xf32>) -> tensor<16x512xf32>
    %mean_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum_result : tensor<16x512xf32>) outs(%div_init : tensor<16x512xf32>) {
    ^bb0(%sv: f32, %init: f32):
      %res = arith.mulf %sv, %scale : f32
      linalg.yield %res : f32
    } -> tensor<16x512xf32>
    %result = tensor.expand_shape %mean_result [[0], [1, 2, 3]] output_shape [16, 512, 1, 1] : tensor<16x512xf32> into tensor<16x512x1x1xf32>
    return %result : tensor<16x512x1x1xf32>
  }
}
