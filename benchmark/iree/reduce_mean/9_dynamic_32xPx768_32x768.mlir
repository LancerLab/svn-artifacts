module {
  func.func @f_9_dynamic_32xPx768_32x768(%input: tensor<32x?x768xf32>) -> tensor<32x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x?x768xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x?x768xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x?x768xf32>
    %sum_t = tensor.empty() : tensor<32x768xf32>
    %zero_f = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<32x768xf32>) -> tensor<32x768xf32>
    %sum_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : tensor<32x?x768xf32>) outs(%sum_init : tensor<32x768xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ns = arith.addf %acc, %in : f32
      linalg.yield %ns : f32
    } -> tensor<32x768xf32>
    %rsz_idx_0 = arith.constant 1 : index
    %rsz_idx_1 = arith.muli %rsz_idx_0, %input_d1 : index
    %rsz_i64 = arith.index_cast %rsz_idx_1 : index to i64
    %rsz    = arith.sitofp %rsz_i64 : i64 to f32
    %scale  = arith.constant 1.0 : f32
    %div_t = tensor.empty() : tensor<32x768xf32>
    %div_init = linalg.fill ins(%zero_f : f32) outs(%div_t : tensor<32x768xf32>) -> tensor<32x768xf32>
    %mean_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum_result : tensor<32x768xf32>) outs(%div_init : tensor<32x768xf32>) {
    ^bb0(%sv: f32, %init: f32):
      %res = arith.divf %sv, %rsz : f32
      linalg.yield %res : f32
    } -> tensor<32x768xf32>
    return %mean_result : tensor<32x768xf32>
  }
}
