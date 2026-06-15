module {
  func.func @f_7_dynamic_64x128xHxW_64x128(%input: tensor<64x128x?x?xf32>) -> tensor<64x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x128x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x128x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x128x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<64x128x?x?xf32>
    %sum_t = tensor.empty() : tensor<64x128xf32>
    %zero_f = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<64x128xf32>) -> tensor<64x128xf32>
    %sum_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%input : tensor<64x128x?x?xf32>) outs(%sum_init : tensor<64x128xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ns = arith.addf %acc, %in : f32
      linalg.yield %ns : f32
    } -> tensor<64x128xf32>
    %rsz_idx_0 = arith.constant 1 : index
    %rsz_idx_1 = arith.muli %rsz_idx_0, %input_d2 : index
    %rsz_idx_2 = arith.muli %rsz_idx_1, %input_d3 : index
    %rsz_i64 = arith.index_cast %rsz_idx_2 : index to i64
    %rsz    = arith.sitofp %rsz_i64 : i64 to f32
    %scale  = arith.constant 1.0 : f32
    %div_t = tensor.empty() : tensor<64x128xf32>
    %div_init = linalg.fill ins(%zero_f : f32) outs(%div_t : tensor<64x128xf32>) -> tensor<64x128xf32>
    %mean_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum_result : tensor<64x128xf32>) outs(%div_init : tensor<64x128xf32>) {
    ^bb0(%sv: f32, %init: f32):
      %res = arith.divf %sv, %rsz : f32
      linalg.yield %res : f32
    } -> tensor<64x128xf32>
    return %mean_result : tensor<64x128xf32>
  }
}
