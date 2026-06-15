#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {
  func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_mismatch(%a: tensor<16x1024x1536xf32>, %b: tensor<1535x6144xf32>) -> tensor<16x1024x6144xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16x1024x6144xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x1024x6144xf32>) -> tensor<16x1024x6144xf32>
    %r = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<16x1024x1536xf32>, tensor<1535x6144xf32>)
      outs(%fill : tensor<16x1024x6144xf32>) {
    ^bb0(%in_a: f32, %in_b: f32, %acc: f32):
      %mul = arith.mulf %in_a, %in_b : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<16x1024x6144xf32>
    return %r : tensor<16x1024x6144xf32>
  }
}
