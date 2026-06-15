#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {
  func.func @f_11_dynamic_32xSx768_768x768_32xSx768_mismatch(%a: tensor<32x?x768xf32>, %b: tensor<767x768xf32>) -> tensor<32x?x768xf32> {
    %cst = arith.constant 0.0 : f32
    %c_i0 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c_i0 : tensor<32x?x768xf32>
    %init = tensor.empty(%d1) : tensor<32x?x768xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x?x768xf32>) -> tensor<32x?x768xf32>
    %r = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<32x?x768xf32>, tensor<767x768xf32>)
      outs(%fill : tensor<32x?x768xf32>) {
    ^bb0(%in_a: f32, %in_b: f32, %acc: f32):
      %mul = arith.mulf %in_a, %in_b : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<32x?x768xf32>
    return %r : tensor<32x?x768xf32>
  }
}
