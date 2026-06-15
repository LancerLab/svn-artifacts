#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {
  func.func @f_9_dynamic_16xSx512_512x256_16xSx256_correct(%a: tensor<16x?x512xf32>, %b: tensor<512x256xf32>) -> tensor<16x?x256xf32> {
    %cst = arith.constant 0.0 : f32
    %c_i0 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c_i0 : tensor<16x?x512xf32>
    %init = tensor.empty(%d1) : tensor<16x?x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x?x256xf32>) -> tensor<16x?x256xf32>
    %r = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<16x?x512xf32>, tensor<512x256xf32>)
      outs(%fill : tensor<16x?x256xf32>) {
    ^bb0(%in_a: f32, %in_b: f32, %acc: f32):
      %mul = arith.mulf %in_a, %in_b : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<16x?x256xf32>
    return %r : tensor<16x?x256xf32>
  }
}
