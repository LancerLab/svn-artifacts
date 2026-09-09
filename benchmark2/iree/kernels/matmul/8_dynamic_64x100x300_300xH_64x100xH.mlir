module {
  func.func @f_8_dynamic_64x100x300_300xH_64x100xH(%lhs: tensor<64x100x300xf32>, %rhs: tensor<300x?xf32>) -> tensor<64x100x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<64x100x300xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<64x100x300xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<64x100x300xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<300x?xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<300x?xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty(%rhs_d1) : tensor<64x100x?xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<64x100x?xf32>) -> tensor<64x100x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<64x100x300xf32>, tensor<300x?xf32>) outs(%filled : tensor<64x100x?xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<64x100x?xf32>
    return %result : tensor<64x100x?xf32>
  }
}
