module {
  func.func @f_4_dynamic_32xNx512x64_32xNx64x512_32xNx512x512(%lhs: tensor<32x?x512x64xf32>, %rhs: tensor<32x?x64x512xf32>) -> tensor<32x?x512x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<32x?x512x64xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<32x?x512x64xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<32x?x512x64xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<32x?x512x64xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<32x?x64x512xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<32x?x64x512xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<32x?x64x512xf32>
    %rhs_d3 = tensor.dim %rhs, %c3 : tensor<32x?x64x512xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty(%lhs_d1) : tensor<32x?x512x512xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<32x?x512x512xf32>) -> tensor<32x?x512x512xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<32x?x512x64xf32>, tensor<32x?x64x512xf32>) outs(%filled : tensor<32x?x512x512xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<32x?x512x512xf32>
    return %result : tensor<32x?x512x512xf32>
  }
}
