module {
  func.func @f_20_vit_32x197x768_768x3072_32x197x3072(%lhs: tensor<32x197x768xf32>, %rhs: tensor<768x3072xf32>) -> tensor<32x197x3072xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<32x197x768xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<32x197x768xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<32x197x768xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<768x3072xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<768x3072xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty() : tensor<32x197x3072xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<32x197x3072xf32>) -> tensor<32x197x3072xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<32x197x768xf32>, tensor<768x3072xf32>) outs(%filled : tensor<32x197x3072xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %r = arith.addf %acc, %p : f32
      linalg.yield %r : f32
    } -> tensor<32x197x3072xf32>
    return %result : tensor<32x197x3072xf32>
  }
}
