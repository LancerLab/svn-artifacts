module {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW(%lhs: tensor<16x512x?x?xf32>, %rhs: tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<16x512x?x?xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<16x512x?x?xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<16x512x?x?xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<16x512x?x?xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<16x512x?x?xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<16x512x?x?xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<16x512x?x?xf32>
    %rhs_d3 = tensor.dim %rhs, %c3 : tensor<16x512x?x?xf32>
    %out = tensor.empty(%lhs_d2, %lhs_d3) : tensor<16x512x?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<16x512x?x?xf32>, tensor<16x512x?x?xf32>) outs(%out : tensor<16x512x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %init: f32):
      %res = arith.addf %a, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x512x?x?xf32>
    return %result : tensor<16x512x?x?xf32>
  }
}
