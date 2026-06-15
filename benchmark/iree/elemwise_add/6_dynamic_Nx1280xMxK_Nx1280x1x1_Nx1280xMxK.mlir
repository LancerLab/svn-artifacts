module {
  func.func @f_6_dynamic_Nx1280xMxK_Nx1280x1x1_Nx1280xMxK(%lhs: tensor<?x1280x?x?xf32>, %rhs: tensor<?x1280x1x1xf32>) -> tensor<?x1280x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<?x1280x?x?xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<?x1280x?x?xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<?x1280x?x?xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<?x1280x?x?xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<?x1280x1x1xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<?x1280x1x1xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<?x1280x1x1xf32>
    %rhs_d3 = tensor.dim %rhs, %c3 : tensor<?x1280x1x1xf32>
    %out = tensor.empty(%lhs_d0, %lhs_d2, %lhs_d3) : tensor<?x1280x?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x1280x?x?xf32>, tensor<?x1280x1x1xf32>) outs(%out : tensor<?x1280x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %init: f32):
      %res = arith.addf %a, %b : f32
      linalg.yield %res : f32
    } -> tensor<?x1280x?x?xf32>
    return %result : tensor<?x1280x?x?xf32>
  }
}
