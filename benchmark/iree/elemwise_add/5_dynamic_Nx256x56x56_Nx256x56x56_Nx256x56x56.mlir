module {
  func.func @f_5_dynamic_Nx256x56x56_Nx256x56x56_Nx256x56x56(%lhs: tensor<?x256x56x56xf32>, %rhs: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<?x256x56x56xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<?x256x56x56xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<?x256x56x56xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<?x256x56x56xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<?x256x56x56xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<?x256x56x56xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<?x256x56x56xf32>
    %rhs_d3 = tensor.dim %rhs, %c3 : tensor<?x256x56x56xf32>
    %out = tensor.empty(%lhs_d0) : tensor<?x256x56x56xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) outs(%out : tensor<?x256x56x56xf32>) {
    ^bb0(%a: f32, %b: f32, %init: f32):
      %res = arith.addf %a, %b : f32
      linalg.yield %res : f32
    } -> tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}
