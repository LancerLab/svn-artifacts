module {
  func.func @f_4_dynamic_32xNx512x64_32xNx64x512_32xNx512x512(%lhs: tensor<32x?x512x64xf32>, %rhs: tensor<32x?x64x512xf32>) -> tensor<32x?x512x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %zero_f = arith.constant 0.0 : f32
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<32x?x512x64xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<32x?x512x64xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<32x?x512x64xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<32x?x512x64xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<32x?x64x512xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<32x?x64x512xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<32x?x64x512xf32>
    %rhs_d3 = tensor.dim %rhs, %c3 : tensor<32x?x64x512xf32>
    %eq0 = arith.cmpi eq, %lhs_d0, %rhs_d0 : index
    cf.assert %eq0, "lhs.dim(0)==rhs.dim(0)"
    %eq1 = arith.cmpi eq, %lhs_d1, %rhs_d1 : index
    cf.assert %eq1, "lhs.dim(1)==rhs.dim(1)"
    %eq2 = arith.cmpi eq, %lhs_d3, %rhs_d2 : index
    cf.assert %eq2, "lhs.dim(3)==rhs.dim(2)"
    %out = tensor.empty(%lhs_d1) : tensor<32x?x512x512xf32>
    %t_res_b0 = scf.for %b0 = %c0 to %lhs_d0 step %c1 iter_args(%t_arg_b0 = %out) -> (tensor<32x?x512x512xf32>) {
    %t_res_b1 = scf.for %b1 = %c0 to %lhs_d1 step %c1 iter_args(%t_arg_b1 = %t_arg_b0) -> (tensor<32x?x512x512xf32>) {
    %t_res_i = scf.for %i = %c0 to %lhs_d2 step %c1 iter_args(%t_arg_i = %t_arg_b1) -> (tensor<32x?x512x512xf32>) {
    %t_res_j = scf.for %j = %c0 to %rhs_d3 step %c1 iter_args(%t_arg_j = %t_arg_i) -> (tensor<32x?x512x512xf32>) {
    %acc = scf.for %k = %c0 to %lhs_d3 step %c1 iter_args(%s = %zero_f) -> (f32) {
    %a   = tensor.extract %lhs[%b0, %b1, %i, %k] : tensor<32x?x512x64xf32>
    %b   = tensor.extract %rhs[%b0, %b1, %k, %j] : tensor<32x?x64x512xf32>
    %p   = arith.mulf %a, %b : f32
    %ns  = arith.addf %s, %p : f32
    scf.yield %ns : f32
    }
    %t_new = tensor.insert %acc into %t_arg_j[%b0, %b1, %i, %j] : tensor<32x?x512x512xf32>
    scf.yield %t_new : tensor<32x?x512x512xf32>
    }
    scf.yield %t_res_j : tensor<32x?x512x512xf32>
    }
    scf.yield %t_res_i : tensor<32x?x512x512xf32>
    }
    scf.yield %t_res_b1 : tensor<32x?x512x512xf32>
    }
    return %t_res_b0 : tensor<32x?x512x512xf32>
  }
}
