module {
  func.func @f_13_dynamic_32x512xV_32x512xV_32x512xV(%lhs: tensor<32x512x?xf32>, %rhs: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<32x512x?xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<32x512x?xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<32x512x?xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<32x512x?xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<32x512x?xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<32x512x?xf32>
    %eq0 = arith.cmpi eq, %lhs_d0, %rhs_d0 : index
    cf.assert %eq0, "lhs.dim(0)==rhs.dim(0)"
    %eq1 = arith.cmpi eq, %lhs_d1, %rhs_d1 : index
    cf.assert %eq1, "lhs.dim(1)==rhs.dim(1)"
    %eq2 = arith.cmpi eq, %lhs_d2, %rhs_d2 : index
    cf.assert %eq2, "lhs.dim(2)==rhs.dim(2)"
    %out = tensor.empty(%lhs_d2) : tensor<32x512x?xf32>
    %t_res_ei0 = scf.for %ei0 = %c0 to %lhs_d0 step %c1 iter_args(%t_ea_ei0 = %out) -> (tensor<32x512x?xf32>) {
    %t_res_ei1 = scf.for %ei1 = %c0 to %lhs_d1 step %c1 iter_args(%t_ea_ei1 = %t_ea_ei0) -> (tensor<32x512x?xf32>) {
    %t_res_ei2 = scf.for %ei2 = %c0 to %lhs_d2 step %c1 iter_args(%t_ea_ei2 = %t_ea_ei1) -> (tensor<32x512x?xf32>) {
    %va = tensor.extract %lhs[%ei0, %ei1, %ei2] : tensor<32x512x?xf32>
    %vb = tensor.extract %rhs[%ei0, %ei1, %ei2] : tensor<32x512x?xf32>
    %rs = arith.addf %va, %vb : f32
    %t_ins = tensor.insert %rs into %t_ea_ei2[%ei0, %ei1, %ei2] : tensor<32x512x?xf32>
    scf.yield %t_ins : tensor<32x512x?xf32>
    }
    scf.yield %t_res_ei2 : tensor<32x512x?xf32>
    }
    scf.yield %t_res_ei1 : tensor<32x512x?xf32>
    }
    return %t_res_ei0 : tensor<32x512x?xf32>
  }
}
