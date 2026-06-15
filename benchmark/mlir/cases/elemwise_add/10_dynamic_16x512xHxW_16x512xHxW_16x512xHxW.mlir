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
    %eq0 = arith.cmpi eq, %lhs_d0, %rhs_d0 : index
    cf.assert %eq0, "lhs.dim(0)==rhs.dim(0)"
    %eq1 = arith.cmpi eq, %lhs_d1, %rhs_d1 : index
    cf.assert %eq1, "lhs.dim(1)==rhs.dim(1)"
    %eq2 = arith.cmpi eq, %lhs_d2, %rhs_d2 : index
    cf.assert %eq2, "lhs.dim(2)==rhs.dim(2)"
    %eq3 = arith.cmpi eq, %lhs_d3, %rhs_d3 : index
    cf.assert %eq3, "lhs.dim(3)==rhs.dim(3)"
    %out = tensor.empty(%lhs_d2, %lhs_d3) : tensor<16x512x?x?xf32>
    %t_res_ei0 = scf.for %ei0 = %c0 to %lhs_d0 step %c1 iter_args(%t_ea_ei0 = %out) -> (tensor<16x512x?x?xf32>) {
    %t_res_ei1 = scf.for %ei1 = %c0 to %lhs_d1 step %c1 iter_args(%t_ea_ei1 = %t_ea_ei0) -> (tensor<16x512x?x?xf32>) {
    %t_res_ei2 = scf.for %ei2 = %c0 to %lhs_d2 step %c1 iter_args(%t_ea_ei2 = %t_ea_ei1) -> (tensor<16x512x?x?xf32>) {
    %t_res_ei3 = scf.for %ei3 = %c0 to %lhs_d3 step %c1 iter_args(%t_ea_ei3 = %t_ea_ei2) -> (tensor<16x512x?x?xf32>) {
    %va = tensor.extract %lhs[%ei0, %ei1, %ei2, %ei3] : tensor<16x512x?x?xf32>
    %vb = tensor.extract %rhs[%ei0, %ei1, %ei2, %ei3] : tensor<16x512x?x?xf32>
    %rs = arith.addf %va, %vb : f32
    %t_ins = tensor.insert %rs into %t_ea_ei3[%ei0, %ei1, %ei2, %ei3] : tensor<16x512x?x?xf32>
    scf.yield %t_ins : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ei3 : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ei2 : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ei1 : tensor<16x512x?x?xf32>
    }
    return %t_res_ei0 : tensor<16x512x?x?xf32>
  }
}
