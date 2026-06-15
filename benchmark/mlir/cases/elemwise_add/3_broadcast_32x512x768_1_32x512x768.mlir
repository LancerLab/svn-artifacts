module {
  func.func @f_3_broadcast_32x512x768_1_32x512x768(%lhs: tensor<32x512x768xf32>, %rhs: tensor<1xf32>) -> tensor<32x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<32x512x768xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<32x512x768xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<32x512x768xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<1xf32>
    %out = tensor.empty() : tensor<32x512x768xf32>
    %t_res_ei0 = scf.for %ei0 = %c0 to %lhs_d0 step %c1 iter_args(%t_ea_ei0 = %out) -> (tensor<32x512x768xf32>) {
    %t_res_ei1 = scf.for %ei1 = %c0 to %lhs_d1 step %c1 iter_args(%t_ea_ei1 = %t_ea_ei0) -> (tensor<32x512x768xf32>) {
    %t_res_ei2 = scf.for %ei2 = %c0 to %lhs_d2 step %c1 iter_args(%t_ea_ei2 = %t_ea_ei1) -> (tensor<32x512x768xf32>) {
    %va = tensor.extract %lhs[%ei0, %ei1, %ei2] : tensor<32x512x768xf32>
    %zidx = arith.constant 0 : index
    %vb = tensor.extract %rhs[%zidx] : tensor<1xf32>
    %rs = arith.addf %va, %vb : f32
    %t_ins = tensor.insert %rs into %t_ea_ei2[%ei0, %ei1, %ei2] : tensor<32x512x768xf32>
    scf.yield %t_ins : tensor<32x512x768xf32>
    }
    scf.yield %t_res_ei2 : tensor<32x512x768xf32>
    }
    scf.yield %t_res_ei1 : tensor<32x512x768xf32>
    }
    return %t_res_ei0 : tensor<32x512x768xf32>
  }
}
