module {
  func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28(%lhs: tensor<128x256x28x28xf32>, %rhs: tensor<256xf32>) -> tensor<128x256x28x28xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<128x256x28x28xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<128x256x28x28xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<128x256x28x28xf32>
    %lhs_d3 = tensor.dim %lhs, %c3 : tensor<128x256x28x28xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<256xf32>
    %out = tensor.empty() : tensor<128x256x28x28xf32>
    %eq0 = arith.cmpi eq, %rhs_d0, %lhs_d1 : index
    cf.assert %eq0, "rhs.dim(0)==lhs.dim(1)"
    %t_res_ei0 = scf.for %ei0 = %c0 to %lhs_d0 step %c1 iter_args(%t_ea_ei0 = %out) -> (tensor<128x256x28x28xf32>) {
    %t_res_ei1 = scf.for %ei1 = %c0 to %lhs_d1 step %c1 iter_args(%t_ea_ei1 = %t_ea_ei0) -> (tensor<128x256x28x28xf32>) {
    %t_res_ei2 = scf.for %ei2 = %c0 to %lhs_d2 step %c1 iter_args(%t_ea_ei2 = %t_ea_ei1) -> (tensor<128x256x28x28xf32>) {
    %t_res_ei3 = scf.for %ei3 = %c0 to %lhs_d3 step %c1 iter_args(%t_ea_ei3 = %t_ea_ei2) -> (tensor<128x256x28x28xf32>) {
    %va = tensor.extract %lhs[%ei0, %ei1, %ei2, %ei3] : tensor<128x256x28x28xf32>
    %vb = tensor.extract %rhs[%ei1] : tensor<256xf32>
    %rs = arith.addf %va, %vb : f32
    %t_ins = tensor.insert %rs into %t_ea_ei3[%ei0, %ei1, %ei2, %ei3] : tensor<128x256x28x28xf32>
    scf.yield %t_ins : tensor<128x256x28x28xf32>
    }
    scf.yield %t_res_ei3 : tensor<128x256x28x28xf32>
    }
    scf.yield %t_res_ei2 : tensor<128x256x28x28xf32>
    }
    scf.yield %t_res_ei1 : tensor<128x256x28x28xf32>
    }
    return %t_res_ei0 : tensor<128x256x28x28xf32>
  }
}
