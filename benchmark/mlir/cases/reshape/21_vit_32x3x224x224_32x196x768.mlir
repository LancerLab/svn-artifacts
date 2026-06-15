module {
  func.func @f_21_vit_32x3x224x224_32x196x768(%input: tensor<32x3x224x224xf32>) -> tensor<32x196x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x3x224x224xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x3x224x224xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x3x224x224xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x3x224x224xf32>
    %out = tensor.empty() : tensor<32x196x768xf32>
    %zf = arith.constant 0.0 : f32
    %rsz0 = arith.constant 32 : index
    %rsz1 = arith.constant 196 : index
    %rsz2 = arith.constant 768 : index
    %t_res_rp0 = scf.for %rp0 = %c0 to %rsz0 step %c1 iter_args(%t_ph_rp0 = %out) -> (tensor<32x196x768xf32>) {
    %t_res_rp1 = scf.for %rp1 = %c0 to %rsz1 step %c1 iter_args(%t_ph_rp1 = %t_ph_rp0) -> (tensor<32x196x768xf32>) {
    %t_res_rp2 = scf.for %rp2 = %c0 to %rsz2 step %c1 iter_args(%t_ph_rp2 = %t_ph_rp1) -> (tensor<32x196x768xf32>) {
    %t_ins = tensor.insert %zf into %t_ph_rp2[%rp0, %rp1, %rp2] : tensor<32x196x768xf32>
    scf.yield %t_ins : tensor<32x196x768xf32>
    }
    scf.yield %t_res_rp2 : tensor<32x196x768xf32>
    }
    scf.yield %t_res_rp1 : tensor<32x196x768xf32>
    }
    return %t_res_rp0 : tensor<32x196x768xf32>
  }
}
