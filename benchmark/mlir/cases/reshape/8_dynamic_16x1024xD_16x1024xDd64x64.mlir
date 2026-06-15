module {
  func.func @f_8_dynamic_16x1024xD_16x1024xDd64x64(%input: tensor<16x1024x?xf32>) -> tensor<16x1024x?x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x1024x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x1024x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x1024x?xf32>
    %out = tensor.empty(%input_d2) : tensor<16x1024x?x64xf32>
    %zf = arith.constant 0.0 : f32
    %rsz0 = arith.constant 16 : index
    %rsz1 = arith.constant 1024 : index
    %rsz3 = arith.constant 64 : index
    %t_res_rp0 = scf.for %rp0 = %c0 to %rsz0 step %c1 iter_args(%t_ph_rp0 = %out) -> (tensor<16x1024x?x64xf32>) {
    %t_res_rp1 = scf.for %rp1 = %c0 to %rsz1 step %c1 iter_args(%t_ph_rp1 = %t_ph_rp0) -> (tensor<16x1024x?x64xf32>) {
    %t_res_rp2 = scf.for %rp2 = %c0 to %input_d2 step %c1 iter_args(%t_ph_rp2 = %t_ph_rp1) -> (tensor<16x1024x?x64xf32>) {
    %t_res_rp3 = scf.for %rp3 = %c0 to %rsz3 step %c1 iter_args(%t_ph_rp3 = %t_ph_rp2) -> (tensor<16x1024x?x64xf32>) {
    %t_ins = tensor.insert %zf into %t_ph_rp3[%rp0, %rp1, %rp2, %rp3] : tensor<16x1024x?x64xf32>
    scf.yield %t_ins : tensor<16x1024x?x64xf32>
    }
    scf.yield %t_res_rp3 : tensor<16x1024x?x64xf32>
    }
    scf.yield %t_res_rp2 : tensor<16x1024x?x64xf32>
    }
    scf.yield %t_res_rp1 : tensor<16x1024x?x64xf32>
    }
    return %t_res_rp0 : tensor<16x1024x?x64xf32>
  }
}
