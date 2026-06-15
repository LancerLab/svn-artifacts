module {
  func.func @f_2_cnn_128x128x28x28_128(%input: tensor<128x128x28x28xf32>) -> tensor<128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %zero_f = arith.constant 0.0 : f32
    %input_d0 = tensor.dim %input, %c0 : tensor<128x128x28x28xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x128x28x28xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x128x28x28xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x128x28x28xf32>
    %sum_buf = tensor.empty() : tensor<128xf32>
    %t_rm_k0 = scf.for %rm_k0 = %c0 to %input_d0 step %c1 iter_args(%t_sb_rm_k0 = %sum_buf) -> (tensor<128xf32>) {
    %red_val = scf.for %rm_r0 = %c0 to %input_d1 step %c1 iter_args(%ia0 = %zero_f) -> (f32) {
    %red_val_d1 = scf.for %rm_r1 = %c0 to %input_d2 step %c1 iter_args(%ia1 = %ia0) -> (f32) {
    %red_val_d2 = scf.for %rm_r2 = %c0 to %input_d3 step %c1 iter_args(%ia2 = %ia1) -> (f32) {
    %rv = tensor.extract %input[%rm_k0, %rm_r0, %rm_r1, %rm_r2] : tensor<128x128x28x28xf32>
    %ns = arith.addf %ia2, %rv : f32
    scf.yield %ns : f32
    }
    scf.yield %red_val_d2 : f32
    }
    scf.yield %red_val_d1 : f32
    }
    %t_sb_new = tensor.insert %red_val into %t_sb_rm_k0[%rm_k0] : tensor<128xf32>
    scf.yield %t_sb_new : tensor<128xf32>
    }
    %scale = arith.constant 9.964923469387754e-06 : f32
    %out = tensor.empty() : tensor<128xf32>
    %t_rm_sc0 = scf.for %sc0 = %c0 to %input_d0 step %c1 iter_args(%t_sc_rm_k0 = %out) -> (tensor<128xf32>) {
    %sv = tensor.extract %t_rm_k0[%sc0] : tensor<128xf32>
    %me = arith.mulf %sv, %scale : f32
    %t_sc_ins = tensor.insert %me into %t_sc_rm_k0[%sc0] : tensor<128xf32>
    scf.yield %t_sc_ins : tensor<128xf32>
    }
    return %t_rm_sc0 : tensor<128xf32>
  }
}
