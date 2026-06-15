module {
  func.func @f_6_dynamic_16x512xE_16x512(%input: tensor<16x512x?xf32>) -> tensor<16x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero_f = arith.constant 0.0 : f32
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x?xf32>
    %sum_buf = tensor.empty() : tensor<16x512xf32>
    %t_rm_k0 = scf.for %rm_k0 = %c0 to %input_d0 step %c1 iter_args(%t_sb_rm_k0 = %sum_buf) -> (tensor<16x512xf32>) {
    %t_rm_k1 = scf.for %rm_k1 = %c0 to %input_d1 step %c1 iter_args(%t_sb_rm_k1 = %t_sb_rm_k0) -> (tensor<16x512xf32>) {
    %red_val = scf.for %rm_r0 = %c0 to %input_d2 step %c1 iter_args(%ia0 = %zero_f) -> (f32) {
    %rv = tensor.extract %input[%rm_k0, %rm_k1, %rm_r0] : tensor<16x512x?xf32>
    %ns = arith.addf %ia0, %rv : f32
    scf.yield %ns : f32
    }
    %t_sb_new = tensor.insert %red_val into %t_sb_rm_k1[%rm_k0, %rm_k1] : tensor<16x512xf32>
    scf.yield %t_sb_new : tensor<16x512xf32>
    }
    scf.yield %t_rm_k1 : tensor<16x512xf32>
    }
    %scale = arith.constant 1.0 : f32
    %out = tensor.empty() : tensor<16x512xf32>
    %t_rm_sc0 = scf.for %sc0 = %c0 to %input_d0 step %c1 iter_args(%t_sc_rm_k0 = %out) -> (tensor<16x512xf32>) {
    %t_rm_sc1 = scf.for %sc1 = %c0 to %input_d1 step %c1 iter_args(%t_sc_rm_k1 = %t_sc_rm_k0) -> (tensor<16x512xf32>) {
    %sv = tensor.extract %t_rm_k0[%sc0, %sc1] : tensor<16x512xf32>
    %me = arith.mulf %sv, %scale : f32
    %t_sc_ins = tensor.insert %me into %t_sc_rm_k1[%sc0, %sc1] : tensor<16x512xf32>
    scf.yield %t_sc_ins : tensor<16x512xf32>
    }
    scf.yield %t_rm_sc1 : tensor<16x512xf32>
    }
    return %t_rm_sc0 : tensor<16x512xf32>
  }
}
