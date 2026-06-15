module {
  func.func @f_19_transformer_32x512x2048_2048_2048(%input: tensor<32x512x2048xf32>, %gamma: tensor<2048xf32>, %beta: tensor<2048xf32>) -> tensor<32x512x2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x512x2048xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x512x2048xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x512x2048xf32>
    %gamma_d0 = tensor.dim %gamma, %c0 : tensor<2048xf32>
    %beta_d0 = tensor.dim %beta, %c0 : tensor<2048xf32>
    %eq0 = arith.cmpi eq, %gamma_d0, %input_d2 : index
    cf.assert %eq0, "gamma.dim(0)==input.dim(2)"
    %eq1 = arith.cmpi eq, %beta_d0, %input_d2 : index
    cf.assert %eq1, "beta.dim(0)==input.dim(2)"
    %out = tensor.empty() : tensor<32x512x2048xf32>
    %nsz = arith.constant 2048.0 : f32
    %acc_sum_t   = tensor.empty() : tensor<f32>
    %acc_sumsq_t = tensor.empty() : tensor<f32>
    %t_res_ln_b0 = scf.for %ln_b0 = %c0 to %input_d0 step %c1 iter_args(%t_ln_ln_b0 = %out) -> (tensor<32x512x2048xf32>) {
    %t_res_ln_b1 = scf.for %ln_b1 = %c0 to %input_d1 step %c1 iter_args(%t_ln_ln_b1 = %t_ln_ln_b0) -> (tensor<32x512x2048xf32>) {
    %zero_sf = arith.constant 0.0 : f32
    %eps     = arith.constant 1.0e-05 : f32
    %ln_sum = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%ln_sa0 = %zero_sf) -> (f32) {
    %lnv1 = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0] : tensor<32x512x2048xf32>
    %ns1  = arith.addf %ln_sa0, %lnv1 : f32
    scf.yield %ns1 : f32
    }
    %ln_sumsq = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%ln_ssa0 = %zero_sf) -> (f32) {
    %lnv2 = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0] : tensor<32x512x2048xf32>
    %sq   = arith.mulf %lnv2, %lnv2 : f32
    %ssq  = arith.addf %ln_ssa0, %sq : f32
    scf.yield %ssq : f32
    }
    %mean    = arith.divf %ln_sum, %nsz : f32
    %msq     = arith.mulf %mean, %mean : f32
    %esq     = arith.divf %ln_sumsq, %nsz : f32
    %var     = arith.subf %esq, %msq : f32
    %vep     = arith.addf %var, %eps : f32
    %std     = math.sqrt %vep : f32
    %one_f   = arith.constant 1.0 : f32
    %inv_std = arith.divf %one_f, %std : f32
    %t_res_ln_n0 = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%t_ln_n0 = %t_ln_ln_b1) -> (tensor<32x512x2048xf32>) {
    %lnv3   = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0] : tensor<32x512x2048xf32>
    %gv     = tensor.extract %gamma[%ln_n0] : tensor<2048xf32>
    %bv     = tensor.extract %beta[%ln_n0]  : tensor<2048xf32>
    %cent   = arith.subf %lnv3, %mean : f32
    %normed = arith.mulf %cent, %inv_std : f32
    %scaled = arith.mulf %normed, %gv : f32
    %res    = arith.addf %scaled, %bv : f32
    %t_ins_ln = tensor.insert %res into %t_ln_n0[%ln_b0, %ln_b1, %ln_n0] : tensor<32x512x2048xf32>
    scf.yield %t_ins_ln : tensor<32x512x2048xf32>
    }
    scf.yield %t_res_ln_n0 : tensor<32x512x2048xf32>
    }
    scf.yield %t_res_ln_b1 : tensor<32x512x2048xf32>
    }
    return %t_res_ln_b0 : tensor<32x512x2048xf32>
  }
}
