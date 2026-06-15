module {
  func.func @f_2_cnn_128x128x28x28_28x28_28x28(%input: tensor<128x128x28x28xf32>, %gamma: tensor<28x28xf32>, %beta: tensor<28x28xf32>) -> tensor<128x128x28x28xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x128x28x28xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x128x28x28xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x128x28x28xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x128x28x28xf32>
    %gamma_d0 = tensor.dim %gamma, %c0 : tensor<28x28xf32>
    %gamma_d1 = tensor.dim %gamma, %c1 : tensor<28x28xf32>
    %beta_d0 = tensor.dim %beta, %c0 : tensor<28x28xf32>
    %beta_d1 = tensor.dim %beta, %c1 : tensor<28x28xf32>
    %eq0 = arith.cmpi eq, %gamma_d0, %input_d2 : index
    cf.assert %eq0, "gamma.dim(0)==input.dim(2)"
    %eq1 = arith.cmpi eq, %gamma_d1, %input_d3 : index
    cf.assert %eq1, "gamma.dim(1)==input.dim(3)"
    %eq2 = arith.cmpi eq, %beta_d0, %input_d2 : index
    cf.assert %eq2, "beta.dim(0)==input.dim(2)"
    %eq3 = arith.cmpi eq, %beta_d1, %input_d3 : index
    cf.assert %eq3, "beta.dim(1)==input.dim(3)"
    %out = tensor.empty() : tensor<128x128x28x28xf32>
    %nsz = arith.constant 784.0 : f32
    %acc_sum_t   = tensor.empty() : tensor<f32>
    %acc_sumsq_t = tensor.empty() : tensor<f32>
    %t_res_ln_b0 = scf.for %ln_b0 = %c0 to %input_d0 step %c1 iter_args(%t_ln_ln_b0 = %out) -> (tensor<128x128x28x28xf32>) {
    %t_res_ln_b1 = scf.for %ln_b1 = %c0 to %input_d1 step %c1 iter_args(%t_ln_ln_b1 = %t_ln_ln_b0) -> (tensor<128x128x28x28xf32>) {
    %zero_sf = arith.constant 0.0 : f32
    %eps     = arith.constant 1.0e-05 : f32
    %ln_sum = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%ln_sa0 = %zero_sf) -> (f32) {
    %ln_sum_d1 = scf.for %ln_n1 = %c0 to %input_d3 step %c1 iter_args(%ln_sa1 = %ln_sa0) -> (f32) {
    %lnv1 = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0, %ln_n1] : tensor<128x128x28x28xf32>
    %ns1  = arith.addf %ln_sa1, %lnv1 : f32
    scf.yield %ns1 : f32
    }
    scf.yield %ln_sum_d1 : f32
    }
    %ln_sumsq = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%ln_ssa0 = %zero_sf) -> (f32) {
    %ln_sumsq_d1 = scf.for %ln_n1 = %c0 to %input_d3 step %c1 iter_args(%ln_ssa1 = %ln_ssa0) -> (f32) {
    %lnv2 = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0, %ln_n1] : tensor<128x128x28x28xf32>
    %sq   = arith.mulf %lnv2, %lnv2 : f32
    %ssq  = arith.addf %ln_ssa1, %sq : f32
    scf.yield %ssq : f32
    }
    scf.yield %ln_sumsq_d1 : f32
    }
    %mean    = arith.divf %ln_sum, %nsz : f32
    %msq     = arith.mulf %mean, %mean : f32
    %esq     = arith.divf %ln_sumsq, %nsz : f32
    %var     = arith.subf %esq, %msq : f32
    %vep     = arith.addf %var, %eps : f32
    %std     = math.sqrt %vep : f32
    %one_f   = arith.constant 1.0 : f32
    %inv_std = arith.divf %one_f, %std : f32
    %t_res_ln_n0 = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%t_ln_n0 = %t_ln_ln_b1) -> (tensor<128x128x28x28xf32>) {
    %t_res_ln_n1 = scf.for %ln_n1 = %c0 to %input_d3 step %c1 iter_args(%t_ln_n1 = %t_ln_n0) -> (tensor<128x128x28x28xf32>) {
    %lnv3   = tensor.extract %input[%ln_b0, %ln_b1, %ln_n0, %ln_n1] : tensor<128x128x28x28xf32>
    %gv     = tensor.extract %gamma[%ln_n0, %ln_n1] : tensor<28x28xf32>
    %bv     = tensor.extract %beta[%ln_n0, %ln_n1]  : tensor<28x28xf32>
    %cent   = arith.subf %lnv3, %mean : f32
    %normed = arith.mulf %cent, %inv_std : f32
    %scaled = arith.mulf %normed, %gv : f32
    %res    = arith.addf %scaled, %bv : f32
    %t_ins_ln = tensor.insert %res into %t_ln_n1[%ln_b0, %ln_b1, %ln_n0, %ln_n1] : tensor<128x128x28x28xf32>
    scf.yield %t_ins_ln : tensor<128x128x28x28xf32>
    }
    scf.yield %t_res_ln_n1 : tensor<128x128x28x28xf32>
    }
    scf.yield %t_res_ln_n0 : tensor<128x128x28x28xf32>
    }
    scf.yield %t_res_ln_b1 : tensor<128x128x28x28xf32>
    }
    return %t_res_ln_b0 : tensor<128x128x28x28xf32>
  }
}
