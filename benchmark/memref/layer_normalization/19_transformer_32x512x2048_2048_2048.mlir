module @f_19_transformer_32x512x2048_2048_2048 {
  func.func @f_19_transformer_32x512x2048_2048_2048(%input: memref<32x512x2048xf32>, %gamma: memref<2048xf32>, %beta: memref<2048xf32>) -> memref<32x512x2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x512x2048xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x512x2048xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x512x2048xf32>
    %gamma_d0 = memref.dim %gamma, %c0 : memref<2048xf32>
    %beta_d0 = memref.dim %beta, %c0 : memref<2048xf32>
    %_lneq0 = arith.cmpi eq, %gamma_d0, %input_d2 : index
    cf.assert %_lneq0, "gamma.dim(0)==input.dim(2)"
    %_lneq1 = arith.cmpi eq, %beta_d0, %input_d2 : index
    cf.assert %_lneq1, "beta.dim(0)==input.dim(2)"
    %out = memref.alloc() : memref<32x512x2048xf32>
    scf.for %ln_b0 = %c0 to %input_d0 step %c1 {
      scf.for %ln_b1 = %c0 to %input_d1 step %c1 {
        %zero_f = arith.constant 0.0 : f32
        %eps    = arith.constant 1.0e-05 : f32
        %nsz    = arith.constant 2048.0 : f32
        %ln_sum = scf.for %ln_n0 = %c0 to %input_d2 step %c1 iter_args(%ln_sa0 = %zero_f) -> (f32) {
          %lnv1 = memref.load %input[%ln_b0, %ln_b1, %ln_n0] : memref<32x512x2048xf32>
          %ns1  = arith.addf %ln_sa0, %lnv1 : f32
          scf.yield %ns1 : f32
          }
        %ln_mean = arith.divf %ln_sum, %nsz : f32
        %ln_sumsq = scf.for %ln_n0v = %c0 to %input_d2 step %c1 iter_args(%ln_sq0 = %zero_f) -> (f32) {
          %lnv2 = memref.load %input[%ln_b0, %ln_b1, %ln_n0v] : memref<32x512x2048xf32>
          %lnd2 = arith.subf %lnv2, %ln_mean : f32
          %sq   = arith.mulf %lnd2, %lnd2 : f32
          %ns2  = arith.addf %ln_sq0, %sq : f32
          scf.yield %ns2 : f32
          }
        %ln_var    = arith.divf %ln_sumsq, %nsz : f32
        %ln_veps   = arith.addf %ln_var, %eps : f32
        %ln_invstd = math.rsqrt %ln_veps : f32
        scf.for %ln_w0 = %c0 to %input_d2 step %c1 {
          %lnv3    = memref.load %input[%ln_b0, %ln_b1, %ln_w0] : memref<32x512x2048xf32>
          %lnd3    = arith.subf %lnv3, %ln_mean : f32
          %ln_norm = arith.mulf %lnd3, %ln_invstd : f32
          %gv      = memref.load %gamma[%ln_w0] : memref<2048xf32>
          %bv      = memref.load %beta[%ln_w0]  : memref<2048xf32>
          %scaled  = arith.mulf %ln_norm, %gv : f32
          %ln_res  = arith.addf %scaled, %bv : f32
          memref.store %ln_res, %out[%ln_b0, %ln_b1, %ln_w0] : memref<32x512x2048xf32>
        }
      }
    }
    return %out : memref<32x512x2048xf32>
  }
}
