module @f_15_gpt_16x1024x4096_4096_4096_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096_4096_4096_16x1024x4096(%input: memref<16x1024x4096xf32>, %gamma: memref<4096xf32>, %beta: memref<4096xf32>) -> memref<16x1024x4096xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<16x1024x4096xf32>
    %input_d1 = memref.dim %input, %c1 : memref<16x1024x4096xf32>
    %input_d2 = memref.dim %input, %c2 : memref<16x1024x4096xf32>
    %gamma_d0 = memref.dim %gamma, %c0 : memref<4096xf32>
    %beta_d0 = memref.dim %beta, %c0 : memref<4096xf32>
    %_bneq0 = arith.cmpi eq, %gamma_d0, %input_d2 : index
    cf.assert %_bneq0, "gamma.dim(0)==input.dim(2)"
    %_bneq1 = arith.cmpi eq, %beta_d0, %input_d2 : index
    cf.assert %_bneq1, "beta.dim(0)==input.dim(2)"
    %out = memref.alloc() : memref<16x1024x4096xf32>
    scf.for %bn0 = %c0 to %input_d0 step %c1 {
      scf.for %bn1 = %c0 to %input_d1 step %c1 {
        scf.for %bn2 = %c0 to %input_d2 step %c1 {
          %xv  = memref.load %input[%bn0, %bn1, %bn2] : memref<16x1024x4096xf32>
          %gv  = memref.load %gamma[%bn2] : memref<4096xf32>
          %bv  = memref.load %beta[%bn2]  : memref<4096xf32>
          %scl = arith.mulf %xv, %gv : f32
          %res = arith.addf %scl, %bv : f32
          memref.store %res, %out[%bn0, %bn1, %bn2] : memref<16x1024x4096xf32>
        }
      }
    }
    return %out : memref<16x1024x4096xf32>
  }
}
