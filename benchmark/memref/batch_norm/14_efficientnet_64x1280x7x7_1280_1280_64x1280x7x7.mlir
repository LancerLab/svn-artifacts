module @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7(%input: memref<64x1280x7x7xf32>, %gamma: memref<1280xf32>, %beta: memref<1280xf32>) -> memref<64x1280x7x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<64x1280x7x7xf32>
    %input_d1 = memref.dim %input, %c1 : memref<64x1280x7x7xf32>
    %input_d2 = memref.dim %input, %c2 : memref<64x1280x7x7xf32>
    %input_d3 = memref.dim %input, %c3 : memref<64x1280x7x7xf32>
    %gamma_d0 = memref.dim %gamma, %c0 : memref<1280xf32>
    %beta_d0 = memref.dim %beta, %c0 : memref<1280xf32>
    %_bneq0 = arith.cmpi eq, %gamma_d0, %input_d1 : index
    cf.assert %_bneq0, "gamma.dim(0)==input.dim(1)"
    %_bneq1 = arith.cmpi eq, %beta_d0, %input_d1 : index
    cf.assert %_bneq1, "beta.dim(0)==input.dim(1)"
    %out = memref.alloc() : memref<64x1280x7x7xf32>
    scf.for %bn0 = %c0 to %input_d0 step %c1 {
      scf.for %bn1 = %c0 to %input_d1 step %c1 {
        scf.for %bn2 = %c0 to %input_d2 step %c1 {
          scf.for %bn3 = %c0 to %input_d3 step %c1 {
            %xv  = memref.load %input[%bn0, %bn1, %bn2, %bn3] : memref<64x1280x7x7xf32>
            %gv  = memref.load %gamma[%bn1] : memref<1280xf32>
            %bv  = memref.load %beta[%bn1]  : memref<1280xf32>
            %scl = arith.mulf %xv, %gv : f32
            %res = arith.addf %scl, %bv : f32
            memref.store %res, %out[%bn0, %bn1, %bn2, %bn3] : memref<64x1280x7x7xf32>
          }
        }
      }
    }
    return %out : memref<64x1280x7x7xf32>
  }
}
