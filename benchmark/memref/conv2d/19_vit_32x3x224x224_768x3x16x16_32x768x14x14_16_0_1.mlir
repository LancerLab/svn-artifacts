module @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1 {
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1(%input: memref<32x3x224x224xf32>, %filter: memref<768x3x16x16xf32>) -> memref<32x768x14x14xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x3x224x224xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x3x224x224xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x3x224x224xf32>
    %input_d3 = memref.dim %input, %c3 : memref<32x3x224x224xf32>
    %filter_d0 = memref.dim %filter, %c0 : memref<768x3x16x16xf32>
    %filter_d1 = memref.dim %filter, %c1 : memref<768x3x16x16xf32>
    %filter_d2 = memref.dim %filter, %c2 : memref<768x3x16x16xf32>
    %filter_d3 = memref.dim %filter, %c3 : memref<768x3x16x16xf32>
    %_cveq0 = arith.cmpi eq, %input_d1, %filter_d1 : index
    cf.assert %_cveq0, "input.dim(1)==filter.dim(1)"
    %out = memref.alloc() : memref<32x768x14x14xf32>
    %kH_sz  = arith.constant 16 : index
    %kW_sz  = arith.constant 16 : index
    %stride = arith.constant 16 : index
    scf.for %n = %c0 to %input_d0 step %c1 {
      scf.for %f_ = %c0 to %filter_d0 step %c1 {
        scf.for %oh = %c0 to %input_d2 step %c1 {
          scf.for %ow = %c0 to %input_d3 step %c1 {
            %zero_f = arith.constant 0.0 : f32
            %acc = scf.for %c = %c0 to %input_d1 step %c1 iter_args(%s_c = %zero_f) -> (f32) {
              %acc2 = scf.for %kh = %c0 to %kH_sz step %c1 iter_args(%s_kh = %s_c) -> (f32) {
                %acc3 = scf.for %kw = %c0 to %kW_sz step %c1 iter_args(%s_kw = %s_kh) -> (f32) {
                  %s_oh = arith.muli %stride, %oh : index
                  %ih   = arith.addi %s_oh, %kh : index
                  %s_ow = arith.muli %stride, %ow : index
                  %iw   = arith.addi %s_ow, %kw : index
                  %av   = memref.load %input[%n, %c, %ih, %iw] : memref<32x3x224x224xf32>
                  %bv   = memref.load %filter[%f_, %c, %kh, %kw] : memref<768x3x16x16xf32>
                  %pv   = arith.mulf %av, %bv : f32
                  %ns   = arith.addf %s_kw, %pv : f32
                  scf.yield %ns : f32
                }
                scf.yield %acc3 : f32
              }
              scf.yield %acc2 : f32
            }
            memref.store %acc, %out[%n, %f_, %oh, %ow] : memref<32x768x14x14xf32>
          }
        }
      }
    }
    return %out : memref<32x768x14x14xf32>
  }
}
