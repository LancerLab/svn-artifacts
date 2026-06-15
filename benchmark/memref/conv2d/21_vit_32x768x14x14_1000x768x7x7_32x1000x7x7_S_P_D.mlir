module @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D {
  func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D(%input: memref<32x768x14x14xf32>, %filter: memref<1000x768x7x7xf32>) -> memref<32x1000x7x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x768x14x14xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x768x14x14xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x768x14x14xf32>
    %input_d3 = memref.dim %input, %c3 : memref<32x768x14x14xf32>
    %filter_d0 = memref.dim %filter, %c0 : memref<1000x768x7x7xf32>
    %filter_d1 = memref.dim %filter, %c1 : memref<1000x768x7x7xf32>
    %filter_d2 = memref.dim %filter, %c2 : memref<1000x768x7x7xf32>
    %filter_d3 = memref.dim %filter, %c3 : memref<1000x768x7x7xf32>
    %_cveq0 = arith.cmpi eq, %input_d1, %filter_d1 : index
    cf.assert %_cveq0, "input.dim(1)==filter.dim(1)"
    %out = memref.alloc() : memref<32x1000x7x7xf32>
    %kH_sz  = arith.constant 7 : index
    %kW_sz  = arith.constant 7 : index
    %stride = arith.constant 2 : index
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
                  %av   = memref.load %input[%n, %c, %ih, %iw] : memref<32x768x14x14xf32>
                  %bv   = memref.load %filter[%f_, %c, %kh, %kw] : memref<1000x768x7x7xf32>
                  %pv   = arith.mulf %av, %bv : f32
                  %ns   = arith.addf %s_kw, %pv : f32
                  scf.yield %ns : f32
                }
                scf.yield %acc3 : f32
              }
              scf.yield %acc2 : f32
            }
            memref.store %acc, %out[%n, %f_, %oh, %ow] : memref<32x1000x7x7xf32>
          }
        }
      }
    }
    return %out : memref<32x1000x7x7xf32>
  }
}
