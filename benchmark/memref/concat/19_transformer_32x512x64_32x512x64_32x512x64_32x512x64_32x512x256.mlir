module @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256 {
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%in0: memref<32x512x64xf32>, %in1: memref<32x512x64xf32>, %in2: memref<32x512x64xf32>, %in3: memref<32x512x64xf32>) -> memref<32x512x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d0 = memref.dim %in0, %c0 : memref<32x512x64xf32>
    %in0_d1 = memref.dim %in0, %c1 : memref<32x512x64xf32>
    %in0_d2 = memref.dim %in0, %c2 : memref<32x512x64xf32>
    %in1_d0 = memref.dim %in1, %c0 : memref<32x512x64xf32>
    %in1_d1 = memref.dim %in1, %c1 : memref<32x512x64xf32>
    %in1_d2 = memref.dim %in1, %c2 : memref<32x512x64xf32>
    %in2_d0 = memref.dim %in2, %c0 : memref<32x512x64xf32>
    %in2_d1 = memref.dim %in2, %c1 : memref<32x512x64xf32>
    %in2_d2 = memref.dim %in2, %c2 : memref<32x512x64xf32>
    %in3_d0 = memref.dim %in3, %c0 : memref<32x512x64xf32>
    %in3_d1 = memref.dim %in3, %c1 : memref<32x512x64xf32>
    %in3_d2 = memref.dim %in3, %c2 : memref<32x512x64xf32>
    %_cceq10 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %_cceq10, "in0.dim(0)==in1.dim(0)"
    %_cceq11 = arith.cmpi eq, %in0_d1, %in1_d1 : index
    cf.assert %_cceq11, "in0.dim(1)==in1.dim(1)"
    %_cceq20 = arith.cmpi eq, %in0_d0, %in2_d0 : index
    cf.assert %_cceq20, "in0.dim(0)==in2.dim(0)"
    %_cceq21 = arith.cmpi eq, %in0_d1, %in2_d1 : index
    cf.assert %_cceq21, "in0.dim(1)==in2.dim(1)"
    %_cceq30 = arith.cmpi eq, %in0_d0, %in3_d0 : index
    cf.assert %_cceq30, "in0.dim(0)==in3.dim(0)"
    %_cceq31 = arith.cmpi eq, %in0_d1, %in3_d1 : index
    cf.assert %_cceq31, "in0.dim(1)==in3.dim(1)"
    %out = memref.alloc() : memref<32x512x256xf32>
    %off_1 = arith.addi %c0, %in0_d2 : index
    %off_2 = arith.addi %off_1, %in1_d2 : index
    %off_3 = arith.addi %off_2, %in2_d2 : index
    scf.for %cc0_d0 = %c0 to %in0_d0 step %c1 {
      scf.for %cc0_d1 = %c0 to %in0_d1 step %c1 {
        scf.for %cc0_d2 = %c0 to %in0_d2 step %c1 {
          %cc0_ax_out  = arith.addi %cc0_d2, %c0 : index
          %ccv0 = memref.load %in0[%cc0_d0, %cc0_d1, %cc0_d2] : memref<32x512x64xf32>
          memref.store %ccv0, %out[%cc0_d0, %cc0_d1, %cc0_ax_out] : memref<32x512x256xf32>
        }
      }
    }
    scf.for %cc1_d0 = %c0 to %in1_d0 step %c1 {
      scf.for %cc1_d1 = %c0 to %in1_d1 step %c1 {
        scf.for %cc1_d2 = %c0 to %in1_d2 step %c1 {
          %cc1_ax_out  = arith.addi %cc1_d2, %off_1 : index
          %ccv1 = memref.load %in1[%cc1_d0, %cc1_d1, %cc1_d2] : memref<32x512x64xf32>
          memref.store %ccv1, %out[%cc1_d0, %cc1_d1, %cc1_ax_out] : memref<32x512x256xf32>
        }
      }
    }
    scf.for %cc2_d0 = %c0 to %in2_d0 step %c1 {
      scf.for %cc2_d1 = %c0 to %in2_d1 step %c1 {
        scf.for %cc2_d2 = %c0 to %in2_d2 step %c1 {
          %cc2_ax_out  = arith.addi %cc2_d2, %off_2 : index
          %ccv2 = memref.load %in2[%cc2_d0, %cc2_d1, %cc2_d2] : memref<32x512x64xf32>
          memref.store %ccv2, %out[%cc2_d0, %cc2_d1, %cc2_ax_out] : memref<32x512x256xf32>
        }
      }
    }
    scf.for %cc3_d0 = %c0 to %in3_d0 step %c1 {
      scf.for %cc3_d1 = %c0 to %in3_d1 step %c1 {
        scf.for %cc3_d2 = %c0 to %in3_d2 step %c1 {
          %cc3_ax_out  = arith.addi %cc3_d2, %off_3 : index
          %ccv3 = memref.load %in3[%cc3_d0, %cc3_d1, %cc3_d2] : memref<32x512x64xf32>
          memref.store %ccv3, %out[%cc3_d0, %cc3_d1, %cc3_ax_out] : memref<32x512x256xf32>
        }
      }
    }
    return %out : memref<32x512x256xf32>
  }
}
