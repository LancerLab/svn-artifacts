module @f_1_bert_32x512x768_32x512x768_32x512x1536 {
  func.func @f_1_bert_32x512x768_32x512x768_32x512x1536(%in0: memref<32x512x768xf32>, %in1: memref<32x512x768xf32>) -> memref<32x512x1536xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d0 = memref.dim %in0, %c0 : memref<32x512x768xf32>
    %in0_d1 = memref.dim %in0, %c1 : memref<32x512x768xf32>
    %in0_d2 = memref.dim %in0, %c2 : memref<32x512x768xf32>
    %in1_d0 = memref.dim %in1, %c0 : memref<32x512x768xf32>
    %in1_d1 = memref.dim %in1, %c1 : memref<32x512x768xf32>
    %in1_d2 = memref.dim %in1, %c2 : memref<32x512x768xf32>
    %_cceq10 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %_cceq10, "in0.dim(0)==in1.dim(0)"
    %_cceq11 = arith.cmpi eq, %in0_d1, %in1_d1 : index
    cf.assert %_cceq11, "in0.dim(1)==in1.dim(1)"
    %out = memref.alloc() : memref<32x512x1536xf32>
    %off_1 = arith.addi %c0, %in0_d2 : index
    scf.for %cc0_d0 = %c0 to %in0_d0 step %c1 {
      scf.for %cc0_d1 = %c0 to %in0_d1 step %c1 {
        scf.for %cc0_d2 = %c0 to %in0_d2 step %c1 {
          %cc0_ax_out  = arith.addi %cc0_d2, %c0 : index
          %ccv0 = memref.load %in0[%cc0_d0, %cc0_d1, %cc0_d2] : memref<32x512x768xf32>
          memref.store %ccv0, %out[%cc0_d0, %cc0_d1, %cc0_ax_out] : memref<32x512x1536xf32>
        }
      }
    }
    scf.for %cc1_d0 = %c0 to %in1_d0 step %c1 {
      scf.for %cc1_d1 = %c0 to %in1_d1 step %c1 {
        scf.for %cc1_d2 = %c0 to %in1_d2 step %c1 {
          %cc1_ax_out  = arith.addi %cc1_d2, %off_1 : index
          %ccv1 = memref.load %in1[%cc1_d0, %cc1_d1, %cc1_d2] : memref<32x512x768xf32>
          memref.store %ccv1, %out[%cc1_d0, %cc1_d1, %cc1_ax_out] : memref<32x512x1536xf32>
        }
      }
    }
    return %out : memref<32x512x1536xf32>
  }
}
