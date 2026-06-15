module @f_21_vit_32x197x768_32x197x768_32x197x768 {
  func.func @f_21_vit_32x197x768_32x197x768_32x197x768(%in0: memref<32x197x768xf32>, %in1: memref<32x197x768xf32>) -> memref<32x197x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d0 = memref.dim %in0, %c0 : memref<32x197x768xf32>
    %in0_d1 = memref.dim %in0, %c1 : memref<32x197x768xf32>
    %in0_d2 = memref.dim %in0, %c2 : memref<32x197x768xf32>
    %in1_d0 = memref.dim %in1, %c0 : memref<32x197x768xf32>
    %in1_d1 = memref.dim %in1, %c1 : memref<32x197x768xf32>
    %in1_d2 = memref.dim %in1, %c2 : memref<32x197x768xf32>
    %_aeq0 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %_aeq0, "in0.dim(0)==in1.dim(0)"
    %_aeq1 = arith.cmpi eq, %in0_d1, %in1_d1 : index
    cf.assert %_aeq1, "in0.dim(1)==in1.dim(1)"
    %_aeq2 = arith.cmpi eq, %in0_d2, %in1_d2 : index
    cf.assert %_aeq2, "in0.dim(2)==in1.dim(2)"
    %out = memref.alloc() : memref<32x197x768xf32>
    scf.for %ae0 = %c0 to %in0_d0 step %c1 {
      scf.for %ae1 = %c0 to %in0_d1 step %c1 {
        scf.for %ae2 = %c0 to %in0_d2 step %c1 {
          %v0 = memref.load %in0[%ae0, %ae1, %ae2] : memref<32x197x768xf32>
          %v1 = memref.load %in1[%ae0, %ae1, %ae2] : memref<32x197x768xf32>
          %out_val = arith.addf %v0, %v1 : f32
          memref.store %out_val, %out[%ae0, %ae1, %ae2] : memref<32x197x768xf32>
        }
      }
    }
    return %out : memref<32x197x768xf32>
  }
}
