module @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112 {
  func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112(%in0: memref<128x96x112x112xf32>, %in1: memref<128x32x112x112xf32>) -> memref<128x128x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d0 = memref.dim %in0, %c0 : memref<128x96x112x112xf32>
    %in0_d1 = memref.dim %in0, %c1 : memref<128x96x112x112xf32>
    %in0_d2 = memref.dim %in0, %c2 : memref<128x96x112x112xf32>
    %in0_d3 = memref.dim %in0, %c3 : memref<128x96x112x112xf32>
    %in1_d0 = memref.dim %in1, %c0 : memref<128x32x112x112xf32>
    %in1_d1 = memref.dim %in1, %c1 : memref<128x32x112x112xf32>
    %in1_d2 = memref.dim %in1, %c2 : memref<128x32x112x112xf32>
    %in1_d3 = memref.dim %in1, %c3 : memref<128x32x112x112xf32>
    %_cceq10 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %_cceq10, "in0.dim(0)==in1.dim(0)"
    %_cceq12 = arith.cmpi eq, %in0_d2, %in1_d2 : index
    cf.assert %_cceq12, "in0.dim(2)==in1.dim(2)"
    %_cceq13 = arith.cmpi eq, %in0_d3, %in1_d3 : index
    cf.assert %_cceq13, "in0.dim(3)==in1.dim(3)"
    %out = memref.alloc() : memref<128x128x112x112xf32>
    %off_1 = arith.addi %c0, %in0_d1 : index
    scf.for %cc0_d0 = %c0 to %in0_d0 step %c1 {
      scf.for %cc0_d1 = %c0 to %in0_d1 step %c1 {
        scf.for %cc0_d2 = %c0 to %in0_d2 step %c1 {
          scf.for %cc0_d3 = %c0 to %in0_d3 step %c1 {
            %cc0_ax_out  = arith.addi %cc0_d1, %c0 : index
            %ccv0 = memref.load %in0[%cc0_d0, %cc0_d1, %cc0_d2, %cc0_d3] : memref<128x96x112x112xf32>
            memref.store %ccv0, %out[%cc0_d0, %cc0_ax_out, %cc0_d2, %cc0_d3] : memref<128x128x112x112xf32>
          }
        }
      }
    }
    scf.for %cc1_d0 = %c0 to %in1_d0 step %c1 {
      scf.for %cc1_d1 = %c0 to %in1_d1 step %c1 {
        scf.for %cc1_d2 = %c0 to %in1_d2 step %c1 {
          scf.for %cc1_d3 = %c0 to %in1_d3 step %c1 {
            %cc1_ax_out  = arith.addi %cc1_d1, %off_1 : index
            %ccv1 = memref.load %in1[%cc1_d0, %cc1_d1, %cc1_d2, %cc1_d3] : memref<128x32x112x112xf32>
            memref.store %ccv1, %out[%cc1_d0, %cc1_ax_out, %cc1_d2, %cc1_d3] : memref<128x128x112x112xf32>
          }
        }
      }
    }
    return %out : memref<128x128x112x112xf32>
  }
}
