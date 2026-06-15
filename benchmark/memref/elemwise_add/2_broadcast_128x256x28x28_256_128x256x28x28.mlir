module @f_2_broadcast_128x256x28x28_256_128x256x28x28 {
  func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28(%in0: memref<128x256x28x28xf32>, %in1: memref<256xf32>) -> memref<128x256x28x28xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d0 = memref.dim %in0, %c0 : memref<128x256x28x28xf32>
    %in0_d1 = memref.dim %in0, %c1 : memref<128x256x28x28xf32>
    %in0_d2 = memref.dim %in0, %c2 : memref<128x256x28x28xf32>
    %in0_d3 = memref.dim %in0, %c3 : memref<128x256x28x28xf32>
    %in1_d0 = memref.dim %in1, %c0 : memref<256xf32>
    %_baeq0 = arith.cmpi eq, %in1_d0, %in0_d1 : index
    cf.assert %_baeq0, "in1.dim(0)==in0.dim(1)"
    %out = memref.alloc() : memref<128x256x28x28xf32>
    scf.for %ae0 = %c0 to %in0_d0 step %c1 {
      scf.for %ae1 = %c0 to %in0_d1 step %c1 {
        scf.for %ae2 = %c0 to %in0_d2 step %c1 {
          scf.for %ae3 = %c0 to %in0_d3 step %c1 {
            %v0 = memref.load %in0[%ae0, %ae1, %ae2, %ae3] : memref<128x256x28x28xf32>
            %v1 = memref.load %in1[%ae1] : memref<256xf32>
            %out_val = arith.addf %v0, %v1 : f32
            memref.store %out_val, %out[%ae0, %ae1, %ae2, %ae3] : memref<128x256x28x28xf32>
          }
        }
      }
    }
    return %out : memref<128x256x28x28xf32>
  }
}
