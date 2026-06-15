module @f_1_bert_32x512x768 {
  func.func @f_1_bert_32x512x768(%input: memref<32x512x768xf32>) -> memref<32x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x512x768xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x512x768xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x512x768xf32>
    %out = memref.alloc() : memref<32x512x768xf32>
    scf.for %si0 = %c0 to %input_d0 step %c1 {
      scf.for %si1 = %c0 to %input_d1 step %c1 {
        scf.for %si2 = %c0 to %input_d2 step %c1 {
          %xv    = memref.load %input[%si0, %si1, %si2] : memref<32x512x768xf32>
          %one   = arith.constant 1.0 : f32
          %negx  = arith.negf %xv : f32
          %expv  = math.exp %negx : f32
          %denom = arith.addf %one, %expv : f32
          %out_val = arith.divf %one, %denom : f32
          memref.store %out_val, %out[%si0, %si1, %si2] : memref<32x512x768xf32>
        }
      }
    }
    return %out : memref<32x512x768xf32>
  }
}
