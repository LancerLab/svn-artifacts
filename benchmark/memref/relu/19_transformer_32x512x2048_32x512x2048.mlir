module @f_19_transformer_32x512x2048_32x512x2048 {
  func.func @f_19_transformer_32x512x2048_32x512x2048(%input: memref<32x512x2048xf32>) -> memref<32x512x2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x512x2048xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x512x2048xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x512x2048xf32>
    %out = memref.alloc() : memref<32x512x2048xf32>
    scf.for %ui0 = %c0 to %input_d0 step %c1 {
      scf.for %ui1 = %c0 to %input_d1 step %c1 {
        scf.for %ui2 = %c0 to %input_d2 step %c1 {
          %in_val  = memref.load %input[%ui0, %ui1, %ui2] : memref<32x512x2048xf32>
          %zero_f  = arith.constant 0.0 : f32
          %out_val = arith.maximumf %in_val, %zero_f : f32
          memref.store %out_val, %out[%ui0, %ui1, %ui2] : memref<32x512x2048xf32>
        }
      }
    }
    return %out : memref<32x512x2048xf32>
  }
}
