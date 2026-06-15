module @f_14_efficientnet_64x1280x7x7_64x1280x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7(%input: memref<64x1280x7x7xf32>) -> memref<64x1280x7x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<64x1280x7x7xf32>
    %input_d1 = memref.dim %input, %c1 : memref<64x1280x7x7xf32>
    %input_d2 = memref.dim %input, %c2 : memref<64x1280x7x7xf32>
    %input_d3 = memref.dim %input, %c3 : memref<64x1280x7x7xf32>
    %out = memref.alloc() : memref<64x1280x7x7xf32>
    scf.for %ui0 = %c0 to %input_d0 step %c1 {
      scf.for %ui1 = %c0 to %input_d1 step %c1 {
        scf.for %ui2 = %c0 to %input_d2 step %c1 {
          scf.for %ui3 = %c0 to %input_d3 step %c1 {
            %in_val  = memref.load %input[%ui0, %ui1, %ui2, %ui3] : memref<64x1280x7x7xf32>
            %zero_f  = arith.constant 0.0 : f32
            %out_val = arith.maximumf %in_val, %zero_f : f32
            memref.store %out_val, %out[%ui0, %ui1, %ui2, %ui3] : memref<64x1280x7x7xf32>
          }
        }
      }
    }
    return %out : memref<64x1280x7x7xf32>
  }
}
