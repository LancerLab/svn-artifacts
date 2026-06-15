module @f_15_lstm_64x100x256_64x256 {
  func.func @f_15_lstm_64x100x256_64x256(%input: memref<64x100x256xf32>) -> memref<64x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<64x100x256xf32>
    %input_d1 = memref.dim %input, %c1 : memref<64x100x256xf32>
    %input_d2 = memref.dim %input, %c2 : memref<64x100x256xf32>
    %out = memref.alloc() : memref<64x256xf32>
    %nsz = arith.constant 100.0 : f32
    scf.for %rm_b0 = %c0 to %input_d0 step %c1 {
      scf.for %rm_b1 = %c0 to %input_d2 step %c1 {
        %zero_f = arith.constant 0.0 : f32
        %rm_sum = scf.for %rm_k = %c0 to %input_d1 step %c1 iter_args(%rm_s = %zero_f) -> (f32) {
          %rv   = memref.load %input[%rm_b0, %rm_k, %rm_b1] : memref<64x100x256xf32>
          %ns   = arith.addf %rm_s, %rv : f32
          scf.yield %ns : f32
        }
        %rm_mean = arith.divf %rm_sum, %nsz : f32
        memref.store %rm_mean, %out[%rm_b0, %rm_b1] : memref<64x256xf32>
      }
    }
    return %out : memref<64x256xf32>
  }
}
