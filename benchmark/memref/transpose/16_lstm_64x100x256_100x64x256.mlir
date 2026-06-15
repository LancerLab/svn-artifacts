module @f_16_lstm_64x100x256_100x64x256 {
  func.func @f_16_lstm_64x100x256_100x64x256(%input: memref<64x100x256xf32>) -> memref<100x64x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<64x100x256xf32>
    %input_d1 = memref.dim %input, %c1 : memref<64x100x256xf32>
    %input_d2 = memref.dim %input, %c2 : memref<64x100x256xf32>
    %out = memref.alloc() : memref<100x64x256xf32>
    scf.for %pi0 = %c0 to %input_d1 step %c1 {
      scf.for %pi1 = %c0 to %input_d0 step %c1 {
        scf.for %pi2 = %c0 to %input_d2 step %c1 {
          %tv    = memref.load %input[%pi1, %pi0, %pi2] : memref<64x100x256xf32>
          memref.store %tv, %out[%pi0, %pi1, %pi2] : memref<100x64x256xf32>
        }
      }
    }
    return %out : memref<100x64x256xf32>
  }
}
