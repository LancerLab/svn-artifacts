module @f_3_cnn_128x1000_50000x256_128x1000x256 {
  func.func @f_3_cnn_128x1000_50000x256_128x1000x256(%table: memref<50000x256xf32>, %indices: memref<128x1000xi64>) -> memref<128x1000x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %indices_d0 = memref.dim %indices, %c0 : memref<128x1000xi64>
    %indices_d1 = memref.dim %indices, %c1 : memref<128x1000xi64>
    %table_d0 = memref.dim %table, %c0 : memref<50000x256xf32>
    %table_d1 = memref.dim %table, %c1 : memref<50000x256xf32>
    %out = memref.alloc() : memref<128x1000x256xf32>
    scf.for %em_b0 = %c0 to %indices_d0 step %c1 {
      scf.for %em_b1 = %c0 to %indices_d1 step %c1 {
        scf.for %em_d0 = %c0 to %table_d1 step %c1 {
          %raw_idx = memref.load %indices[%em_b0, %em_b1] : memref<128x1000xi64>
          %row_idx = arith.index_cast %raw_idx : i64 to index
          %tv      = memref.load %table[%row_idx, %em_d0] : memref<50000x256xf32>
          memref.store %tv, %out[%em_b0, %em_b1, %em_d0] : memref<128x1000x256xf32>
        }
      }
    }
    return %out : memref<128x1000x256xf32>
  }
}
