module @f_20_unet_16x512x32x32_16x32x32x512 {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%input: memref<16x512x32x32xf32>) -> memref<16x32x32x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %flat_in   = memref.collapse_shape %input [[0, 1, 2, 3]] : memref<16x512x32x32xf32> into memref<8388608xf32>
    %flat_out  = memref.alloc() : memref<8388608xf32>
    %flat_n    = arith.constant 8388608 : index
    scf.for %fi = %c0 to %flat_n step %c1 {
      %fv = memref.load %flat_in[%fi] : memref<8388608xf32>
      memref.store %fv, %flat_out[%fi] : memref<8388608xf32>
    }
    %out = memref.expand_shape %flat_out [[0, 1, 2, 3]] output_shape [16, 32, 32, 512] : memref<8388608xf32> into memref<16x32x32x512xf32>
    return %out : memref<16x32x32x512xf32>
  }
}
