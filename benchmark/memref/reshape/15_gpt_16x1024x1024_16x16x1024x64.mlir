module @f_15_gpt_16x1024x1024_16x16x1024x64 {
  func.func @f_15_gpt_16x1024x1024_16x16x1024x64(%input: memref<16x1024x1024xf32>) -> memref<16x16x1024x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %flat_in   = memref.collapse_shape %input [[0, 1, 2]] : memref<16x1024x1024xf32> into memref<16777216xf32>
    %flat_out  = memref.alloc() : memref<16777216xf32>
    %flat_n    = arith.constant 16777216 : index
    scf.for %fi = %c0 to %flat_n step %c1 {
      %fv = memref.load %flat_in[%fi] : memref<16777216xf32>
      memref.store %fv, %flat_out[%fi] : memref<16777216xf32>
    }
    %out = memref.expand_shape %flat_out [[0, 1, 2, 3]] output_shape [16, 16, 1024, 64] : memref<16777216xf32> into memref<16x16x1024x64xf32>
    return %out : memref<16x16x1024x64xf32>
  }
}
