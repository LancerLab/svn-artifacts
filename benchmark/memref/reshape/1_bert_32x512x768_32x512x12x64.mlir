module @f_1_bert_32x512x768_32x512x12x64 {
  func.func @f_1_bert_32x512x768_32x512x12x64(%input: memref<32x512x768xf32>) -> memref<32x512x12x64xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.expand_shape %input [[0], [1], [2, 3]] output_shape [32, 512, 12, 64] : memref<32x512x768xf32> into memref<32x512x12x64xf32>
    return %out : memref<32x512x12x64xf32>
  }
}
