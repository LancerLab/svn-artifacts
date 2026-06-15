module @f_17_mobilenet_128x96x7x7_128x4704 {
  func.func @f_17_mobilenet_128x96x7x7_128x4704(%input: memref<128x96x7x7xf32>) -> memref<128x4704xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0], [1, 2, 3]] : memref<128x96x7x7xf32> into memref<128x4704xf32>
    return %out : memref<128x4704xf32>
  }
}
