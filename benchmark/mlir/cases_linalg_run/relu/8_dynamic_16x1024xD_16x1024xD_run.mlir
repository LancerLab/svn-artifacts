#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main() {
    %c0 = arith.constant 16 : index
    %c1 = arith.constant 1024 : index
    %c2 = arith.constant 16 : index
    %cst_neg = arith.constant -2.5 : f32
    %inp = memref.alloc(%c0, %c1, %c2) : memref<?x?x?xf32>
    %out = memref.alloc(%c0, %c1, %c2) : memref<?x?x?xf32>
    linalg.fill ins(%cst_neg : f32) outs(%inp : memref<?x?x?xf32>)
    linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%inp : memref<?x?x?xf32>) outs(%out : memref<?x?x?xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    }
    %z0 = arith.constant 0 : index
    %z1 = arith.constant 0 : index
    %z2 = arith.constant 0 : index
    %val = memref.load %out[%z0, %z1, %z2] : memref<?x?x?xf32>
    vector.print %val : f32
    memref.dealloc %inp : memref<?x?x?xf32>
    memref.dealloc %out : memref<?x?x?xf32>
    return
  }
}
