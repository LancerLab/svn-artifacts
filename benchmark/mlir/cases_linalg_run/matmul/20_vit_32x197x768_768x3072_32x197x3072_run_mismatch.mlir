#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {
  func.func @main() {
    %cB = arith.constant 32 : index
    %cM = arith.constant 197 : index
    %cK = arith.constant 768 : index
    %cK_rhs = arith.constant 767 : index
    %cN = arith.constant 3072 : index
    %cst = arith.constant 0.0 : f32
    %cst1 = arith.constant 1.0 : f32

    %lhs = memref.alloc(%cB, %cM, %cK) : memref<?x?x?xf32>
    %rhs = memref.alloc(%cK_rhs, %cN) : memref<?x?xf32>
    %out = memref.alloc(%cB, %cM, %cN) : memref<?x?x?xf32>

    linalg.fill ins(%cst1 : f32) outs(%lhs : memref<?x?x?xf32>)
    linalg.fill ins(%cst1 : f32) outs(%rhs : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%out : memref<?x?x?xf32>)

    linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : memref<?x?x?xf32>, memref<?x?xf32>)
      outs(%out : memref<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
    }

    %c0 = arith.constant 0 : index
    %val = memref.load %out[%c0, %c0, %c0] : memref<?x?x?xf32>
    vector.print %val : f32

    memref.dealloc %lhs : memref<?x?x?xf32>
    memref.dealloc %rhs : memref<?x?xf32>
    memref.dealloc %out : memref<?x?x?xf32>
    return
  }
}

