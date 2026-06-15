module {
  func.func @main() {
    %cM = arith.constant 128 : index
    %cK = arith.constant 1280 : index
    %cK_rhs = arith.constant 1279 : index
    %cN = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %cst1 = arith.constant 1.0 : f32

    %lhs = memref.alloc(%cM, %cK) : memref<?x?xf32>
    %rhs = memref.alloc(%cK_rhs, %cN) : memref<?x?xf32>
    %out = memref.alloc(%cM, %cN) : memref<?x?xf32>

    linalg.fill ins(%cst1 : f32) outs(%lhs : memref<?x?xf32>)
    linalg.fill ins(%cst1 : f32) outs(%rhs : memref<?x?xf32>)
    linalg.fill ins(%cst : f32) outs(%out : memref<?x?xf32>)

    linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
                   outs(%out : memref<?x?xf32>)

    %c0 = arith.constant 0 : index
    %val = memref.load %out[%c0, %c0] : memref<?x?xf32>
    vector.print %val : f32

    memref.dealloc %lhs : memref<?x?xf32>
    memref.dealloc %rhs : memref<?x?xf32>
    memref.dealloc %out : memref<?x?xf32>
    return
  }
}

