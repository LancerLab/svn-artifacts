module {
  func.func @main() {
    %c0 = arith.constant 32 : index
    %c1 = arith.constant 197 : index
    %c2 = arith.constant 768 : index
    %cr0 = arith.constant 31 : index
    %cr1 = arith.constant 197 : index
    %cr2 = arith.constant 768 : index
    %cst1 = arith.constant 1.0 : f32
    %cst2 = arith.constant 2.0 : f32
    %a = memref.alloc(%c0, %c1, %c2) : memref<?x?x?xf32>
    %b = memref.alloc(%cr0, %cr1, %cr2) : memref<?x?x?xf32>
    %out = memref.alloc(%c0, %c1, %c2) : memref<?x?x?xf32>
    linalg.fill ins(%cst1 : f32) outs(%a : memref<?x?x?xf32>)
    linalg.fill ins(%cst2 : f32) outs(%b : memref<?x?x?xf32>)
    linalg.add ins(%a, %b : memref<?x?x?xf32>, memref<?x?x?xf32>)
              outs(%out : memref<?x?x?xf32>)
    %z0 = arith.constant 0 : index
    %z1 = arith.constant 0 : index
    %z2 = arith.constant 0 : index
    %val = memref.load %out[%z0, %z1, %z2] : memref<?x?x?xf32>
    vector.print %val : f32
    memref.dealloc %a : memref<?x?x?xf32>
    memref.dealloc %b : memref<?x?x?xf32>
    memref.dealloc %out : memref<?x?x?xf32>
    return
  }
}
