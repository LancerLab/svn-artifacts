module {
  func.func @main() {
    %c0 = arith.constant 64 : index
    %c1 = arith.constant 128 : index
    %c2 = arith.constant 16 : index
    %c3 = arith.constant 16 : index
    %cr0 = arith.constant 63 : index
    %cr1 = arith.constant 128 : index
    %cr2 = arith.constant 16 : index
    %cr3 = arith.constant 16 : index
    %cst1 = arith.constant 1.0 : f32
    %cst2 = arith.constant 2.0 : f32
    %a = memref.alloc(%c0, %c1, %c2, %c3) : memref<?x?x?x?xf32>
    %b = memref.alloc(%cr0, %cr1, %cr2, %cr3) : memref<?x?x?x?xf32>
    %out = memref.alloc(%c0, %c1, %c2, %c3) : memref<?x?x?x?xf32>
    linalg.fill ins(%cst1 : f32) outs(%a : memref<?x?x?x?xf32>)
    linalg.fill ins(%cst2 : f32) outs(%b : memref<?x?x?x?xf32>)
    linalg.add ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
              outs(%out : memref<?x?x?x?xf32>)
    %z0 = arith.constant 0 : index
    %z1 = arith.constant 0 : index
    %z2 = arith.constant 0 : index
    %z3 = arith.constant 0 : index
    %val = memref.load %out[%z0, %z1, %z2, %z3] : memref<?x?x?x?xf32>
    vector.print %val : f32
    memref.dealloc %a : memref<?x?x?x?xf32>
    memref.dealloc %b : memref<?x?x?x?xf32>
    memref.dealloc %out : memref<?x?x?x?xf32>
    return
  }
}
