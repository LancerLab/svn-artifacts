module {
  func.func @f_4_attention_32xNx512x64_32xNx512x64_32xNx512x64_mismatch(%a: tensor<32x?x512x64xf32>, %b: tensor<31x?x512x64xf32>) -> tensor<32x?x512x64xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c1 : tensor<32x?x512x64xf32>
    %init = tensor.empty(%d1) : tensor<32x?x512x64xf32>
    %r = linalg.add ins(%a, %b : tensor<32x?x512x64xf32>, tensor<31x?x512x64xf32>)
                    outs(%init : tensor<32x?x512x64xf32>) -> tensor<32x?x512x64xf32>
    return %r : tensor<32x?x512x64xf32>
  }
}
