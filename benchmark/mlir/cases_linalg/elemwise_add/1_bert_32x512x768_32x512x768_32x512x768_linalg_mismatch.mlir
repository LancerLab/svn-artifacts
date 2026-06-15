module {
  func.func @f_1_bert_32x512x768_32x512x768_32x512x768_mismatch(%a: tensor<32x512x768xf32>, %b: tensor<31x512x768xf32>) -> tensor<32x512x768xf32> {
    %init = tensor.empty() : tensor<32x512x768xf32>
    %r = linalg.add ins(%a, %b : tensor<32x512x768xf32>, tensor<31x512x768xf32>)
                    outs(%init : tensor<32x512x768xf32>) -> tensor<32x512x768xf32>
    return %r : tensor<32x512x768xf32>
  }
}
