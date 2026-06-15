module {
  func.func @f_21_vit_32x3x224x224_32x224x224x3_linalg(%input: tensor<32x3x224x224xf32>) -> tensor<32x224x224x3xf32> {
    %init = tensor.empty() : tensor<32x224x224x3xf32>
    %r = linalg.transpose ins(%input : tensor<32x3x224x224xf32>) outs(%init : tensor<32x224x224x3xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<32x224x224x3xf32>
  }
}
