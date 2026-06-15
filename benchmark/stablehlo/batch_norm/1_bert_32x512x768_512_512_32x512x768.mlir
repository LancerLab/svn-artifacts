module @f_1_bert_32x512x768_512_512_32x512x768 {
  func.func @f_1_bert_32x512x768_512_512_32x512x768(%input: tensor<32x512x768xf32>, %gamma: tensor<512xf32>, %beta: tensor<512xf32>) -> tensor<32x512x768xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<512xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<512xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<32x512x768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}
