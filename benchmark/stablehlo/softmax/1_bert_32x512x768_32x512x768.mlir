module @f_1_bert_32x512x768_32x512x768 {
  func.func @f_1_bert_32x512x768_32x512x768(%input: tensor<32x512x768xf32>) -> tensor<32x512x768xf32> {
    %neg_inf = stablehlo.constant dense<-3.402820e+38> : tensor<f32>
    %max_red = stablehlo.reduce(%input init: %neg_inf) across dimensions = [2] : (tensor<32x512x768xf32>, tensor<f32>) -> tensor<32x512xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
        stablehlo.return %m : tensor<f32>
      }
    %max_bcast = stablehlo.broadcast_in_dim %max_red, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x768xf32>
    %shifted   = stablehlo.subtract %input, %max_bcast : tensor<32x512x768xf32>
    %exp_vals  = stablehlo.exponential %shifted : tensor<32x512x768xf32>
    %zero_sum  = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red   = stablehlo.reduce(%exp_vals init: %zero_sum) across dimensions = [2] : (tensor<32x512x768xf32>, tensor<f32>) -> tensor<32x512xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %sum_bcast = stablehlo.broadcast_in_dim %sum_red, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x768xf32>
    %result    = stablehlo.divide %exp_vals, %sum_bcast : tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}
