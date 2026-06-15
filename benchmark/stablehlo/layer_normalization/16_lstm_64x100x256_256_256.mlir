module @f_16_lstm_64x100x256_256_256 {
  func.func @f_16_lstm_64x100x256_256_256(%input: tensor<64x100x256xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<64x100x256xf32> {
    %zero_f   = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero_f) across dimensions = [2] : (tensor<64x100x256xf32>, tensor<f32>) -> tensor<64x100xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %sq_in    = stablehlo.multiply %input, %input : tensor<64x100x256xf32>
    %sq_red   = stablehlo.reduce(%sq_in init: %zero_f) across dimensions = [2] : (tensor<64x100x256xf32>, tensor<f32>) -> tensor<64x100xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s2 = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s2 : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<256.0> : tensor<64x100xf32>
    %eps      = stablehlo.constant dense<1.0e-05> : tensor<64x100xf32>
    %mean     = stablehlo.divide %sum_red, %nsz : tensor<64x100xf32>
    %mean_sq  = stablehlo.divide %sq_red, %nsz : tensor<64x100xf32>
    %m2       = stablehlo.multiply %mean, %mean : tensor<64x100xf32>
    %variance = stablehlo.subtract %mean_sq, %m2 : tensor<64x100xf32>
    %var_eps  = stablehlo.add %variance, %eps : tensor<64x100xf32>
    %inv_std  = stablehlo.rsqrt %var_eps : tensor<64x100xf32>
    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<64x100xf32>) -> tensor<64x100x256xf32>
    %istd_b = stablehlo.broadcast_in_dim %inv_std, dims = [0, 1] : (tensor<64x100xf32>) -> tensor<64x100x256xf32>
    %cent     = stablehlo.subtract %input, %mean_b : tensor<64x100x256xf32>
    %normed   = stablehlo.multiply %cent, %istd_b : tensor<64x100x256xf32>
    %g_b = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<256xf32>) -> tensor<64x100x256xf32>
    %b_b = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<256xf32>) -> tensor<64x100x256xf32>
    %scaled   = stablehlo.multiply %normed, %g_b : tensor<64x100x256xf32>
    %result   = stablehlo.add %scaled, %b_b : tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}
