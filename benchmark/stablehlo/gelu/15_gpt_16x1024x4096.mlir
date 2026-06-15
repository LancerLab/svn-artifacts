module @f_15_gpt_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096(%input: tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32> {
    %half    = stablehlo.constant dense<5.000000e-01> : tensor<16x1024x4096xf32>
    %one     = stablehlo.constant dense<1.000000e+00> : tensor<16x1024x4096xf32>
    %sqrt2pi = stablehlo.constant dense<7.978846e-01> : tensor<16x1024x4096xf32>
    %coeff   = stablehlo.constant dense<4.471500e-02> : tensor<16x1024x4096xf32>
    %x2      = stablehlo.multiply %input, %input : tensor<16x1024x4096xf32>
    %x3      = stablehlo.multiply %x2, %input : tensor<16x1024x4096xf32>
    %cx3     = stablehlo.multiply %coeff, %x3 : tensor<16x1024x4096xf32>
    %inner   = stablehlo.add %input, %cx3 : tensor<16x1024x4096xf32>
    %targ    = stablehlo.multiply %sqrt2pi, %inner : tensor<16x1024x4096xf32>
    %tv      = stablehlo.tanh %targ : tensor<16x1024x4096xf32>
    %one_tv  = stablehlo.add %one, %tv : tensor<16x1024x4096xf32>
    %hx      = stablehlo.multiply %half, %input : tensor<16x1024x4096xf32>
    %result  = stablehlo.multiply %hx, %one_tv : tensor<16x1024x4096xf32>
    return %result : tensor<16x1024x4096xf32>
  }
}
