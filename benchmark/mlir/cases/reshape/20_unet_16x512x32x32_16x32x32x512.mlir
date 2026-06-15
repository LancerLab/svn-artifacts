module {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%input: tensor<16x512x32x32xf32>) -> tensor<16x32x32x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x32x32xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x32x32xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x32x32xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x512x32x32xf32>
    %out = tensor.empty() : tensor<16x32x32x512xf32>
    %t_res_rsi0 = scf.for %rsi0 = %c0 to %input_d0 step %c1 iter_args(%t_rs_rsi0 = %out) -> (tensor<16x32x32x512xf32>) {
    %t_res_rsi1 = scf.for %rsi1 = %c0 to %input_d2 step %c1 iter_args(%t_rs_rsi1 = %t_rs_rsi0) -> (tensor<16x32x32x512xf32>) {
    %t_res_rsi2 = scf.for %rsi2 = %c0 to %input_d3 step %c1 iter_args(%t_rs_rsi2 = %t_rs_rsi1) -> (tensor<16x32x32x512xf32>) {
    %t_res_rsi3 = scf.for %rsi3 = %c0 to %input_d1 step %c1 iter_args(%t_rs_rsi3 = %t_rs_rsi2) -> (tensor<16x32x32x512xf32>) {
    %rsv = tensor.extract %input[%rsi0, %rsi3, %rsi1, %rsi2] : tensor<16x512x32x32xf32>
    %t_ins = tensor.insert %rsv into %t_rs_rsi3[%rsi0, %rsi1, %rsi2, %rsi3] : tensor<16x32x32x512xf32>
    scf.yield %t_ins : tensor<16x32x32x512xf32>
    }
    scf.yield %t_res_rsi3 : tensor<16x32x32x512xf32>
    }
    scf.yield %t_res_rsi2 : tensor<16x32x32x512xf32>
    }
    scf.yield %t_res_rsi1 : tensor<16x32x32x512xf32>
    }
    return %t_res_rsi0 : tensor<16x32x32x512xf32>
  }
}
