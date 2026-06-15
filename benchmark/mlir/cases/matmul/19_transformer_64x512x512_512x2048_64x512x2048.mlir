module {
  func.func @f_19_transformer_64x512x512_512x2048_64x512x2048(%lhs: tensor<64x512x512xf32>, %rhs: tensor<512x2048xf32>) -> tensor<64x512x2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero_f = arith.constant 0.0 : f32
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<64x512x512xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<64x512x512xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<64x512x512xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<512x2048xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<512x2048xf32>
    %eq0 = arith.cmpi eq, %lhs_d2, %rhs_d0 : index
    cf.assert %eq0, "lhs.dim(2)==rhs.dim(0)"
    %out = tensor.empty() : tensor<64x512x2048xf32>
    %t_res_bs = scf.for %bs = %c0 to %lhs_d0 step %c1 iter_args(%t_arg_bs = %out) -> (tensor<64x512x2048xf32>) {
    %t_res_i = scf.for %i = %c0 to %lhs_d1 step %c1 iter_args(%t_arg_i = %t_arg_bs) -> (tensor<64x512x2048xf32>) {
    %t_res_j = scf.for %j = %c0 to %rhs_d1 step %c1 iter_args(%t_arg_j = %t_arg_i) -> (tensor<64x512x2048xf32>) {
    %acc = scf.for %k = %c0 to %lhs_d2 step %c1 iter_args(%s = %zero_f) -> (f32) {
    %a   = tensor.extract %lhs[%bs, %i, %k] : tensor<64x512x512xf32>
    %b   = tensor.extract %rhs[%k, %j] : tensor<512x2048xf32>
    %p   = arith.mulf %a, %b : f32
    %ns  = arith.addf %s, %p : f32
    scf.yield %ns : f32
    }
    %t_new = tensor.insert %acc into %t_arg_j[%bs, %i, %j] : tensor<64x512x2048xf32>
    scf.yield %t_new : tensor<64x512x2048xf32>
    }
    scf.yield %t_res_j : tensor<64x512x2048xf32>
    }
    scf.yield %t_res_i : tensor<64x512x2048xf32>
    }
    return %t_res_bs : tensor<64x512x2048xf32>
  }
}
