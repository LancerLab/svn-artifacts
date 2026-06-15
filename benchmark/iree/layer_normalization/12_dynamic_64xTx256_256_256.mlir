module {
  func.func @f_12_dynamic_64xTx256_256_256(%input: tensor<64x?x256xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<64x?x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x?x256xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x?x256xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x?x256xf32>
    %gamma_d0 = tensor.dim %gamma, %c0 : tensor<256xf32>
    %beta_d0 = tensor.dim %beta, %c0 : tensor<256xf32>
    %out = tensor.empty(%input_d1) : tensor<64x?x256xf32>
    %sum_t = tensor.empty(%input_d1) : tensor<64x?xf32>
    %sumsq_t = tensor.empty(%input_d1) : tensor<64x?xf32>
    %zero_f   = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<64x?xf32>) -> tensor<64x?xf32>
    %sumsq_init = linalg.fill ins(%zero_f : f32) outs(%sumsq_t : tensor<64x?xf32>) -> tensor<64x?xf32>
    %nsz = arith.constant 256.0 : f32
    %sum_result, %sumsq_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input : tensor<64x?x256xf32>) outs(%sum_init, %sumsq_init : tensor<64x?xf32>, tensor<64x?xf32>) {
    ^bb0(%in: f32, %sum_acc: f32, %sq_acc: f32):
      %sq    = arith.mulf %in, %in : f32
      %nsum  = arith.addf %sum_acc, %in : f32
      %nsq   = arith.addf %sq_acc, %sq : f32
      linalg.yield %nsum, %nsq : f32, f32
    } -> (tensor<64x?xf32>, tensor<64x?xf32>)
    %eps = arith.constant 1.0e-05 : f32
    %mean_t = tensor.empty(%input_d1) : tensor<64x?xf32>
    %istd_t = tensor.empty(%input_d1) : tensor<64x?xf32>
    %mean_dummy = linalg.fill ins(%zero_f : f32) outs(%mean_t : tensor<64x?xf32>) -> tensor<64x?xf32>
    %istd_dummy = linalg.fill ins(%zero_f : f32) outs(%istd_t : tensor<64x?xf32>) -> tensor<64x?xf32>
    %mean_result, %istd_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum_result, %sumsq_result : tensor<64x?xf32>, tensor<64x?xf32>) outs(%mean_dummy, %istd_dummy : tensor<64x?xf32>, tensor<64x?xf32>) {
    ^bb0(%sm: f32, %sq: f32, %m_init: f32, %s_init: f32):
      %one_f  = arith.constant 1.0 : f32
      %mean   = arith.divf %sm, %nsz : f32
      %msq    = arith.mulf %mean, %mean : f32
      %esq    = arith.divf %sq, %nsz : f32
      %var    = arith.subf %esq, %msq : f32
      %vep    = arith.addf %var, %eps : f32
      %std    = math.sqrt %vep : f32
      %istd   = arith.divf %one_f, %std : f32
      linalg.yield %mean, %istd : f32, f32
    } -> (tensor<64x?xf32>, tensor<64x?xf32>)
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %mean_result, %istd_result, %gamma, %beta : tensor<64x?x256xf32>, tensor<64x?xf32>, tensor<64x?xf32>, tensor<256xf32>, tensor<256xf32>) outs(%out : tensor<64x?x256xf32>) {
    ^bb0(%x: f32, %mn: f32, %is: f32, %g: f32, %b: f32, %init: f32):
      %cent   = arith.subf %x, %mn : f32
      %normed = arith.mulf %cent, %is : f32
      %scaled = arith.mulf %normed, %g : f32
      %res    = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x?x256xf32>
    return %result : tensor<64x?x256xf32>
  }
}
