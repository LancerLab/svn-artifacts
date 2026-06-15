module {
  func.func @f_17_mobilenet_128x96x112x112_112x112_112x112(%input: tensor<128x96x112x112xf32>, %gamma: tensor<112x112xf32>, %beta: tensor<112x112xf32>) -> tensor<128x96x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x96x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x96x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x96x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x96x112x112xf32>
    %gamma_d0 = tensor.dim %gamma, %c0 : tensor<112x112xf32>
    %gamma_d1 = tensor.dim %gamma, %c1 : tensor<112x112xf32>
    %beta_d0 = tensor.dim %beta, %c0 : tensor<112x112xf32>
    %beta_d1 = tensor.dim %beta, %c1 : tensor<112x112xf32>
    %out = tensor.empty() : tensor<128x96x112x112xf32>
    %sum_t = tensor.empty() : tensor<128x96xf32>
    %sumsq_t = tensor.empty() : tensor<128x96xf32>
    %zero_f   = arith.constant 0.0 : f32
    %sum_init = linalg.fill ins(%zero_f : f32) outs(%sum_t : tensor<128x96xf32>) -> tensor<128x96xf32>
    %sumsq_init = linalg.fill ins(%zero_f : f32) outs(%sumsq_t : tensor<128x96xf32>) -> tensor<128x96xf32>
    %nsz = arith.constant 12544.0 : f32
    %sum_result, %sumsq_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%input : tensor<128x96x112x112xf32>) outs(%sum_init, %sumsq_init : tensor<128x96xf32>, tensor<128x96xf32>) {
    ^bb0(%in: f32, %sum_acc: f32, %sq_acc: f32):
      %sq    = arith.mulf %in, %in : f32
      %nsum  = arith.addf %sum_acc, %in : f32
      %nsq   = arith.addf %sq_acc, %sq : f32
      linalg.yield %nsum, %nsq : f32, f32
    } -> (tensor<128x96xf32>, tensor<128x96xf32>)
    %eps = arith.constant 1.0e-05 : f32
    %mean_t = tensor.empty() : tensor<128x96xf32>
    %istd_t = tensor.empty() : tensor<128x96xf32>
    %mean_dummy = linalg.fill ins(%zero_f : f32) outs(%mean_t : tensor<128x96xf32>) -> tensor<128x96xf32>
    %istd_dummy = linalg.fill ins(%zero_f : f32) outs(%istd_t : tensor<128x96xf32>) -> tensor<128x96xf32>
    %mean_result, %istd_result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sum_result, %sumsq_result : tensor<128x96xf32>, tensor<128x96xf32>) outs(%mean_dummy, %istd_dummy : tensor<128x96xf32>, tensor<128x96xf32>) {
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
    } -> (tensor<128x96xf32>, tensor<128x96xf32>)
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %mean_result, %istd_result, %gamma, %beta : tensor<128x96x112x112xf32>, tensor<128x96xf32>, tensor<128x96xf32>, tensor<112x112xf32>, tensor<112x112xf32>) outs(%out : tensor<128x96x112x112xf32>) {
    ^bb0(%x: f32, %mn: f32, %is: f32, %g: f32, %b: f32, %init: f32):
      %cent   = arith.subf %x, %mn : f32
      %normed = arith.mulf %cent, %is : f32
      %scaled = arith.mulf %normed, %g : f32
      %res    = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<128x96x112x112xf32>
    return %result : tensor<128x96x112x112xf32>
  }
}
