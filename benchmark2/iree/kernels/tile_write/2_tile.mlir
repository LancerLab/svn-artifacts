module {
  func.func @f_tw(%tiles: tensor<?x2xf32>) -> tensor<2x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cNR = arith.constant 2 : index
    %c0f = arith.constant 0.0 : f32
    %nt = tensor.dim %tiles, %c0 : tensor<?x2xf32>
    %init = tensor.empty() : tensor<2x2xf32>
    %z = linalg.fill ins(%c0f : f32) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    %out = scf.for %j = %c0 to %nt step %c1 iter_args(%acc = %z) -> (tensor<2x2xf32>) {
      %tile = tensor.extract_slice %tiles[%j, 0][1, 2][1, 1] : tensor<?x2xf32> to tensor<1x2xf32>
      %rowm = arith.remui %j, %cNR : index
      %acc2 = tensor.insert_slice %tile into %acc[%rowm, 0][1, 2][1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      scf.yield %acc2 : tensor<2x2xf32>
    }
    return %out : tensor<2x2xf32>
  }
}
