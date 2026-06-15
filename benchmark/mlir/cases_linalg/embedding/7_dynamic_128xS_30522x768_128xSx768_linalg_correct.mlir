module {
  func.func @f_7_dynamic_128xS_30522x768_128xSx768_correct(%indices: tensor<128x?xi64>, %weights: tensor<30522x768xf32>) -> tensor<128x?x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %indices, %ci1 : tensor<128x?xi64>
    %init = tensor.empty(%od1) : tensor<128x?x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x?x768xf32>) -> tensor<128x?x768xf32>
    return %fill : tensor<128x?x768xf32>
  }
}
