module {
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%in0: tensor<32x512x64xf32>, %in1: tensor<32x512x64xf32>, %in2: tensor<32x512x64xf32>, %in3: tensor<32x512x64xf32>) -> tensor<32x512x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = tensor.empty() : tensor<32x512x256xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][32, 512, 64][1, 1, 1] : tensor<32x512x64xf32> into tensor<32x512x256xf32>
    %coff64 = arith.constant 64 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %coff64][32, 512, 64][1, 1, 1] : tensor<32x512x64xf32> into tensor<32x512x256xf32>
    %coff128 = arith.constant 128 : index
    %ins2 = tensor.insert_slice %in2 into %ins1[%c0, %c0, %coff128][32, 512, 64][1, 1, 1] : tensor<32x512x64xf32> into tensor<32x512x256xf32>
    %coff192 = arith.constant 192 : index
    %ins3 = tensor.insert_slice %in3 into %ins2[%c0, %c0, %coff192][32, 512, 64][1, 1, 1] : tensor<32x512x64xf32> into tensor<32x512x256xf32>
    return %ins3 : tensor<32x512x256xf32>
  }
}
