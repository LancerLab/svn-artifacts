module @f_1_bert_32x512x768_32x512x768 {
  flow.executable private @f_1_bert_32x512x768_32x512x768_dispatch_0 {
    flow.executable.export public @f_1_bert_32x512x768_32x512x768_dispatch_0_generic_32x512x768_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_1_bert_32x512x768_32x512x768_dispatch_0_generic_32x512x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>) {
        %cst = arith.constant -3.402820e+38 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>> -> tensor<32x512x768xf32>
        %1 = tensor.empty() : tensor<32x512x768xf32>
        %2 = tensor.empty() : tensor<32x512xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<32x512x768xf32>) outs(%3 : tensor<32x512xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.maxf %out, %in : f32
          linalg.yield %6 : f32
        } -> tensor<32x512xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %4 : tensor<32x512x768xf32>, tensor<32x512xf32>) outs(%1 : tensor<32x512x768xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %6 = arith.subf %in, %in_0 : f32
          %7 = math.exp %6 : f32
          linalg.yield %7 : f32
        } -> tensor<32x512x768xf32>
        flow.dispatch.tensor.store %5, %arg1, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : tensor<32x512x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>
        return
      }
    }
  }
  flow.executable private @f_1_bert_32x512x768_32x512x768_dispatch_1 {
    flow.executable.export public @f_1_bert_32x512x768_32x512x768_dispatch_1_generic_32x512x768_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_1_bert_32x512x768_32x512x768_dispatch_1_generic_32x512x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>> -> tensor<32x512x768xf32>
        %1 = tensor.empty() : tensor<32x512x768xf32>
        %2 = tensor.empty() : tensor<32x512xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<32x512x768xf32>) outs(%3 : tensor<32x512xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.addf %out, %in : f32
          linalg.yield %6 : f32
        } -> tensor<32x512xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %4 : tensor<32x512x768xf32>, tensor<32x512xf32>) outs(%1 : tensor<32x512x768xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %6 = arith.divf %in, %in_0 : f32
          linalg.yield %6 : f32
        } -> tensor<32x512x768xf32>
        flow.dispatch.tensor.store %5, %arg1, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : tensor<32x512x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>
        return
      }
    }
  }
  func.func @f_1_bert_32x512x768_32x512x768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x768xf32>
    %1 = flow.dispatch @f_1_bert_32x512x768_32x512x768_dispatch_0::@f_1_bert_32x512x768_32x512x768_dispatch_0_generic_32x512x768_f32(%0) : (tensor<32x512x768xf32>) -> tensor<32x512x768xf32>
    %2 = flow.dispatch @f_1_bert_32x512x768_32x512x768_dispatch_1::@f_1_bert_32x512x768_32x512x768_dispatch_1_generic_32x512x768_f32(%1) : (tensor<32x512x768xf32>) -> tensor<32x512x768xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<32x512x768xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}