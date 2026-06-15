module @f_19_vit_32x197_1000x768_32x197x768 {
  flow.executable private @f_19_vit_32x197_1000x768_32x197x768_dispatch_0 {
    flow.executable.export public @f_19_vit_32x197_1000x768_32x197x768_dispatch_0_generic_32x197x768_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_vit_32x197_1000x768_32x197x768_dispatch_0_generic_32x197x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x197xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1000x768xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<32x197x768xf32>>) {
        %c0 = arith.constant 0 : index
        %c999 = arith.constant 999 : index
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 197], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x197xi32>> -> tensor<32x197xi32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1000x768xf32>> -> tensor<1000x768xf32>
        %2 = tensor.empty() : tensor<32x197x768xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%2 : tensor<32x197x768xf32>) {
        ^bb0(%out: f32):
          %4 = linalg.index 0 : index
          %5 = linalg.index 1 : index
          %6 = linalg.index 2 : index
          %extracted = tensor.extract %0[%4, %5] : tensor<32x197xi32>
          %7 = arith.index_cast %extracted : i32 to index
          %8 = arith.maxsi %7, %c0 : index
          %9 = arith.minsi %8, %c999 : index
          %extracted_0 = tensor.extract %1[%9, %6] : tensor<1000x768xf32>
          linalg.yield %extracted_0 : f32
        } -> tensor<32x197x768xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0], sizes = [32, 197, 768], strides = [1, 1, 1] : tensor<32x197x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x197x768xf32>>
        return
      }
    }
  }
  func.func @f_19_vit_32x197_1000x768_32x197x768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<1000x768xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x197xi32>
    %2 = flow.dispatch @f_19_vit_32x197_1000x768_32x197x768_dispatch_0::@f_19_vit_32x197_1000x768_32x197x768_dispatch_0_generic_32x197x768_f32(%1, %0) : (tensor<32x197xi32>, tensor<1000x768xf32>) -> tensor<32x197x768xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<32x197x768xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}