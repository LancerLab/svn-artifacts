module @f_16_general_64x256_20000x512_64x256x512 {
  flow.executable private @f_16_general_64x256_20000x512_64x256x512_dispatch_0 {
    flow.executable.export public @f_16_general_64x256_20000x512_64x256x512_dispatch_0_generic_64x256x512_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_16_general_64x256_20000x512_64x256x512_dispatch_0_generic_64x256x512_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x256xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<20000x512xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<64x256x512xf32>>) {
        %c0 = arith.constant 0 : index
        %c19999 = arith.constant 19999 : index
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xi32>> -> tensor<64x256xi32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [20000, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<20000x512xf32>> -> tensor<20000x512xf32>
        %2 = tensor.empty() : tensor<64x256x512xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%2 : tensor<64x256x512xf32>) {
        ^bb0(%out: f32):
          %4 = linalg.index 0 : index
          %5 = linalg.index 1 : index
          %6 = linalg.index 2 : index
          %extracted = tensor.extract %0[%4, %5] : tensor<64x256xi32>
          %7 = arith.index_cast %extracted : i32 to index
          %8 = arith.maxsi %7, %c0 : index
          %9 = arith.minsi %8, %c19999 : index
          %extracted_0 = tensor.extract %1[%9, %6] : tensor<20000x512xf32>
          linalg.yield %extracted_0 : f32
        } -> tensor<64x256x512xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0], sizes = [64, 256, 512], strides = [1, 1, 1] : tensor<64x256x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256x512xf32>>
        return
      }
    }
  }
  func.func @f_16_general_64x256_20000x512_64x256x512(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<20000x512xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<64x256xi32>
    %2 = flow.dispatch @f_16_general_64x256_20000x512_64x256x512_dispatch_0::@f_16_general_64x256_20000x512_64x256x512_dispatch_0_generic_64x256x512_f32(%1, %0) : (tensor<64x256xi32>, tensor<20000x512xf32>) -> tensor<64x256x512xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<64x256x512xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}