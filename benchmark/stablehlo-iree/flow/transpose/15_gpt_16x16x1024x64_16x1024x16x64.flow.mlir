module @f_15_gpt_16x16x1024x64_16x1024x16x64 {
  flow.executable private @f_15_gpt_16x16x1024x64_16x1024x16x64_dispatch_0 {
    flow.executable.export public @f_15_gpt_16x16x1024x64_16x1024x16x64_dispatch_0_generic_16x1024x16x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_15_gpt_16x16x1024x64_16x1024x16x64_dispatch_0_generic_16x1024x16x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x16x1024x64xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<16x1024x16x64xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 16, 1024, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16x1024x64xf32>> -> tensor<16x16x1024x64xf32>
        %1 = tensor.empty() : tensor<16x1024x16x64xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<16x16x1024x64xf32>) outs(%1 : tensor<16x1024x16x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x1024x16x64xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0, 0], sizes = [16, 1024, 16, 64], strides = [1, 1, 1, 1] : tensor<16x1024x16x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x16x64xf32>>
        return
      }
    }
  }
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x16x1024x64xf32>
    %1 = flow.dispatch @f_15_gpt_16x16x1024x64_16x1024x16x64_dispatch_0::@f_15_gpt_16x16x1024x64_16x1024x16x64_dispatch_0_generic_16x1024x16x64_f32(%0) : (tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<16x1024x16x64xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}