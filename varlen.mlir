module @jit_doc_offset_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x4x64xf16>, %arg1: tensor<1024x4x64xf16>, %arg2: tensor<1024x4x64xf16>, %arg3: tensor<9xi32>) -> (tensor<1024x4x64xf16> {jax.result_info = "result"}) {
    %0 = call @doc_offset_attention(%arg0, %arg1, %arg2, %arg3) : (tensor<1024x4x64xf16>, tensor<1024x4x64xf16>, tensor<1024x4x64xf16>, tensor<9xi32>) -> tensor<1024x4x64xf16>
    return %0 : tensor<1024x4x64xf16>
  }
  func.func private @doc_offset_attention(%arg0: tensor<1024x4x64xf16>, %arg1: tensor<1024x4x64xf16>, %arg2: tensor<1024x4x64xf16>, %arg3: tensor<9xi32>) -> tensor<1024x4x64xf16> {
    %0 = stablehlo.slice %arg3 [0:8] : (tensor<9xi32>) -> tensor<8xi32>
    %1 = stablehlo.slice %arg3 [1:9] : (tensor<9xi32>) -> tensor<8xi32>
    %2 = stablehlo.subtract %1, %0 : tensor<8xi32>
    %3 = stablehlo.iota dim = 0 : tensor<128xi32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %5 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %7 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %8 = stablehlo.add %6, %7 : tensor<8x128xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %10 = "stablehlo.gather"(%arg0, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %11 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %12 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %15 = stablehlo.add %13, %14 : tensor<8x128xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %17 = "stablehlo.gather"(%arg1, %16) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %18 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %19 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %21 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %22 = stablehlo.add %20, %21 : tensor<8x128xi32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %24 = "stablehlo.gather"(%arg2, %23) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4, 64>}> : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>) -> tensor<8x128x4x64xf16>
    %25 = stablehlo.dot_general %10, %17, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x128x4x64xf16>, tensor<8x128x4x64xf16>) -> tensor<8x4x128x128xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x4x128x128xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<8x4x128x128xf32>
    %28 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %29 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %30 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %31 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %32 = stablehlo.compare LT, %30, %31, SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1>
    %33 = stablehlo.reshape %32 : (tensor<8x128xi1>) -> tensor<8x1x1x128xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %34 = call @_where(%33, %27, %cst_0) : (tensor<8x1x1x128xi1>, tensor<8x4x128x128xf32>, tensor<f32>) -> tensor<8x4x128x128xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %35 = stablehlo.reduce(%34 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<8x4x128x128xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x4x128xf32>
    %37 = stablehlo.maximum %36, %35 : tensor<8x4x128xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<8x4x128xf32>) -> tensor<8x4x128x1xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2, 3] : (tensor<8x4x128x1xf32>) -> tensor<8x4x128x128xf32>
    %40 = stablehlo.subtract %34, %39 : tensor<8x4x128x128xf32>
    %41 = stablehlo.exponential %40 : tensor<8x4x128x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.reduce(%41 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<8x4x128x128xf32>, tensor<f32>) -> tensor<8x4x128xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2] : (tensor<8x4x128xf32>) -> tensor<8x4x128x1xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2, 3] : (tensor<8x4x128x1xf32>) -> tensor<8x4x128x128xf32>
    %45 = stablehlo.divide %41, %44 : tensor<8x4x128x128xf32>
    %46 = stablehlo.convert %24 : (tensor<8x128x4x64xf16>) -> tensor<8x128x4x64xf32>
    %47 = stablehlo.convert %45 : tensor<8x4x128x128xf32>
    %48 = stablehlo.dot_general %46, %47, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3], precision = [DEFAULT, DEFAULT] : (tensor<8x128x4x64xf32>, tensor<8x4x128x128xf32>) -> tensor<8x4x64x128xf32>
    %49 = stablehlo.transpose %48, dims = [0, 3, 1, 2] : (tensor<8x4x64x128xf32>) -> tensor<8x128x4x64xf32>
    %50 = stablehlo.convert %49 : (tensor<8x128x4x64xf32>) -> tensor<8x128x4x64xf16>
    %51 = stablehlo.iota dim = 0 : tensor<8xi32>
    %52 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %53 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %54 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %55 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %56 = stablehlo.add %54, %55 : tensor<8x128xi32>
    %57 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %58 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %59 = stablehlo.broadcast_in_dim %57, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %60 = stablehlo.broadcast_in_dim %58, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %61 = stablehlo.compare GT, %59, %60, SIGNED : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi1>
    %62 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %c = stablehlo.constant dense<128> : tensor<i32>
    %63 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %64 = stablehlo.multiply %62, %63 : tensor<8x1xi32>
    %c_4 = stablehlo.constant dense<1024> : tensor<i32>
    %65 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<8x1xi32>
    %66 = stablehlo.add %65, %64 : tensor<8x1xi32>
    %67 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<128xi32>) -> tensor<1x128xi32>
    %68 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<8x1xi32>) -> tensor<8x128xi32>
    %69 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<8x128xi32>
    %70 = stablehlo.add %68, %69 : tensor<8x128xi32>
    %71 = call @_where_0(%61, %56, %70) : (tensor<8x128xi1>, tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %72 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f16>) -> tensor<1024x4x64xf16>
    %73 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<8x128xi32>) -> tensor<8x128x1xi32>
    %74 = "stablehlo.scatter"(%72, %73, %50) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f16>, %arg5: tensor<f16>):
      stablehlo.return %arg5 : tensor<f16>
    }) : (tensor<1024x4x64xf16>, tensor<8x128x1xi32>, tensor<8x128x4x64xf16>) -> tensor<1024x4x64xf16>
    return %74 : tensor<1024x4x64xf16>
  }
  func.func private @_where(%arg0: tensor<8x1x1x128xi1>, %arg1: tensor<8x4x128x128xf32>, %arg2: tensor<f32>) -> tensor<8x4x128x128xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<8x1x1x128xi1>) -> tensor<8x4x128x128xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4x128x128xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2, 3] : (tensor<4x128x128xf32>) -> tensor<8x4x128x128xf32>
    %4 = stablehlo.select %1, %arg1, %3 : tensor<8x4x128x128xi1>, tensor<8x4x128x128xf32>
    return %4 : tensor<8x4x128x128xf32>
  }
  func.func private @_where_0(%arg0: tensor<8x128xi1>, %arg1: tensor<8x128xi32>, %arg2: tensor<8x128xi32>) -> tensor<8x128xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<8x128xi1>, tensor<8x128xi32>
    return %0 : tensor<8x128xi32>
  }
}

