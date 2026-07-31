module {
  func.func @attention(%arg0: tensor<1x4x1024x64xf16>, %arg1: tensor<1x4x1024x64xf16>, %arg2: tensor<1x4x1024x64xf16>) -> tensor<1x4x1024x64xf16> {
    %alloc = memref.alloc() : memref<1x4x1024x64xf16>
    %0 = bufferization.to_buffer %arg0 read_only : tensor<1x4x1024x64xf16> to memref<1x4x1024x64xf16>
    %1 = bufferization.to_buffer %arg1 read_only : tensor<1x4x1024x64xf16> to memref<1x4x1024x64xf16>
    %2 = bufferization.to_buffer %arg2 read_only : tensor<1x4x1024x64xf16> to memref<1x4x1024x64xf16>
    htile.launch_func @attention_kernel(%0, %1, %2, %alloc) {program_bounds = array<i64: 4, 8>} : memref<1x4x1024x64xf16>, memref<1x4x1024x64xf16>, memref<1x4x1024x64xf16>, memref<1x4x1024x64xf16>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<1x4x1024x64xf16> to tensor<1x4x1024x64xf16>
    return %3 : tensor<1x4x1024x64xf16>
  }
  htile.kernel @attention_kernel(%arg0 : memref<1x4x1024x64xf16>, %arg1 : memref<1x4x1024x64xf16>, %arg2 : memref<1x4x1024x64xf16>, %arg3 : memref<1x4x1024x64xf16>) attributes {program_bounds = array<i64: 4, 8>} {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.180336878 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = htile.program_id 0
    %1 = htile.program_id 1
    %2 = arith.muli %1, %c128 overflow<nsw> : index
    %3 = htile.full %cst_1 : f32 -> tensor<128xf32>
    %4 = htile.full %cst_0 : f32 -> tensor<128xf32>
    %5 = htile.full %cst_0 : f32 -> tensor<128x64xf32>
    %6 = arith.muli %1, %c2 overflow<nsw> : index
    %7 = arith.maxsi %6, %c0 : index
    %8 = arith.minsi %7, %c16 : index
    %9:3 = scf.for %arg4 = %c0 to %8 step %c1 iter_args(%arg5 = %3, %arg6 = %4, %arg7 = %5) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32>){
      %17 = arith.muli %arg4, %c64 overflow<nsw> : index
      %18 = htile.load %arg0[%c0, %0, %2, %c0] : memref<1x4x1024x64xf16> -> tensor<128x64xf16>
      %19 = htile.load %arg1[%c0, %0, %17, %c0] {dimension_order = array<i64: 1, 0>} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
      %20 = htile.dot %18, %19, %5 : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
      %21 = htile.full %cst : f32 -> tensor<128x64xf32>
      %22 = arith.mulf %20, %21 : tensor<128x64xf32>
      %23 = htile.reduce %22 axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
      %24 = arith.maximumf %arg5, %23 : tensor<128xf32>
      %25 = htile.broadcast %24 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %26 = arith.subf %22, %25 : tensor<128x64xf32>
      %27 = math.exp2 %26 : tensor<128x64xf32>
      %28 = arith.subf %arg5, %24 : tensor<128xf32>
      %29 = math.exp2 %28 : tensor<128xf32>
      %30 = arith.mulf %arg6, %29 : tensor<128xf32>
      %31 = htile.reduce %27 axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
      %32 = arith.addf %30, %31 : tensor<128xf32>
      %33 = arith.truncf %27 : tensor<128x64xf32> to tensor<128x64xf16>
      %34 = htile.broadcast %arg5 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %35 = htile.broadcast %24 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %36 = arith.subf %34, %35 : tensor<128x64xf32>
      %37 = math.exp2 %36 : tensor<128x64xf32>
      %38 = arith.mulf %arg7, %37 : tensor<128x64xf32>
      %39 = htile.load %arg2[%c0, %0, %17, %c0] : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
      %40 = htile.dot %33, %39, %38 : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
      scf.yield %24, %32, %40 : tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32>
    }
    %10 = arith.addi %6, %c2 : index
    %11 = arith.maxsi %10, %c0 : index
    %12 = arith.minsi %11, %c16 : index
    %13:3 = scf.for %arg4 = %8 to %12 step %c1 iter_args(%arg5 = %9#0, %arg6 = %9#1, %arg7 = %9#2) -> (tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32>) {
      %17 = arith.muli %arg4, %c64 overflow<nsw> : index
      %18 = htile.load %arg0[%c0, %0, %2, %c0] : memref<1x4x1024x64xf16> -> tensor<128x64xf16>
      %19 = htile.load %arg1[%c0, %0, %17, %c0] {dimension_order = array<i64: 1, 0>} : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
      %20 = htile.dot %18, %19, %5 : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
      %21 = htile.full %cst : f32 -> tensor<128x64xf32>
      %22 = arith.mulf %20, %21 : tensor<128x64xf32>
      %23 = htile.arange %c0 to %c128 : tensor<128xindex>
      %24 = htile.broadcast %23 dimensions = [1] : tensor<128xindex> -> tensor<128x64xindex>
      %25 = htile.full %1 : index -> tensor<128x64xindex>
      %26 = htile.full %c128 : index -> tensor<128x64xindex>
      %27 = arith.muli %25, %26 overflow<nsw> : tensor<128x64xindex>
      %28 = arith.addi %27, %24 : tensor<128x64xindex>
      %29 = htile.arange %c0 to %c64 : tensor<64xindex>
      %30 = htile.broadcast %29 dimensions = [0] : tensor<64xindex> -> tensor<128x64xindex>
      %31 = htile.full %arg4 : index -> tensor<128x64xindex>
      %32 = htile.full %c64 : index -> tensor<128x64xindex>
      %33 = arith.muli %31, %32 overflow<nsw> : tensor<128x64xindex>
      %34 = arith.addi %33, %30 : tensor<128x64xindex>
      %35 = arith.index_cast %34 : tensor<128x64xindex> to tensor<128x64xi64>
      %36 = arith.index_cast %28 : tensor<128x64xindex> to tensor<128x64xi64>
      %37 = arith.cmpi sle, %35, %36 : tensor<128x64xi64>
      %38 = htile.full %cst_1 : f32 -> tensor<128x64xf32>
      %39 = arith.select %37, %22, %38 : tensor<128x64xi1>, tensor<128x64xf32>
      %40 = htile.reduce %39 axis 1 kind "max" : tensor<128x64xf32> -> tensor<128xf32>
      %41 = arith.maximumf %arg5, %40 : tensor<128xf32>
      %42 = htile.broadcast %41 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %43 = arith.subf %39, %42 : tensor<128x64xf32>
      %44 = math.exp2 %43 : tensor<128x64xf32>
      %45 = arith.subf %arg5, %41 : tensor<128xf32>
      %46 = math.exp2 %45 : tensor<128xf32>
      %47 = arith.mulf %arg6, %46 : tensor<128xf32>
      %48 = htile.reduce %44 axis 1 kind "sum" : tensor<128x64xf32> -> tensor<128xf32>
      %49 = arith.addf %47, %48 : tensor<128xf32>
      %50 = arith.truncf %44 : tensor<128x64xf32> to tensor<128x64xf16>
      %51 = htile.broadcast %arg5 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %52 = htile.broadcast %41 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
      %53 = arith.subf %51, %52 : tensor<128x64xf32>
      %54 = math.exp2 %53 : tensor<128x64xf32>
      %55 = arith.mulf %arg7, %54 : tensor<128x64xf32>
      %56 = htile.load %arg2[%c0, %0, %17, %c0] : memref<1x4x1024x64xf16> -> tensor<64x64xf16>
      %57 = htile.dot %50, %56, %55 : tensor<128x64xf16>, tensor<64x64xf16>, tensor<128x64xf32> -> tensor<128x64xf32>
      scf.yield %41, %49, %57 : tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32>
    }
    %14 = htile.broadcast %13#1 dimensions = [1] : tensor<128xf32> -> tensor<128x64xf32>
    %15 = arith.divf %13#2, %14 : tensor<128x64xf32>
    %16 = arith.truncf %15 : tensor<128x64xf32> to tensor<128x64xf16>
    htile.store %16, %arg3[%c0, %0, %2, %c0] : tensor<128x64xf16>, memref<1x4x1024x64xf16>
    htile.return
  }
}
