module {
func.func @initTensor() -> tensor<?x?xf32> {
  %0 = arith.constant dense<[[1.0, 2.0, 3.0],
                             [4.,  5.,  6.],
                             [7.,  8.,  9.]]> : tensor<3x3xf32>
  // this cast makes sure sizes and strides are preserved in
  // generated code
  %ret = tensor.cast %0 : tensor<3x3xf32> to tensor<?x?xf32>
  return %ret: tensor<?x?xf32>
}

func.func @mm(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // https://github.com/llvm/llvm-project/blob/806dea46be4c49dc587b98dab5e4d9d242a6abdb/mlir/test/Integration/Dialect/Linalg/CPU/test-tensor-matmul.mlir#L23
  %ret0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %ret0: tensor<?x?xf32>
}
}
