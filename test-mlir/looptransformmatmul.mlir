func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32> {
  %index_0 = arith.constant 0: index
  %index_1 = arith.constant 1: index
  %I = memref.dim %A, %index_0 : memref<?x?xf32>
  %J = memref.dim %B, %index_1 : memref<?x?xf32>
  %K = memref.dim %A, %index_1 : memref<?x?xf32>
  scf.for %k = %index_0 to %K step %index_1{
    scf.for %j = %index_0 to %J step %index_1 {
     scf.for %i = %index_0 to %I step %index_1 {
        %a_val = memref.load %A[%i, %k] : memref<?x?xf32>
        %b_val = memref.load %B[%k, %j] : memref<?x?xf32>
        %c_val = memref.load %C[%i, %j] : memref<?x?xf32>
        %product = arith.mulf %a_val, %b_val : f32
        %sum1 = arith.addf %c_val, %product : f32
        memref.store %sum1, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return %C: memref<?x?xf32>
}

