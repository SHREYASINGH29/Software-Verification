func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32> {
    %index_0 = arith.constant 0: index
    %index_1 = arith.constant 1: index
    %I = memref.dim %A, %index_0 : memref<?x?xf32>
    %J = memref.dim %B, %index_1 : memref<?x?xf32>
    %K = memref.dim %A, %index_1 : memref<?x?xf32>
    %T = arith.constant 4: index  // Example tile size
    
    // Compute number of tiles for k loop
    %num_tiles_k = arith.divui %K, %T : index
    %len =  arith.muli %num_tiles_k, %T : index
    %has_partial_tile_k = arith.cmpi ne, %K, %len : index
    %extra_k = arith.select %has_partial_tile_k, %index_1, %index_0 : index
    %total_tiles_k = arith.addi %num_tiles_k, %extra_k : index
    
    scf.for %i = %index_0 to %I step %index_1 {
      scf.for %j = %index_0 to %J step %index_1 {
        scf.for %tile_k = %index_0 to %total_tiles_k step %index_1 {
          %start_k = arith.muli %tile_k, %T : index
          %end_k = arith.addi %start_k, %T : index
          // Bound check for the last tile in k
          %tile_end_k = arith.minui %end_k, %K : index
        
          scf.for %k = %start_k to %tile_end_k step %index_1 {
            %a_val = memref.load %A[%i, %k] : memref<?x?xf32>
            %b_val = memref.load %B[%k, %j] : memref<?x?xf32>
            %c_val = memref.load %C[%i, %j] : memref<?x?xf32>
            %product = arith.mulf %a_val, %b_val : f32 
            %sum1 = arith.addf %c_val, %product : f32 
            memref.store %sum1, %C[%i, %j] : memref<?x?xf32>
          }
        }
      }   
    }   
    return %C : memref<?x?xf32>
}
